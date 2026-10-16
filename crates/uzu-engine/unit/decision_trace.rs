//! Diagnostic replica of the Decision path: copies per-layer tensors after the
//! actual kernels; not a timing benchmark. Arguments: MODEL INPUT OUTPUT [--f32].
use std::{error::Error, fs::File, path::PathBuf};

use half::bf16;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    backends::{
        Backend, BackendSelection,
        common::{
            Buffer, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::{activation_type::ActivationType, trie::TrieNode},
            kernel::{ActivationKernel, TensorAddScaleKernel},
        },
        select_backend,
    },
    config::model::decision_model::DecisionModelConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::Embedding,
        linear::Linear,
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
        transformer::Transformer,
    },
    safetensors::ParameterLoader,
};
#[derive(Deserialize)]
struct Input {
    input_ids: Vec<u32>,
    markers: Vec<u32>,
    question_type: u32,
}
struct Run {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    dtype: DataType,
}
impl BackendSelection for Run {
    type Output = ();
    type Error = Box<dyn Error>;
    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        if let Ok(layer) = std::env::var("UZU_TRACE_LAYER") {
            return trace_layer::<B>(&self, layer.parse()?);
        }
        let ctx = B::Context::new().map_err(|e| e.to_string())?;
        let conf: DecisionModelConfig = serde_json::from_reader(File::open(self.model.join("config.json"))?)?;
        let c = &conf.decision_config;
        let input: Input = serde_json::from_reader(File::open(self.input)?)?;
        let n = input.input_ids.len() as u32;
        let d = c.model_dim;
        let dt = self.dtype;
        let file = File::open(self.model.join("model.safetensors"))?;
        let loader = ParameterLoader::<B>::new(&file, ctx.as_ref()).map_err(|e| e.to_string())?;
        let tree = loader.tree().subtree("decision");
        let (embedding, _) =
            Embedding::new(c.vocab_size, d, dt, &c.embedding_config, &tree.subtree("embedding"), ctx.as_ref())
                .map_err(|e| e.to_string())?;
        let norm = |cfg, name| {
            Normalization::new(
                d,
                dt,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                cfg,
                &tree.subtree(name),
                ctx.as_ref(),
            )
            .map_err(|e| e.to_string())
        };
        let emb_norm = norm(&c.embedding_norm_config, "embedding_norm")?;
        let encoder = Transformer::new(dt, None, &c.encoder_config, &tree.subtree("encoder"), ctx.as_ref())
            .map_err(|e| e.to_string())?;
        let encoder = if dt == DataType::BF16 {
            encoder.with_bf16_rope_tables()
        } else {
            encoder
        };
        let (type_emb, _) = Embedding::new(
            c.num_question_types,
            d,
            dt,
            &c.type_embedding_config,
            &tree.subtree("type_embedding"),
            ctx.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        let head = Transformer::new(dt, None, &c.head_config, &tree.subtree("head"), ctx.as_ref())
            .map_err(|e| e.to_string())?;
        let scorer_norm = norm(&c.scorer_norm_config, "scorer_norm")?;
        let dense = <dyn Linear<B>>::new(d, d, dt, true, &tree.subtree("scorer_dense"), ctx.as_ref())
            .map_err(|e| e.to_string())?;
        let readout = <dyn Linear<B>>::new(d, 1, dt, true, &tree.subtree("scorer_readout"), ctx.as_ref())
            .map_err(|e| e.to_string())?;
        loader.assert_all_tensors_read().map_err(|e| e.to_string())?;
        let add =
            <B::Kernels as Kernels>::TensorAddScaleKernel::new(ctx.as_ref(), dt, false).map_err(|e| e.to_string())?;
        let act = <B::Kernels as Kernels>::ActivationKernel::new(ctx.as_ref(), dt, true).map_err(|e| e.to_string())?;
        let mut cb = ctx.create_command_buffer(None, None).map_err(|e| e.to_string())?;
        let mut captures: Vec<(String, B::GlobalBuffer)> = Vec::new();
        macro_rules! capture {
            ($name:expr,$buffer:expr) => {{
                let mut out = ctx.create_buffer($buffer.size()).map_err(|e| e.to_string())?;
                cb.encode_copy($buffer, &mut out);
                captures.push(($name.to_string(), out));
            }};
        }
        let ids = cb.allocate_constant_from_slice(&input.input_ids).map_err(|e| e.to_string())?;
        let embedded = embedding.encode_lookup(&ids, n, &mut cb).map_err(|e| e.to_string())?;
        let hidden =
            emb_norm.encode(&embedded, 0, n, None::<&mut B::ScratchBuffer>, &mut cb).map_err(|e| e.to_string())?;
        capture!("encoder_embedding", &hidden);
        let nodes = (0..n)
            .map(|i| TrieNode {
                trie_start: i,
                trie_end: n - 1,
                height: i,
            })
            .collect::<Vec<_>>();
        let topology = BatchTopology::new(&nodes, true);
        let layers = (0..c.encoder_config.layer_configs.len() as u32).collect::<Vec<_>>();
        let encoded = encoder
            .encode(hidden, None::<&B::ScratchBuffer>, &topology, Some((0..n).into()), Some(&layers), None, &mut cb)
            .map_err(|e| e.to_string())?;
        for (i, b) in encoded.hidden_features.as_ref().unwrap().iter().enumerate() {
            capture!(format!("encoder_layer_{i}"), b);
        }
        let hidden = encoded.output.unwrap();
        capture!("encoder_final", &hidden);
        drop(encoded.hidden_features);
        let type_id = cb.allocate_constant_from_slice(&[input.question_type]).map_err(|e| e.to_string())?;
        let type_hidden = type_emb.encode_lookup(&type_id, 1, &mut cb).map_err(|e| e.to_string())?;
        let mut typed = cb.allocate_scratch_for_shape(&[n, d], dt).map_err(|e| e.to_string())?;
        add.encode(Some(&hidden), &type_hidden, &mut typed, d, n * d, 1., &mut cb);
        capture!("head_input", &typed);
        drop(hidden);
        let layers = (0..c.head_config.layer_configs.len() as u32).collect::<Vec<_>>();
        let head_out = head
            .encode(typed, None::<&B::ScratchBuffer>, &topology, None, Some(&layers), None, &mut cb)
            .map_err(|e| e.to_string())?
            .hidden_features
            .unwrap();
        for (i, b) in head_out.iter().enumerate() {
            capture!(format!("head_layer_{i}"), b);
        }
        let hidden = head_out.last().unwrap();
        let row_bytes = d as usize * dt.size_in_bytes();
        let mut selected = cb.allocate_scratch(input.markers.len() * row_bytes).map_err(|e| e.to_string())?;
        for (row, &marker) in input.markers.iter().enumerate() {
            cb.encode_copy(
                hidden.subrange(marker as usize * row_bytes..(marker as usize + 1) * row_bytes),
                (&mut selected).subrange_mut(row * row_bytes..(row + 1) * row_bytes),
            );
        }
        capture!("markers_final", &selected);
        let count = input.markers.len() as u32;
        let normalized = scorer_norm
            .encode(&selected, 0, count, None::<&mut B::ScratchBuffer>, &mut cb)
            .map_err(|e| e.to_string())?;
        capture!("scorer_0", &normalized);
        let mut projected = dense.encode(normalized, count, &mut cb).map_err(|e| e.to_string())?;
        capture!("scorer_1", &projected);
        act.encode(
            None::<&B::ScratchBuffer>,
            &mut projected,
            count * d,
            if c.pytorch_bf16 {
                ActivationType::GELUExactPyTorchBF16
            } else {
                ActivationType::GELUExact
            },
            &mut cb,
        );
        capture!("scorer_2", &projected);
        let logits = readout.encode(projected, count, &mut cb).map_err(|e| e.to_string())?;
        capture!("scorer_3", &logits);
        drop(logits);
        drop(selected);
        drop(head_out);
        drop(embedded);
        drop(ids);
        drop(type_id);
        drop(type_hidden);
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        let mut tensors = Map::new();
        for (name, b) in captures {
            let v = if dt == DataType::BF16 {
                b.copyout::<bf16>().into_iter().map(bf16::to_f32).collect::<Vec<_>>()
            } else {
                b.copyout::<f32>()
            };
            tensors.insert(name, json!(v));
        }
        serde_json::to_writer(File::create(self.output)?, &Value::Object(tensors))?;
        Ok(())
    }
}
#[uzu_engine_macros::uzu_test]
#[ignore]
fn trace_converted_checkpoint_layers() {
    let run = Run {
        model: std::env::var("UZU_TRACE_MODEL").unwrap().into(),
        input: std::env::var("UZU_TRACE_INPUT").unwrap().into(),
        output: std::env::var("UZU_TRACE_OUTPUT").unwrap().into(),
        dtype: if std::env::var_os("UZU_TRACE_F32").is_some() {
            DataType::F32
        } else {
            DataType::BF16
        },
    };
    select_backend(run, "unknown backend".into()).unwrap();
}

fn trace_layer<B: Backend>(
    run: &Run,
    index: usize,
) -> Result<(), Box<dyn Error>> {
    use crate::encodable_block::{mixer::attention::rope::PrecalculatedRoPE, transformer_layer::TransformerLayer};
    let ctx = B::Context::new().map_err(|e| e.to_string())?;
    let conf: DecisionModelConfig = serde_json::from_reader(File::open(run.model.join("config.json"))?)?;
    let c = &conf.decision_config;
    let dt = run.dtype;
    let request: Input = serde_json::from_reader(File::open(&run.input)?)?;
    let n = request.input_ids.len() as u32;
    let d = c.model_dim;
    let values: Vec<f32> = serde_json::from_reader(File::open(std::env::var("UZU_TRACE_HIDDEN")?)?)?;
    let file = File::open(run.model.join("model.safetensors"))?;
    let loader = ParameterLoader::<B>::new(&file, ctx.as_ref()).map_err(|e| e.to_string())?;
    let cfg = &c.encoder_config.layer_configs[index];
    let layer = TransformerLayer::new(
        d,
        c.encoder_config.hidden_dim,
        c.encoder_config.layer_configs.len() as u32,
        index as u32,
        dt,
        cfg,
        &loader.tree().subtree(&format!("decision.encoder.layers.{index}")),
        ctx.as_ref(),
    )
    .map_err(|e| e.to_string())?;
    let mut cb = ctx.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    let mut captures: Vec<(String, B::GlobalBuffer)> = Vec::new();
    macro_rules! capture {
        ($name:expr,$buffer:expr) => {{
            let mut out = ctx.create_buffer($buffer.size()).map_err(|e| e.to_string())?;
            cb.encode_copy($buffer, &mut out);
            captures.push(($name.to_string(), out));
        }};
    }
    let data: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
    let data = cb.allocate_constant_from_slice(&data).map_err(|e| e.to_string())?;
    let mut hidden = cb.allocate_scratch(values.len() * dt.size_in_bytes()).map_err(|e| e.to_string())?;
    cb.encode_copy(&data, &mut hidden);
    let mut shortcut = cb.allocate_scratch(values.len() * dt.size_in_bytes()).map_err(|e| e.to_string())?;
    cb.encode_fill(&mut shortcut, 0);
    let norm = layer
        .pre_mixer_norm
        .as_ref()
        .unwrap()
        .encode(&hidden, 0, n, Some(&mut shortcut), &mut cb)
        .map_err(|e| e.to_string())?;
    drop(hidden);
    capture!("attn_norm", &norm);
    let nodes = (0..n)
        .map(|i| TrieNode {
            trie_start: i,
            trie_end: n - 1,
            height: i,
        })
        .collect::<Vec<_>>();
    let topology = BatchTopology::new(&nodes, true);
    let rope = PrecalculatedRoPE::<B>::precalculate_with_precision(
        cfg.rope_config.as_ref().unwrap(),
        &(0..n).collect::<Vec<_>>(),
        true,
        &mut cb,
    )
    .map_err(|e| e.to_string())?;
    let out = layer.mixer.encode(norm, Some(&rope), &topology, None, &mut cb).map_err(|e| e.to_string())?;
    capture!("attn", &out);
    drop(rope);
    let norm = layer.pre_mlp_norm.encode(&out, 0, n, Some(&mut shortcut), &mut cb).map_err(|e| e.to_string())?;
    capture!("mlp_norm", &norm);
    capture!("mlp_input", &shortcut);
    drop(out);
    let debug_loader = ParameterLoader::<B>::new(&file, ctx.as_ref()).map_err(|e| e.to_string())?;
    let debug_tree = debug_loader.tree().subtree(&format!("decision.encoder.layers.{index}.mlp"));
    let hd = c.encoder_config.hidden_dim;
    let up = <dyn Linear<B>>::new(d, 2 * hd, dt, false, &debug_tree.subtree("up_projection"), ctx.as_ref())
        .map_err(|e| e.to_string())?;
    let mut debug_norm = cb.allocate_scratch(norm.size()).map_err(|e| e.to_string())?;
    cb.encode_copy(&norm, &mut debug_norm);
    let projected = up.encode(debug_norm, n, &mut cb).map_err(|e| e.to_string())?;
    capture!("mlp_up", &projected);
    use crate::backends::common::kernel::{GatedActMul, GatedActMulSettings};
    let gate = GatedActMul::<B>::full_precision(
        ctx.as_ref(),
        dt,
        true,
        false,
        GatedActMulSettings {
            activation_alpha: None,
            gate_clipping: Default::default(),
            value_clipping: Default::default(),
        },
    )
    .map_err(|e| e.to_string())?;
    let mut gated = cb.allocate_scratch_for_shape(&[n, hd], dt).map_err(|e| e.to_string())?;
    gate.encode_fp(
        &projected,
        None::<&B::ScratchBuffer>,
        &mut gated,
        None::<&B::GlobalBuffer>,
        hd,
        n,
        0,
        0,
        if c.pytorch_bf16 {
            ActivationType::GELUExactPyTorchBF16
        } else {
            ActivationType::GELUExact
        },
        &mut cb,
    );
    capture!("mlp_gate", &gated);
    drop(gated);
    drop(projected);
    let mlp = layer.mlp.encode(norm, n, &mut cb).map_err(|e| e.to_string())?;
    capture!("mlp", &mlp);
    let add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(ctx.as_ref(), dt, false).map_err(|e| e.to_string())?;
    let mut result = cb.allocate_scratch(mlp.size()).map_err(|e| e.to_string())?;
    add.encode(Some(&shortcut), &mlp, &mut result, n * d, n * d, 1., &mut cb);
    capture!("layer_output", &result);
    drop(result);
    drop(mlp);
    drop(shortcut);
    drop(data);
    cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let mut tensors = Map::new();
    for (name, b) in captures {
        let v = b.copyout::<bf16>().into_iter().map(bf16::to_f32).collect::<Vec<_>>();
        tensors.insert(name, json!(v));
    }
    serde_json::to_writer(File::create(&run.output)?, &Value::Object(tensors))?;
    Ok(())
}
