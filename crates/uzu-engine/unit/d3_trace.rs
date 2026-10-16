//! Opt-in layer/operator trace, with copies from the actual encoded graph.

use super::*;
use crate::backends::{BackendSelection, common::Buffer, select_backend};
thread_local! { static SAVES: std::cell::RefCell<Vec<Box<dyn FnOnce()>>> = std::cell::RefCell::new(Vec::new()); }
pub(crate) fn capture<B: Backend>(
    name: &str,
    buffer: impl BufferRef<Backend = B>,
    cb: &mut <B::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding,
) {
    use crate::backends::common::CommandBufferEncoding;
    if std::env::var_os("UZU_TRACE_LAYER").is_none() {
        return;
    }
    // Keep a persistent copy until the caller has submitted and waited.
    let mut out = cb.context().create_buffer(buffer.size()).ok().unwrap();
    cb.encode_copy(buffer, &mut out);
    let path = std::path::PathBuf::from(std::env::var("UZU_TRACE_OUTPUT").unwrap()).join(format!("{name}.bf16"));
    SAVES.with(|s| {
        s.borrow_mut().push(Box::new(move || {
            let bytes: Vec<u8> = out.copyout::<half::bf16>().iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            std::fs::write(path, bytes).unwrap();
        }))
    });
}
struct Run;
impl BackendSelection for Run {
    type Output = ();
    type Error = Box<dyn std::error::Error>;
    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        let path = std::env::var("UZU_TRACE_MODEL")?;
        let output = std::path::PathBuf::from(std::env::var("UZU_TRACE_OUTPUT")?);
        std::fs::create_dir_all(&output)?;
        let input: Vec<u32> = serde_json::from_reader(File::open(std::env::var("UZU_TRACE_INPUT")?)?)?;
        if let Ok(index) = std::env::var("UZU_TRACE_LAYER") {
            return trace_layer::<B>(&path, &output, &input, index.parse()?);
        }
        let engine = Engine::<B>::new().map_err(|e| e.to_string())?;
        let model = engine.load_code_decision_model(Path::new(&path))?;
        let ctx = model.context.as_ref();
        let n = input.len() as u32;
        let mut state = model.decoder.create_empty_state(Some(n), ctx).map_err(|e| e.to_string())?;
        state.prepare(0, n).map_err(|e| e.to_string())?;
        let mut cb = ctx.create_command_buffer(None, None).map_err(|e| e.to_string())?;
        let tokens = cb.allocate_constant_from_slice(&input).map_err(|e| e.to_string())?;
        let nodes: Vec<_> = (0..n)
            .map(|i| TrieNode {
                trie_start: i,
                trie_end: n - 1,
                height: i,
            })
            .collect();
        let layers: Vec<_> = (0..model.config.decoder_config.transformer_config.layer_configs.len() as u32).collect();
        let encoded = model
            .decoder
            .encode(&tokens, &BatchTopology::new(&nodes, true), None, Some(&layers), &mut state, &mut cb)
            .map_err(|e| e.to_string())?;
        let mut copies = Vec::new();
        for b in encoded.hidden_features.as_ref().unwrap().iter() {
            let mut out = ctx.create_buffer(b.size()).map_err(|e| e.to_string())?;
            cb.encode_copy(b, &mut out);
            copies.push(out);
        }
        drop(encoded);
        drop(tokens);
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        for (i, b) in copies.into_iter().enumerate() {
            let values = b.copyout::<half::bf16>();
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            std::fs::write(output.join(format!("layer_{i:02}.bf16")), bytes)?;
        }
        Ok(())
    }
}
fn trace_layer<B: Backend>(
    path: &str,
    output: &Path,
    input: &[u32],
    index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        encodable_block::{mixer::attention::rope::PrecalculatedRoPE, transformer_layer::TransformerLayer},
        utils::maybe_mut::MaybeMut,
    };
    let ctx = B::Context::new().map_err(|e| e.to_string())?;
    let config: CodeDecisionModelConfig = serde_json::from_reader(File::open(Path::new(path).join("config.json"))?)?;
    let c = &config.decoder_config.transformer_config;
    let file = File::open(Path::new(path).join("model.safetensors"))?;
    let loader = ParameterLoader::<B>::new(&file, ctx.as_ref()).map_err(|e| e.to_string())?;
    let cfg = &c.layer_configs[index];
    let layer = TransformerLayer::new(
        c.model_dim,
        c.hidden_dim,
        c.layer_configs.len() as u32,
        0,
        DataType::BF16,
        cfg,
        &loader.tree().subtree(&format!("decoder.transformer.layers.{index}")),
        ctx.as_ref(),
    )
    .map_err(|e| e.to_string())?;
    let raw = std::fs::read(std::env::var("UZU_TRACE_HIDDEN")?)?;
    let data: Vec<half::bf16> =
        raw.chunks_exact(2).map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))).collect();
    let n = input.len() as u32;
    let mut state = layer.mixer.create_empty_state(Some(n), ctx.as_ref()).map_err(|e| e.to_string())?;
    state.prepare(0, n).map_err(|e| e.to_string())?;
    let mut cb = ctx.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    let data = cb.allocate_constant_from_slice(&data).map_err(|e| e.to_string())?;
    let mut hidden = cb.allocate_scratch(raw.len()).map_err(|e| e.to_string())?;
    cb.encode_copy(&data, &mut hidden);
    let mut shortcut = cb.allocate_scratch(raw.len()).map_err(|e| e.to_string())?;
    let mut captures = Vec::new();
    macro_rules! capture {
        ($name:expr,$b:expr) => {{
            let mut b = ctx.create_buffer($b.size()).map_err(|e| e.to_string())?;
            cb.encode_copy($b, &mut b);
            captures.push(($name, b));
        }};
    }
    let norm = layer
        .pre_mixer_norm
        .as_ref()
        .unwrap()
        .encode(&hidden, 0, n, Some(&mut shortcut), &mut cb)
        .map_err(|e| e.to_string())?;
    capture!("input_layernorm", &norm);
    drop(hidden);
    let nodes: Vec<_> = (0..n)
        .map(|i| TrieNode {
            trie_start: i,
            trie_end: n - 1,
            height: i,
        })
        .collect();
    let rope = cfg
        .rope_config
        .as_ref()
        .map(|r| PrecalculatedRoPE::<B>::precalculate(r, &(0..n).collect::<Vec<_>>(), &mut cb))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mixer = layer
        .mixer
        .encode(norm, rope.as_ref(), &BatchTopology::new(&nodes, true), Some(MaybeMut::Mut(state.as_mut())), &mut cb)
        .map_err(|e| e.to_string())?;
    capture!("mixer", &mixer);
    let norm = layer.pre_mlp_norm.encode(&mixer, 0, n, Some(&mut shortcut), &mut cb).map_err(|e| e.to_string())?;
    capture!("post_attention_layernorm", &norm);
    drop(mixer);
    let mlp = layer.mlp.encode(norm, n, &mut cb).map_err(|e| e.to_string())?;
    capture!("mlp", &mlp);
    drop(mlp);
    drop(shortcut);
    drop(data);
    drop(rope);
    cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    SAVES.with(|s| {
        for save in s.borrow_mut().drain(..) {
            save();
        }
    });
    for (name, b) in captures {
        let bytes: Vec<u8> = b.copyout::<half::bf16>().iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        std::fs::write(output.join(format!("layer_{index:02}_{name}.bf16")), bytes)?;
    }
    Ok(())
}
#[uzu_engine_macros::uzu_test]
#[ignore]
fn trace_full_d3_layers() {
    select_backend(Run, "unknown backend".into()).unwrap();
}
