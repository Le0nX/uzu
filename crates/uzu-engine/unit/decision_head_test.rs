//! A tiny independently calculated head checks residuals, biases, ReLU and the
//! absence of a final normalization through the actual Transformer path.
use std::{
    fs::{File, remove_file},
    io::Write,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::trie::TrieNode,
        },
        cpu::Cpu,
    },
    config::transformer::TransformerConfig,
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, transformer::Transformer},
    safetensors::ParameterLoader,
};

#[uzu_test]
fn test_decision_head_residual_bias_relu() {
    let norm = json!({"epsilon":1e-5,"scale_offset":null,"upcast_mode":"only_normalization","subtract_mean":true,"has_scale":true,"has_biases":true});
    let config:TransformerConfig=serde_json::from_value(json!({
        "model_dim":4,"hidden_dim":4,
        "output_norm_config":{"epsilon":1e-5,"scale_offset":null,"upcast_mode":"only_normalization","subtract_mean":true,"has_scale":false,"has_biases":false},
        "layer_configs":[{
            "pre_mixer_norm_config":norm,"mixer_conv_config":null,"post_mixer_norm_config":null,
            "pre_mlp_norm_config":norm,"mlp_conv_config":null,"post_mlp_norm_config":null,
            "hidden_dim":null,"ple_config":null,"has_post_layer_scalar":false,"kv_source_layer_index":null,"rope_config":null,
            "mixer_config":{"type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
                "query_norm_config":null,"key_norm_config":null,"num_heads":1,"num_groups":1,"head_dim":64,
                "is_causal":false,"scale":null,"sliding_window_size":null,"logit_soft_cap":null,"has_sinks":false,
                "has_qkvg_biases":true,"has_out_biases":true,"has_gate":false,"normalize_values":false,"is_kv_sharing":false},
            "mlp_config":{"type":"FeedForwardMLPConfig","linear_config":{},"activation":{"type":"ReLU"},"has_up_biases":true,"has_down_biases":true}
        }]
    })).unwrap();
    let mut header = Map::<String, Value>::new();
    let mut metadata = Map::<String, Value>::new();
    let mut data = Vec::<u8>::new();
    let mut tensor = |name: &str, shape: &[usize], values: &[f32]| {
        let start = data.len();
        for v in values {
            data.extend(v.to_le_bytes());
        }
        header.insert(name.into(), json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}));
    };
    let identity = (0..16)
        .map(|i| {
            if i / 4 == i % 4 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    for name in ["pre_mixer_norm", "pre_mlp_norm"] {
        tensor(&format!("layers.0.{name}.scales"), &[4], &[1.0; 4]);
        tensor(&format!("layers.0.{name}.biases"), &[4], &[0.0; 4]);
    }
    for (name, rows, cols, weights, bias) in [
        ("mixer.qkvg_projection", 192, 4, vec![0.0; 768], vec![0.0; 192]),
        ("mixer.out_projection", 4, 64, vec![0.0; 256], vec![0.1, 0.2, 0.3, 0.4]),
        ("mlp.up_projection", 4, 4, identity.clone(), vec![0.05; 4]),
        ("mlp.down_projection", 4, 4, identity, vec![0.2; 4]),
    ] {
        tensor(&format!("layers.0.{name}.weights.weights"), &[rows, cols], &weights);
        tensor(&format!("layers.0.{name}.biases"), &[rows], &bias);
        metadata.insert(
            format!("layers.0.{name}.weights.spec"),
            json!(r#"{"type":"FullPrecisionSpec","layout":"output_input"}"#),
        );
    }
    drop(tensor);
    header.insert("__metadata__".into(), Value::Object(metadata));
    let path = std::env::temp_dir().join(format!(
        "uzu-decision-head-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut file = File::create(&path).unwrap();
    let bytes = serde_json::to_vec(&header).unwrap();
    file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&bytes).unwrap();
    file.write_all(&data).unwrap();
    drop(file);
    let file = File::open(&path).unwrap();
    let context = <Cpu as crate::backends::common::Backend>::Context::new().unwrap();
    let loader = ParameterLoader::<Cpu>::new(&file, context.as_ref()).unwrap();
    let transformer = Transformer::<Cpu>::new(DataType::F32, None, &config, &loader.tree(), context.as_ref()).unwrap();
    loader.assert_all_tensors_read().unwrap();
    let mut cb = context.create_command_buffer(None, None).unwrap();
    let input_values = [-1.0f32, 0.0, 1.0, 2.0, 4.0, 3.0, 2.0, 1.0];
    let constants = cb.allocate_constant_from_slice(&input_values).unwrap();
    let mut input = cb.allocate_scratch(32).unwrap();
    cb.encode_copy(&constants, &mut input);
    let nodes = [
        TrieNode {
            trie_start: 0,
            trie_end: 1,
            height: 0,
        },
        TrieNode {
            trie_start: 1,
            trie_end: 1,
            height: 1,
        },
    ];
    let features = transformer
        .encode(
            input,
            None::<&<Cpu as crate::backends::common::Backend>::ScratchBuffer>,
            &BatchTopology::new(&nodes, true),
            None,
            Some(&[0]),
            None,
            &mut cb,
        )
        .unwrap()
        .hidden_features
        .unwrap();
    let mut output = context.create_buffer(32).unwrap();
    cb.encode_copy(&features[0], &mut output);
    drop(features);
    drop(constants);
    cb.end_encoding().submit().wait_until_completed().unwrap();
    let actual = output.copyout::<f32>();
    for row in 0..2 {
        let h = (0..4).map(|i| input_values[row * 4 + i] + [0.1, 0.2, 0.3, 0.4][i]).collect::<Vec<_>>();
        let mean = h.iter().sum::<f32>() / 4.0;
        let variance = h.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        for i in 0..4 {
            let expected = h[i] + ((h[i] - mean) / (variance + 1e-5).sqrt() + 0.05).max(0.0) + 0.2;
            assert!(
                (actual[row * 4 + i] - expected).abs() < 2e-5,
                "row={row},col={i}: {} vs {expected}",
                actual[row * 4 + i]
            );
        }
    }
    drop(loader);
    drop(file);
    remove_file(path).unwrap();
}
