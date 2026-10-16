//! Known arithmetic through the real decoder/readout path, without a checkpoint download.
#![cfg(feature = "cpu")]
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use half::bf16;
use serde_json::{Map, Value, json};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::Engine,
};

struct CheckModel {
    path: PathBuf,
}
impl BackendSelection for CheckModel {
    type Output = ();
    type Error = Box<dyn std::error::Error>;
    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        let engine = Engine::<B>::new().map_err(|e| e.to_string())?;
        let model = engine.load_language_model(&self.path).map_err(|e| e.to_string())?;
        let score = model.score_candidate_tokens(&[0, 1], &[5, 1, 3]).map_err(|e| e.to_string())?;
        assert_eq!(&*score, &[5.5, 1.5, 3.5]);
        assert_eq!(&*model.score_candidate_tokens(&[0, 1], &[5, 1]).map_err(|e| e.to_string())?, &[5.5, 1.5]);
        assert_eq!(&*model.score_candidate_tokens(&[0, 1], &[3]).map_err(|e| e.to_string())?, &[3.5]);
        assert!(model.score_candidate_tokens(&[], &[0]).is_err());
        assert!(model.score_candidate_tokens(&[1], &[6]).is_err());
        assert!(model.score_candidate_tokens(&[1], &[0, 0]).is_err());
        assert_eq!(
            &*model.score_candidate_tokens(&[1, 0], &[5, 1, 3]).map_err(|e| e.to_string())?,
            &[-5.5, -1.5, -3.5]
        );
        Ok(())
    }
}

#[test]
fn selected_readout_uses_last_prompt_token_and_keeps_candidate_order() {
    let path = std::env::temp_dir().join(format!(
        "uzu-candidate-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    let norm = json!({"epsilon":0.0,"scale_offset":null,"upcast_mode":"full_layer","subtract_mean":false,"has_scale":false,"has_biases":false});
    let config = json!({"type":"LanguageModelConfig","token_codec_config":{"type":"RawTextCodecConfig"},
        "generation_config":{"stop_token_ids":[],"temperature":null,"top_k":null,"top_p":null,"min_p":null,"banned_tokens":null,"repetition_penalty":null,"presence_penalty":null,"frequency_penalty":null,"suffix_repetition_length":null},"decoder_config":{"vocab_size":6,"embedding_norm_config":null,"ple_model_config":null,
        "embedding_config":{"type":"UntiedEmbeddingConfig","input_scale":null,"logit_scale":null,"logit_soft_cap":null},
        "transformer_config":{"model_dim":4,"hidden_dim":4,"layer_configs":[],"output_norm_config":norm}}});
    fs::write(path.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab((0..6).map(|i| (format!("t{i}"), i)).collect())
            .unk_token("t0".into())
            .build()
            .unwrap(),
    );
    tokenizer.save(path.join("tokenizer.json"), false).unwrap();
    let mut header = Map::<String, Value>::new();
    let mut data = Vec::new();
    for (name, values) in [
        (
            "decoder.embedding.input_embedding.weights",
            (0..24)
                .map(|i| {
                    if i / 4 == 1 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect::<Vec<_>>(),
        ),
        (
            "decoder.embedding.output_embedding.weights",
            (0..24)
                .map(|i| {
                    if i % 4 == 0 {
                        i as f32 / 4.0 + 0.5
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        ),
    ] {
        let start = data.len();
        for v in values {
            data.extend(bf16::from_f32(v).to_le_bytes());
        }
        header.insert(name.into(), json!({"dtype":"BF16","shape":[6,4],"data_offsets":[start,data.len()]}));
    }
    header.insert(
        "__metadata__".into(),
        json!({
        "decoder.embedding.input_embedding.spec":"{\"type\":\"FullPrecisionSpec\",\"layout\":\"input_output\"}",
        "decoder.embedding.output_embedding.spec":"{\"type\":\"FullPrecisionSpec\",\"layout\":\"output_input\"}"}),
    );
    // EmbeddingMatrix stores its vocabulary rows in input_output shape.
    header.get_mut("decoder.embedding.input_embedding.weights").unwrap()["shape"] = json!([6, 4]);
    let mut encoded = serde_json::to_vec(&header).unwrap();
    encoded.extend(std::iter::repeat_n(b' ', (8 - encoded.len() % 8) % 8));
    let mut file = (encoded.len() as u64).to_le_bytes().to_vec();
    file.extend(encoded);
    file.extend(data);
    fs::write(path.join("model.safetensors"), file).unwrap();
    select_backend(
        CheckModel {
            path: path.clone(),
        },
        "backend unavailable".into(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    fs::remove_dir_all(path).unwrap();
}
