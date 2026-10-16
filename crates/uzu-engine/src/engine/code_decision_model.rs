//! Separate code readout over a fresh, whole-prompt Qwen decision backbone.
use std::{collections::HashSet, fs::File, io::BufReader, path::Path, sync::Arc};

use half::bf16;
use tokenizers::Tokenizer;

use super::Engine;
use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    config::{model::code_decision_model::CodeDecisionModelConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::Decoder,
        linear::{Linear, LinearMatmul},
    },
    safetensors::ParameterLoader,
};

pub struct CodeDecisionModel<B: Backend> {
    context: Arc<B::Context>,
    decoder: Decoder<B>,
    readout: LinearMatmul<B>,
    config: CodeDecisionModelConfig,
    tokenizer: Arc<Tokenizer>,
}

impl<B: Backend> Engine<B> {
    pub fn load_code_decision_model(
        &self,
        path: &Path,
    ) -> Result<CodeDecisionModel<B>, Box<dyn std::error::Error>> {
        let config: CodeDecisionModelConfig =
            serde_json::from_reader(BufReader::new(File::open(path.join("config.json"))?))?;
        if config.architecture_id != "qwen_code_readout_v1"
            || config.readout_rows != 255
            || config.max_sequence_length == 0
            || config.decoder_config.ple_model_config.is_some()
            || config.decoder_config.embedding_norm_config.is_some()
        {
            return Err("unsupported code-decision graph or invalid dimensions".into());
        }
        for layer in &config.decoder_config.transformer_config.layer_configs {
            if layer.mixer_conv_config.is_some() || layer.mlp_conv_config.is_some() {
                return Err("code-decision execution tiles do not support convolutional transformer branches".into());
            }
            match &layer.mixer_config {
                AnyTokenMixerConfig::AttentionConfig(a)
                    if !a.is_causal && a.sliding_window_size.is_none() && !a.is_kv_sharing && !a.bf16_eager => {},
                AnyTokenMixerConfig::DeltaNetConfig(_) => {},
                _ => {
                    return Err(
                        "qwen_code_readout_v1 requires noncausal full attention and native causal DeltaNet".into()
                    );
                },
            }
        }
        let readout_type = match config.readout_dtype.as_str() {
            "float32" => DataType::F32,
            "bfloat16" => DataType::BF16,
            _ => return Err("code-readout adapter requires FP32 or BF16 readout".into()),
        };
        let file = File::open(path.join("model.safetensors"))?;
        let loader = ParameterLoader::new(&file, self.context.as_ref()).map_err(|e| e.to_string())?;
        let decoder = Decoder::new(
            DataType::BF16,
            &config.decoder_config,
            &loader.tree().subtree("decoder"),
            self.context.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        let tree = loader.tree().subtree("readout").subtree("weights");
        let readout = LinearMatmul::load(
            config.decoder_config.transformer_config.model_dim,
            config.readout_rows,
            readout_type,
            DataType::BF16,
            readout_type,
            tree.metadata("spec").map_err(|e| e.to_string())?,
            None,
            &tree,
            None,
            self.context.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        loader.assert_all_tensors_read().map_err(|e| e.to_string())?;
        let tokenizer = Arc::new(Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| e.to_string())?);
        Ok(CodeDecisionModel {
            context: self.context.clone(),
            decoder,
            readout,
            config,
            tokenizer,
        })
    }
}

impl<B: Backend> CodeDecisionModel<B> {
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn max_context_length(&self) -> u32 {
        self.config.max_sequence_length
    }
    pub fn readout_dtype(&self) -> &str {
        &self.config.readout_dtype
    }
    pub fn score(
        &self,
        input: &[u32],
        rows: &[u32],
    ) -> Result<Box<[f32]>, Box<dyn std::error::Error>> {
        if input.is_empty() || rows.is_empty() || input.len() > u32::MAX as usize {
            return Err("nonempty prompt and readout rows required".into());
        }
        if input.len() > self.config.max_sequence_length as usize {
            return Err(format!(
                "context_exceeded: {} > {}; nothing truncated",
                input.len(),
                self.config.max_sequence_length
            )
            .into());
        }
        if input.iter().any(|&id| id >= self.config.decoder_config.vocab_size)
            || rows.iter().any(|&r| r >= self.config.readout_rows)
            || rows.iter().collect::<HashSet<_>>().len() != rows.len()
        {
            return Err("token outside vocabulary, invalid or duplicate readout row".into());
        }
        // Whole-prompt layer execution preserves bidirectional semantics while
        // releasing idle temporaries and each layer's prepared KV pages.
        let nodes: Box<[_]> = (0..input.len() as u32)
            .map(|i| TrieNode {
                trie_start: i,
                trie_end: input.len() as u32 - 1,
                height: i,
            })
            .collect();
        let saved = self.decoder.encode_code_prompt(input, &BatchTopology::new(&nodes, true), self.context.as_ref())?;
        let mut cb =
            self.context.create_command_buffer(Some("code decision readout"), None).map_err(|e| e.to_string())?;
        let mut hidden = cb.allocate_scratch(saved.size()).map_err(|e| e.to_string())?;
        cb.encode_copy(&saved, &mut hidden);
        // Preserve the head's native output rounding: d3 FP32, Perplexity BF16.
        // Both accumulate in FP32; BF16 logits are rounded before host softmax.
        let logits = self.readout.encode(hidden, 1, &mut cb).map_err(|e| e.to_string())?;
        let mut buffer = self
            .context
            .create_buffer(
                self.config.readout_rows as usize
                    * if self.config.readout_dtype == "float32" {
                        4
                    } else {
                        2
                    },
            )
            .map_err(|e| e.to_string())?;
        cb.encode_copy(&logits, &mut buffer);
        drop(logits);
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        let all = if self.config.readout_dtype == "float32" {
            buffer.copyout::<f32>()
        } else {
            buffer.copyout::<bf16>().into_iter().map(bf16::to_f32).collect()
        };
        drop(saved);
        self.context.release_cached_buffers();
        Ok(rows.iter().map(|&r| all[r as usize]).collect())
    }
}

#[cfg(test)]
#[path = "../../unit/d3_trace.rs"]
pub(crate) mod trace_d3;
