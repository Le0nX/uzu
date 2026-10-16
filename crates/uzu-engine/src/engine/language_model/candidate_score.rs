//! Bounded option scoring through the existing causal decoder and readout gather.
use std::collections::HashSet;

use derive_more::Debug;
use half::bf16;
use thiserror::Error;

use super::LanguageModel;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::DecoderError},
};

#[derive(Debug, Error)]
pub enum CandidateScoreError<B: Backend> {
    #[error("Invalid candidate-scoring request: {0}")]
    Invalid(String),
    #[error("context_exceeded: {actual} tokens > {maximum}; nothing was truncated")]
    ContextExceeded {
        actual: usize,
        maximum: u32,
    },
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
}

impl<B: Backend> LanguageModel<B> {
    pub fn score_candidate_tokens(
        &self,
        input: &[u32],
        candidates: &[u32],
    ) -> Result<Box<[f32]>, CandidateScoreError<B>> {
        let vocabulary = self.decoder.embedding().vocab_size();
        if input.is_empty() || candidates.is_empty() || input.len() > u32::MAX as usize {
            return Err(CandidateScoreError::Invalid("nonempty prompt and candidate tokens are required".into()));
        }
        if let Some(maximum) = self.max_context_length()
            && input.len() > maximum as usize
        {
            return Err(CandidateScoreError::ContextExceeded {
                actual: input.len(),
                maximum,
            });
        }
        if input.iter().chain(candidates).any(|&id| id >= vocabulary)
            || candidates.iter().collect::<HashSet<_>>().len() != candidates.len()
        {
            return Err(CandidateScoreError::Invalid("token outside vocabulary or duplicate candidate token".into()));
        }
        let context = self.engine.context.as_ref();
        let mut state =
            self.decoder.create_empty_state(Some(input.len() as u32), context).map_err(CandidateScoreError::Backend)?;
        state.prepare(0, input.len() as u32).map_err(CandidateScoreError::Backend)?;
        let mut cb = context
            .create_command_buffer(Some("typed candidate scoring"), None)
            .map_err(CandidateScoreError::Backend)?;
        let tokens = cb.allocate_constant_from_slice(input).map_err(CandidateScoreError::Backend)?;
        let selected = cb.allocate_constant_from_slice(candidates).map_err(CandidateScoreError::Backend)?;
        let nodes: Box<[_]> = (0..input.len() as u32)
            .map(|i| TrieNode {
                trie_start: i,
                trie_end: input.len() as u32 - 1,
                height: i,
            })
            .collect();
        let output = self.decoder.encode_candidates(
            &tokens,
            &BatchTopology::new(&nodes, true),
            &selected,
            &mut state,
            &mut cb,
        )?;
        let logits = output.logits.expect("candidate scoring always requests a readout");
        let mut result = context
            .create_buffer(candidates.len() * DataType::BF16.size_in_bytes())
            .map_err(CandidateScoreError::Backend)?;
        cb.encode_copy(&logits, &mut result);
        drop(output.hidden_features);
        drop(output.final_hidden);
        drop(logits);
        drop(tokens);
        drop(selected);
        cb.end_encoding().submit().wait_until_completed().map_err(CandidateScoreError::Backend)?;
        Ok(result.copyout::<bf16>().into_iter().map(bf16::to_f32).collect())
    }
}
