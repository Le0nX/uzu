//! Inference for variable-option marker decision models (Julia and Laya).
use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use half::bf16;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    },
    config::model::decision_model::DecisionModelConfig,
    encodable_block::decision::{Decision, DecisionError},
    engine::Engine,
    safetensors::{ParameterLoader, ParameterLoaderError},
};
pub use crate::{config::decision::DecisionConfig, data_type::DataType};

pub struct DecisionModel<B: Backend> {
    context: Arc<B::Context>,
    decision: Decision<B>,
    config: DecisionConfig,
    data_type: DataType,
    tokenizer: Arc<Tokenizer>,
}
#[derive(Debug, Error)]
pub enum EngineLoadDecisionError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Configuration error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Parameter loader error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Decision model error: {0}")]
    Decision(#[from] DecisionError<B>),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}
impl<B: Backend> Engine<B> {
    pub fn load_decision_model(
        &self,
        path: &Path,
    ) -> Result<DecisionModel<B>, EngineLoadDecisionError<B>> {
        self.load_decision_model_with_data_type(path, DataType::BF16)
    }
    pub fn load_decision_model_with_data_type(
        &self,
        path: &Path,
        data_type: DataType,
    ) -> Result<DecisionModel<B>, EngineLoadDecisionError<B>> {
        let config: DecisionModelConfig =
            serde_json::from_reader(BufReader::new(File::open(path.join("config.json"))?))?;
        let weights = File::open(path.join("model.safetensors"))?;
        let loader = ParameterLoader::new(&weights, self.context.as_ref())?;
        let decision = Decision::new(
            data_type,
            &config.decision_config,
            &loader.tree().subtree("decision"),
            self.context.as_ref(),
        )?;
        loader.assert_all_tensors_read()?;
        let tokenizer = Arc::new(Tokenizer::from_file(path.join("tokenizer.json"))?);
        Ok(DecisionModel {
            context: self.context.clone(),
            decision,
            config: config.decision_config,
            data_type,
            tokenizer,
        })
    }
}
#[derive(Debug, Error)]
pub enum DecisionModelScoreError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decision model error: {0}")]
    Decision(#[from] DecisionError<B>),
    #[error("Input is empty")]
    EmptyInput,
    #[error("Input exceeds the model context ({actual} > {maximum}); no truncation was performed")]
    ContextExceeded {
        actual: usize,
        maximum: u32,
    },
    #[error("Question type {0} is outside the model vocabulary")]
    InvalidQuestionType(u32),
    #[error("Token id {0} is outside the model vocabulary")]
    InvalidToken(u32),
    #[error("At least one option marker is required")]
    EmptyMarkers,
    #[error("Option marker {0} is outside the input")]
    InvalidMarker(u32),
}
impl<B: Backend> DecisionModel<B> {
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn config(&self) -> &DecisionConfig {
        &self.config
    }
    /// Return one raw logit per supplied marker, in marker order. Input is a
    /// single unpadded sequence; question boundaries/softmax are caller-owned.
    pub fn score(
        &self,
        input: &[u32],
        question_type: u32,
        markers: &[u32],
    ) -> Result<Vec<f32>, DecisionModelScoreError<B>> {
        if input.is_empty() {
            return Err(DecisionModelScoreError::EmptyInput);
        }
        if input.len() > self.config.max_sequence_length as usize {
            return Err(DecisionModelScoreError::ContextExceeded {
                actual: input.len(),
                maximum: self.config.max_sequence_length,
            });
        }
        if question_type >= self.config.num_question_types {
            return Err(DecisionModelScoreError::InvalidQuestionType(question_type));
        }
        if let Some(&id) = input.iter().find(|&&id| id >= self.config.vocab_size) {
            return Err(DecisionModelScoreError::InvalidToken(id));
        }
        if markers.is_empty() {
            return Err(DecisionModelScoreError::EmptyMarkers);
        }
        if let Some(&id) = markers.iter().find(|&&id| id as usize >= input.len()) {
            return Err(DecisionModelScoreError::InvalidMarker(id));
        }
        let mut cb = self.context.create_command_buffer(None, None).map_err(DecisionModelScoreError::Backend)?;
        let tokens = cb.allocate_constant_from_slice(input).map_err(DecisionModelScoreError::Backend)?;
        let logits = self.decision.encode(&tokens, input.len() as u32, question_type, markers, &mut cb)?;
        let mut output = self
            .context
            .create_buffer(markers.len() * self.data_type.size_in_bytes())
            .map_err(DecisionModelScoreError::Backend)?;
        cb.encode_copy(&logits, &mut output);
        drop(logits);
        drop(tokens);
        cb.end_encoding().submit().wait_until_completed().map_err(DecisionModelScoreError::Backend)?;
        Ok(match self.data_type {
            DataType::BF16 => output.copyout::<bf16>().into_iter().map(bf16::to_f32).collect(),
            DataType::F32 => output.copyout::<f32>(),
            _ => unreachable!("validated at model construction"),
        })
    }
}
