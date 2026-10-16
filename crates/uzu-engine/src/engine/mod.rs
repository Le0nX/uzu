use std::{fs::File, io::BufReader, path::Path, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Context,
        capture_matcher::{CAPTURE_MATCHER, CaptureError, CaptureMatcherSpan},
    },
    config::{model::AnyModelConfig, token_mixer::AnyTokenMixerConfig},
    encodable_block::sampling::SamplingMethod,
};

pub mod classifier_model;
pub mod code_decision_model;
pub mod code_readout;
pub mod d3;
pub mod decision_model;
pub mod decision_request;
pub mod language_model;
pub mod modernbert_marker;
pub mod nimble;
pub mod typed_decision;

pub struct Engine<B: Backend> {
    _capture_span: Option<CaptureMatcherSpan>,
    context: Arc<B::Context>,
}

#[derive(Debug, Error)]
pub enum EngineNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Capture error: {0}")]
    Capture(#[from] CaptureError<B>),
}

impl<B: Backend> Engine<B> {
    pub fn new() -> Result<Arc<Self>, EngineNewError<B>> {
        let context = <B::Context as Context>::new().map_err(EngineNewError::Backend)?;
        let capture_span = CAPTURE_MATCHER.capture::<B>(&context, "")?;

        Ok(Arc::new(Self {
            _capture_span: capture_span,
            context,
        }))
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.context.peak_memory_usage()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    LanguageModel,
    Classifier,
    Decision,
}

#[derive(Debug, Error)]
pub enum ResolveModelTypeError {
    #[error("Unable to open model configuration: {0}")]
    UnableToOpenConfig(#[from] std::io::Error),
    #[error("Unable to deserialize model configuration: {0}")]
    UnableToDeserializeConfig(#[from] serde_json::Error),
}

pub fn resolve_model_type(model_path: &Path) -> Result<ModelType, ResolveModelTypeError> {
    Ok(match read_model_config(model_path)? {
        AnyModelConfig::LanguageModelConfig(_) => ModelType::LanguageModel,
        AnyModelConfig::ClassifierModelConfig(_) => ModelType::Classifier,
        AnyModelConfig::DecisionModelConfig(_) | AnyModelConfig::CodeDecisionModelConfig(_) => ModelType::Decision,
    })
}

pub fn resolve_model_sampling_method(model_path: &Path) -> Result<Option<SamplingMethod>, ResolveModelTypeError> {
    Ok(match read_model_config(model_path)? {
        AnyModelConfig::LanguageModelConfig(config) => Some(config.generation_config.default_sampling_method()),
        AnyModelConfig::ClassifierModelConfig(_)
        | AnyModelConfig::DecisionModelConfig(_)
        | AnyModelConfig::CodeDecisionModelConfig(_) => None,
    })
}

pub fn resolve_model_context_length(model_path: &Path) -> Result<Option<u32>, ResolveModelTypeError> {
    let transformer = match read_model_config(model_path)? {
        AnyModelConfig::LanguageModelConfig(config) => config.decoder_config.transformer_config,
        AnyModelConfig::ClassifierModelConfig(_) => return Ok(None),
        AnyModelConfig::DecisionModelConfig(config) => return Ok(Some(config.decision_config.max_sequence_length)),
        AnyModelConfig::CodeDecisionModelConfig(config) => return Ok(Some(config.max_sequence_length)),
    };
    Ok(transformer
        .layer_configs
        .iter()
        .filter(|layer| matches!(layer.mixer_config, AnyTokenMixerConfig::AttentionConfig(_)))
        .filter_map(|layer| layer.rope_config.as_ref().map(|rope| *rope.max_sequence_length()))
        .min())
}

fn read_model_config(model_path: &Path) -> Result<AnyModelConfig, ResolveModelTypeError> {
    let file = File::open(model_path.join("config.json"))?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
}
