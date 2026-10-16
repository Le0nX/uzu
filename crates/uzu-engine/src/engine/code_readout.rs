//! Explicit d3 / Perplexity prompt policies over the shared Qwen code-readout graph.
use std::{collections::HashSet, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::{
    code_decision_model::CodeDecisionModel,
    decision_request::{DecisionQuestion, DecisionQuestionKind, DecisionRequest},
    modernbert_marker::{contains_float, source_text},
    typed_decision::{DecisionAdapter, DecisionValues, ScoredDecisionQuestion},
};
use crate::backends::Backend;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CodeReadoutPolicy {
    pub architecture_id: String,
    pub adapter_id: String,
    pub model_id: String,
    pub revision: String,
    pub max_sequence_length: u32,
    pub temperature: f64,
    pub readout_dtype: String,
    pub codes: Box<[String]>,
    pub token_ids: Box<[u32]>,
    pub chat_prefix: String,
    pub chat_suffix: String,
}

#[derive(Debug, Error)]
pub enum CodeReadoutError {
    #[error("Invalid code-readout policy or request: {0}")]
    Invalid(String),
    #[error("context_exceeded: question {question_id}: {actual} > {maximum}; nothing truncated")]
    ContextExceeded {
        question_id: String,
        actual: usize,
        maximum: u32,
    },
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Inference: {0}")]
    Inference(String),
}

#[derive(Debug, Serialize)]
pub struct EncodedCodeQuestion {
    pub question_id: String,
    pub kind: DecisionQuestionKind,
    pub option_ids: Box<[String]>,
    pub input_ids: Box<[u32]>,
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Number(v) => v.as_f64() != Some(0.0),
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}

impl CodeReadoutPolicy {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
    pub fn validate(
        &self,
        tokenizer: &Tokenizer,
    ) -> Result<(), CodeReadoutError> {
        if self.architecture_id != "qwen_code_readout_v1"
            || !matches!(
                (self.adapter_id.as_str(), self.readout_dtype.as_str()),
                ("d3_code_v1", "float32") | ("pplx_code_v1", "bfloat16")
            )
            || self.model_id.is_empty()
            || self.revision.is_empty()
            || self.max_sequence_length == 0
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.codes.len() != 255
            || self.token_ids.len() != 255
            || self.chat_prefix.is_empty()
            || self.chat_suffix.is_empty()
            || self.token_ids.iter().collect::<HashSet<_>>().len() != 255
        {
            return Err(CodeReadoutError::Invalid(
                "unsupported architecture/encoding or invalid calibration/codebook".into(),
            ));
        }
        for (code, &id) in self.codes.iter().zip(self.token_ids.iter()) {
            if code.is_empty()
                || !code.bytes().all(|c| c.is_ascii_uppercase())
                || tokenizer.encode(code.as_str(), false)?.get_ids() != [id]
            {
                return Err(CodeReadoutError::Invalid("code/tokenizer mismatch".into()));
            }
        }
        Ok(())
    }
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        request: &DecisionRequest,
    ) -> Result<Box<[EncodedCodeQuestion]>, CodeReadoutError> {
        self.validate(tokenizer)?;
        if request.questions.is_empty() || contains_float(&request.state) {
            return Err(CodeReadoutError::Invalid(
                "questions required; serialize structured floats to source text".into(),
            ));
        }
        let state = match &request.state {
            Value::Null if self.adapter_id == "d3_code_v1" => "(empty)".into(),
            Value::String(v) if v.is_empty() && self.adapter_id == "d3_code_v1" => "(empty)".into(),
            _ => source_text(&request.state),
        };
        request
            .questions
            .iter()
            .map(|(name, q)| {
                let (instructions, texts): (&str, Vec<String>) = match q {
                    DecisionQuestion::Choice {
                        instructions,
                        criteria,
                    } => {
                        if criteria.keys().any(String::is_empty) || criteria.values().any(contains_float) {
                            return Err(CodeReadoutError::Invalid(
                                "empty option key or float description; serialize floats first".into(),
                            ));
                        }
                        (
                            instructions,
                            criteria
                                .iter()
                                .map(|(k, v)| {
                                    if v.is_null() {
                                        k.clone()
                                    } else {
                                        format!("{k}: {}", source_text(v))
                                    }
                                })
                                .collect(),
                        )
                    },
                    DecisionQuestion::Noul {
                        instructions,
                        criteria,
                    } => {
                        if criteria.as_ref().is_some_and(|c| {
                            c.keys().any(|k| k != "false" && k != "true") || c.values().any(contains_float)
                        }) {
                            return Err(CodeReadoutError::Invalid(
                                "noul criteria must describe false/true; serialize floats first".into(),
                            ));
                        }
                        (
                            instructions,
                            [("false", "No / false"), ("true", "Yes / true")]
                                .iter()
                                .map(|(k, fallback)| {
                                    criteria
                                        .as_ref()
                                        .and_then(|c| c.get(*k))
                                        .filter(|v| truthy(v))
                                        .map_or_else(|| (*fallback).into(), source_text)
                                })
                                .collect(),
                        )
                    },
                    DecisionQuestion::Score {
                        instructions,
                        criteria,
                    } => {
                        let texts: Vec<_> = criteria.iter().map(source_text).collect();
                        let d3 = self.adapter_id == "d3_code_v1";
                        if !(2..=if d3 {
                            10
                        } else {
                            255
                        })
                            .contains(&texts.len())
                            || (d3 && texts.iter().any(String::is_empty))
                            || (d3 && texts.iter().collect::<HashSet<_>>().len() != texts.len())
                            || criteria.iter().any(contains_float)
                        {
                            return Err(CodeReadoutError::Invalid(
                                "invalid score levels for this native adapter; serialize floats first".into(),
                            ));
                        }
                        (instructions, texts)
                    },
                };
                if name.is_empty() || texts.is_empty() || texts.len() > 255 {
                    return Err(CodeReadoutError::Invalid("question id required and 1..255 options supported".into()));
                }
                let instructions = if instructions.is_empty() {
                    "Choose the best matching option."
                } else {
                    instructions
                };
                let mut parts = vec![
                    "State:".to_string(),
                    state.clone(),
                    String::new(),
                    "Question:".into(),
                    instructions.into(),
                    String::new(),
                    "Options:".into(),
                ];
                parts.extend(self.codes.iter().zip(texts.iter()).map(|(code, text)| format!("{code}: {text}")));
                let closing = match self.adapter_id.as_str() {
                    "d3_code_v1" => "Reply with only the code of the best option.",
                    "pplx_code_v1" => "Return only the letter code of the best option.",
                    _ => unreachable!("policy validated before encoding"),
                };
                parts.extend([String::new(), closing.into()]);
                let prompt = format!("{}{}{}", self.chat_prefix, parts.join("\n"), self.chat_suffix);
                let ids = tokenizer.encode(prompt, false)?.get_ids().to_vec();
                if ids.len() > self.max_sequence_length as usize {
                    return Err(CodeReadoutError::ContextExceeded {
                        question_id: name.clone(),
                        actual: ids.len(),
                        maximum: self.max_sequence_length,
                    });
                }
                Ok(EncodedCodeQuestion {
                    question_id: name.clone(),
                    kind: q.kind(),
                    option_ids: q.option_ids(),
                    input_ids: ids.into(),
                })
            })
            .collect()
    }
}

pub struct CodeReadoutAdapter<B: Backend> {
    model: CodeDecisionModel<B>,
    policy: CodeReadoutPolicy,
}
impl<B: Backend> CodeReadoutAdapter<B> {
    pub fn new(
        model: CodeDecisionModel<B>,
        policy: CodeReadoutPolicy,
    ) -> Result<Self, CodeReadoutError> {
        policy.validate(model.tokenizer())?;
        if policy.max_sequence_length != model.max_context_length() || policy.readout_dtype != model.readout_dtype() {
            return Err(CodeReadoutError::Invalid("policy and graph context/readout precision disagree".into()));
        }
        Ok(Self {
            model,
            policy,
        })
    }
    pub fn model(&self) -> &CodeDecisionModel<B> {
        &self.model
    }
    pub fn policy(&self) -> &CodeReadoutPolicy {
        &self.policy
    }
}
impl<B: Backend> DecisionAdapter for CodeReadoutAdapter<B> {
    type Error = CodeReadoutError;
    fn score_questions(
        &self,
        request: &DecisionRequest,
    ) -> Result<Box<[ScoredDecisionQuestion]>, Self::Error> {
        self.policy
            .encode(self.model.tokenizer(), request)?
            .iter()
            .map(|q| {
                let rows: Box<[_]> = (0..q.option_ids.len() as u32).collect();
                let values =
                    self.model.score(&q.input_ids, &rows).map_err(|e| CodeReadoutError::Inference(e.to_string()))?;
                Ok(ScoredDecisionQuestion {
                    question_id: q.question_id.clone(),
                    kind: q.kind,
                    option_ids: q.option_ids.clone(),
                    values: DecisionValues::Logits {
                        values,
                        temperature: self.policy.temperature,
                    },
                    input_tokens: Some(q.input_ids.len()),
                })
            })
            .collect()
    }
    fn adapter_id(&self) -> Option<&str> {
        Some(&self.policy.adapter_id)
    }
    fn architecture_id(&self) -> Option<&str> {
        Some(&self.policy.architecture_id)
    }
    fn model_id(&self) -> Option<&str> {
        Some(&self.policy.model_id)
    }
    fn model_revision(&self) -> Option<&str> {
        Some(&self.policy.revision)
    }
}
