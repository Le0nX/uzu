//! Causal candidate-token scoring. Release prompt strings and codebook are
//! imported from the checkpoint contract, not embedded in the engine license.
use std::{collections::HashSet, path::Path};

use derive_more::Debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokenizers::Tokenizer;

use super::{
    decision_request::{DecisionQuestion, DecisionQuestionKind, DecisionRequest},
    language_model::LanguageModel,
    modernbert_marker::{contains_float, json_text, source_text},
    typed_decision::{DecisionAdapter, DecisionValues, ScoredDecisionQuestion},
};
use crate::backends::Backend;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NimblePolicy {
    pub architecture_id: String,
    pub adapter_id: String,
    pub max_sequence_length: u32,
    pub temperature: f64,
    pub candidate_codes: Box<[String]>,
    pub candidate_token_ids: Box<[u32]>,
    pub chat_prefix_letter: String,
    pub chat_prefix_short: String,
    pub chat_suffix: String,
}

#[derive(Debug, Error)]
pub enum NimbleEncodeError {
    #[error("Invalid Nimble input or policy: {0}")]
    Invalid(String),
    #[error("context_exceeded: question {question_id}: {actual} tokens > {maximum}; nothing was truncated")]
    ContextExceeded {
        question_id: String,
        actual: usize,
        maximum: u32,
    },
    #[error("Tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

#[derive(Debug, Clone, Serialize)]
pub struct EncodedNimbleQuestion {
    pub question_id: String,
    pub kind: DecisionQuestionKind,
    pub option_ids: Box<[String]>,
    pub input_ids: Box<[u32]>,
    pub candidate_token_ids: Box<[u32]>,
}

fn safe_json(value: &Value) -> String {
    json_text(value).replace('<', "\\u003c").replace('>', "\\u003e")
}

impl NimblePolicy {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    pub fn validate(
        &self,
        tokenizer: &Tokenizer,
    ) -> Result<(), NimbleEncodeError> {
        if self.architecture_id != "causal_token_readout_v1"
            || self.adapter_id != "nimble_v3"
            || self.max_sequence_length == 0
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.candidate_codes.len() != 255
            || self.candidate_token_ids.len() != 255
            || self.candidate_token_ids.iter().collect::<HashSet<_>>().len() != 255
            || self.chat_prefix_letter.is_empty()
            || self.chat_prefix_short.is_empty()
            || self.chat_suffix.is_empty()
        {
            return Err(NimbleEncodeError::Invalid(
                "unsupported architecture/release policy or invalid codebook".into(),
            ));
        }
        for (code, &id) in self.candidate_codes.iter().zip(self.candidate_token_ids.iter()) {
            if code.is_empty()
                || !code.bytes().all(|b| b.is_ascii_uppercase())
                || tokenizer.encode(code.as_str(), false)?.get_ids() != [id]
            {
                return Err(NimbleEncodeError::Invalid("candidate codebook does not match the tokenizer".into()));
            }
        }
        Ok(())
    }

    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        request: &DecisionRequest,
    ) -> Result<Box<[EncodedNimbleQuestion]>, NimbleEncodeError> {
        self.validate(tokenizer)?;
        if request.questions.is_empty() || contains_float(&request.state) {
            return Err(NimbleEncodeError::Invalid(
                "questions required; serialize floats as source text before inference".into(),
            ));
        }
        let context = match &request.state {
            Value::String(text) if text.trim().is_empty() => json_text(&request.state),
            _ => source_text(&request.state),
        };
        let mut fields = Vec::new();
        let mut max_choices = 0;
        for (name, question) in &request.questions {
            let (instructions, entries): (&str, Vec<(String, Value, Option<Value>)>) = match question {
                DecisionQuestion::Choice {
                    instructions,
                    criteria,
                } => (
                    instructions,
                    criteria.iter().map(|(k, v)| (k.clone(), json!(k), (!v.is_null()).then(|| v.clone()))).collect(),
                ),
                DecisionQuestion::Score {
                    instructions,
                    criteria,
                } => (
                    instructions,
                    criteria
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), json!(i.to_string()), Some(v.clone())))
                        .collect(),
                ),
                DecisionQuestion::Noul {
                    instructions,
                    criteria,
                } => {
                    if criteria.as_ref().is_some_and(|c| c.keys().any(|k| k != "false" && k != "true")) {
                        return Err(NimbleEncodeError::Invalid("noul criteria keys must be false/true".into()));
                    }
                    (
                        instructions,
                        [("false", false), ("true", true)]
                            .into_iter()
                            .map(|(k, v)| {
                                (
                                    k.into(),
                                    json!(v),
                                    criteria.as_ref().and_then(|c| c.get(k)).filter(|v| !v.is_null()).cloned(),
                                )
                            })
                            .collect(),
                    )
                },
            };
            if name.trim().is_empty()
                || instructions.trim().is_empty()
                || entries.is_empty()
                || entries.len() > 255
                || entries.iter().any(|(k, _, d)| k.trim().is_empty() || d.as_ref().is_some_and(contains_float))
            {
                return Err(NimbleEncodeError::Invalid(
                    "nonempty names/instructions and 1..255 choices required; serialize float descriptions first"
                        .into(),
                ));
            }
            max_choices = max_choices.max(entries.len());
            let choices: Vec<Value> = entries
                .iter()
                .zip(self.candidate_codes.iter())
                .map(|((_, value, description), code)| {
                    let mut item = json!({"code":code,"value":value});
                    if let Some(description) = description {
                        item.as_object_mut().unwrap().insert("description".into(), json!(source_text(description)));
                    }
                    item
                })
                .collect();
            fields.push(json!({"name":name,"description":instructions,"choices":choices}));
        }
        let body = safe_json(&json!({"context":context,"schema":fields}));
        let prefix = if max_choices > 26 {
            &self.chat_prefix_short
        } else {
            &self.chat_prefix_letter
        };
        request
            .questions
            .iter()
            .map(|(name, question)| {
                let prompt =
                    format!("{prefix}{body}\n\nRequested field: {}{}", safe_json(&json!(name)), self.chat_suffix);
                let ids = tokenizer.encode(prompt.as_str(), false)?.get_ids().to_vec();
                if ids.len() > self.max_sequence_length as usize {
                    return Err(NimbleEncodeError::ContextExceeded {
                        question_id: name.clone(),
                        actual: ids.len(),
                        maximum: self.max_sequence_length,
                    });
                }
                let options = question.option_ids();
                for (code, &token) in
                    self.candidate_codes.iter().zip(self.candidate_token_ids.iter()).take(options.len())
                {
                    let with_code = tokenizer.encode(format!("{prompt}{code}"), false)?;
                    if with_code.get_ids().len() != ids.len() + 1
                        || with_code.get_ids()[..ids.len()] != ids
                        || with_code.get_ids()[ids.len()] != token
                    {
                        return Err(NimbleEncodeError::Invalid(
                            "candidate code merges at the assistant boundary".into(),
                        ));
                    }
                }
                Ok(EncodedNimbleQuestion {
                    question_id: name.clone(),
                    kind: question.kind(),
                    candidate_token_ids: self.candidate_token_ids[..options.len()].into(),
                    option_ids: options,
                    input_ids: ids.into_boxed_slice(),
                })
            })
            .collect()
    }
}

pub struct NimbleAdapter<B: Backend> {
    model: LanguageModel<B>,
    policy: NimblePolicy,
}

#[derive(Debug, Error)]
pub enum NimbleError {
    #[error(transparent)]
    Encoding(#[from] NimbleEncodeError),
    #[error("Candidate inference: {0}")]
    Inference(String),
}

impl<B: Backend> NimbleAdapter<B> {
    pub fn new(
        model: LanguageModel<B>,
        policy: NimblePolicy,
    ) -> Result<Self, NimbleEncodeError> {
        policy.validate(model.tokenizer())?;
        if model.max_context_length().is_some_and(|m| policy.max_sequence_length > m) {
            return Err(NimbleEncodeError::Invalid("release policy exceeds decoder context".into()));
        }
        Ok(Self {
            model,
            policy,
        })
    }
    pub fn model(&self) -> &LanguageModel<B> {
        &self.model
    }
    pub fn policy(&self) -> &NimblePolicy {
        &self.policy
    }
}

impl<B: Backend> DecisionAdapter for NimbleAdapter<B> {
    type Error = NimbleError;
    fn score_questions(
        &self,
        request: &DecisionRequest,
    ) -> Result<Box<[ScoredDecisionQuestion]>, Self::Error> {
        // Validate all prompts before the first forward pass. Each question is
        // independent; no approximate cache reuse is implied by this adapter.
        self.policy
            .encode(self.model.tokenizer(), request)?
            .iter()
            .map(|q| {
                Ok(ScoredDecisionQuestion {
                    question_id: q.question_id.clone(),
                    kind: q.kind,
                    option_ids: q.option_ids.clone(),
                    values: DecisionValues::Logits {
                        values: self
                            .model
                            .score_candidate_tokens(&q.input_ids, &q.candidate_token_ids)
                            .map_err(|e| NimbleError::Inference(e.to_string()))?,
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
}
