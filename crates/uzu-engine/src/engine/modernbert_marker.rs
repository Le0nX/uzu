//! Architecture-specific encoding and release policy for ModernBERT marker heads (Julia/Laya).

use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::{
    decision_model::DecisionModel,
    decision_request::{DecisionQuestion, DecisionRequest},
    typed_decision::{DecisionAdapter, DecisionValues, ScoredDecisionQuestion},
};
use crate::backends::Backend;

pub const MODERNBERT_MARKER_ARCHITECTURE: &str = "modernbert_marker_v1";
fn default_architecture() -> String {
    MODERNBERT_MARKER_ARCHITECTURE.into()
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DecisionFamily {
    Julia,
    Laya,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DecisionPolicy {
    pub family: DecisionFamily,
    #[serde(default = "default_architecture")]
    pub architecture_id: String,
    #[serde(default)]
    pub adapter_id: Option<String>,
    pub max_length: usize,
    pub head_length: usize,
    pub strict_encoding: bool,
    pub temperatures: [f64; 3],
    pub temperatures_by_options: HashMap<String, f64>,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
    pub mask_token: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedDecisionQuestion {
    pub question_id: String,
    pub input_ids: Box<[u32]>,
    pub markers: Box<[u32]>,
    pub question_type: u32,
    pub option_ids: Box<[String]>,
    pub temperature: f64,
}

#[derive(Debug, Error)]
pub enum DecisionRequestError {
    #[error("I/O error: {0}")]
    IO(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Invalid decision request: {0}")]
    Invalid(String),
    #[error("Question {question_id} needs {actual} tokens; model context is {maximum}")]
    ContextExceeded {
        question_id: String,
        actual: usize,
        maximum: usize,
    },
}

// Match the original Python serializer's separators without changing Unicode or key order.
pub(crate) fn json_text(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("[{}]", items.iter().map(json_text).collect::<Vec<_>>().join(", ")),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, value)| format!("{}: {}", Value::String(key.clone()), json_text(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => value.to_string(),
    }
}
pub(crate) fn source_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => json_text(value),
    }
}

pub(crate) fn contains_float(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.is_f64(),
        Value::Array(values) => values.iter().any(contains_float),
        Value::Object(values) => values.values().any(contains_float),
        _ => false,
    }
}

fn question_parts(
    question: &DecisionQuestion,
    family: DecisionFamily,
) -> Result<(u32, &str, Vec<String>, Vec<String>), DecisionRequestError> {
    let (kind, instructions, entries): (u32, &str, Vec<(String, Value)>) = match question {
        DecisionQuestion::Choice {
            instructions,
            criteria,
        } => (0, instructions, criteria.iter().map(|(key, value)| (key.clone(), value.clone())).collect()),
        DecisionQuestion::Score {
            instructions,
            criteria,
        } => (1, instructions, criteria.iter().enumerate().map(|(i, value)| (i.to_string(), value.clone())).collect()),
        DecisionQuestion::Noul {
            instructions,
            criteria,
        } => {
            if family == DecisionFamily::Julia
                && criteria
                    .as_ref()
                    .is_some_and(|c| c.len() != 2 || c.values().any(|v| v.as_str().is_none_or(str::is_empty)))
            {
                return Err(DecisionRequestError::Invalid(
                    "Julia noul criteria must describe both false and true".into(),
                ));
            }
            if criteria.as_ref().is_some_and(|c| c.keys().any(|key| key != "false" && key != "true")) {
                return Err(DecisionRequestError::Invalid("noul criteria accept only false and true".into()));
            }
            (
                2,
                instructions,
                ["false", "true"]
                    .into_iter()
                    .map(|key| {
                        (key.to_string(), criteria.as_ref().and_then(|c| c.get(key)).cloned().unwrap_or(Value::Null))
                    })
                    .collect(),
            )
        },
    };
    if instructions.trim().is_empty() || entries.len() < 2 || (family == DecisionFamily::Julia && entries.len() > 20) {
        return Err(DecisionRequestError::Invalid(
            "instructions must be nonempty; at least two options are required (Julia: at most 20)".into(),
        ));
    }
    let mut ids = Vec::with_capacity(entries.len());
    let mut texts = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        if contains_float(&value) {
            return Err(DecisionRequestError::Invalid("serialize floating-point criteria to source text before passing them to preserve the original tokenizer input".into()));
        }
        let missing = value.is_null() || value.as_str() == Some("");
        let description = match (family, kind, missing) {
            (DecisionFamily::Julia, 2, true) => key.clone(),
            (DecisionFamily::Julia, _, _) => value
                .as_str()
                .filter(|text| !text.is_empty())
                .ok_or_else(|| {
                    DecisionRequestError::Invalid("Julia option descriptions must be nonempty strings".into())
                })?
                .to_string(),
            (DecisionFamily::Laya, 0, true) => key.clone(),
            (DecisionFamily::Laya, 0, false) => format!("{key}: {}", source_text(&value)),
            (DecisionFamily::Laya, 1, _) => {
                if value.is_null() {
                    return Err(DecisionRequestError::Invalid("score levels must not be null".into()));
                }
                format!("level {key}: {}", source_text(&value))
            },
            (DecisionFamily::Laya, 2, _) => format!(
                "{key}: {}",
                if missing {
                    if key == "true" {
                        "yes, the statement holds".into()
                    } else {
                        "no, the statement does not hold".into()
                    }
                } else {
                    source_text(&value)
                }
            ),
            _ => unreachable!(),
        };
        ids.push(key);
        texts.push(description);
    }
    Ok((kind, instructions, ids, texts))
}

impl DecisionPolicy {
    pub fn from_file(path: &Path) -> Result<Self, DecisionRequestError> {
        Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
    }

    fn temperature(
        &self,
        kind: u32,
        options: usize,
    ) -> Result<f64, DecisionRequestError> {
        let bucket = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let name = ["choice", "score", "noul"][kind as usize];
        let value = self
            .temperatures_by_options
            .get(&format!("{name}:{bucket}"))
            .copied()
            .unwrap_or(self.temperatures[kind as usize]);
        if !value.is_finite() || value <= 0.0 {
            return Err(DecisionRequestError::Invalid("temperature must be finite and positive".into()));
        }
        // Laya retains raw checkpoint temperatures, then clamps serving values
        // in Agent.__init__ (laya 0.3.28). Julia has unit calibration.
        Ok(if self.family == DecisionFamily::Laya {
            value.clamp(0.5, 5.0)
        } else {
            value
        })
    }

    /// Reject any input that the release encoder would shorten; never silently truncate evidence.
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        request: &DecisionRequest,
    ) -> Result<Box<[EncodedDecisionQuestion]>, DecisionRequestError> {
        if self.max_length == 0
            || self.head_length == 0
            || self.mask_token.is_empty()
            || request.questions.is_empty()
            || request.state.is_null()
        {
            return Err(DecisionRequestError::Invalid("invalid policy, missing state or empty question set".into()));
        }
        if self.family == DecisionFamily::Julia
            && !matches!(request.state, Value::String(_) | Value::Array(_) | Value::Object(_))
        {
            return Err(DecisionRequestError::Invalid("Julia state must be text, an object or an array".into()));
        }
        if contains_float(&request.state) {
            return Err(DecisionRequestError::Invalid("serialize floating-point state to source text before passing it to preserve the original tokenizer input".into()));
        }
        let mut tokenizer = tokenizer.clone();
        tokenizer.with_truncation(None)?;
        tokenizer.with_padding(None);
        let tokenize = |text: &str| -> Result<Vec<u32>, DecisionRequestError> {
            if text.contains(&self.mask_token) {
                return Err(DecisionRequestError::Invalid("source text contains the reserved option marker".into()));
            }
            Ok(tokenizer.encode(text, false)?.get_ids().to_vec())
        };
        let state = tokenize(&source_text(&request.state))?;
        request
            .questions
            .iter()
            .map(|(qid, question)| {
                let (kind, instructions, option_ids, descriptions) = question_parts(question, self.family)?;
                let name = ["choice", "score", "noul"][kind as usize];
                let question_ids = tokenize(&format!("{name} question: {instructions}"))?;
                let options =
                    descriptions.iter().map(|text| tokenize(&format!(" {text}"))).collect::<Result<Vec<_>, _>>()?;
                if options.iter().any(|tokens| tokens.len() > 48) {
                    return Err(DecisionRequestError::Invalid(format!(
                        "question {qid}: option description exceeds 48 tokens"
                    )));
                }
                let option_tokens = options.iter().map(|tokens| tokens.len() + 1).sum::<usize>();
                if option_tokens + 16 > self.head_length || question_ids.len() + option_tokens > self.head_length {
                    return Err(DecisionRequestError::Invalid(format!(
                        "question {qid}: question/options exceed release head budget {}",
                        self.head_length
                    )));
                }
                let actual = question_ids.len() + option_tokens + state.len() + 4;
                if actual > self.max_length {
                    return Err(DecisionRequestError::ContextExceeded {
                        question_id: qid.clone(),
                        actual,
                        maximum: self.max_length,
                    });
                }
                let mut ids = Vec::with_capacity(actual);
                ids.push(self.cls_token_id);
                ids.extend(question_ids);
                ids.push(self.sep_token_id);
                let mut markers = Vec::with_capacity(options.len());
                for tokens in options {
                    markers.push(ids.len() as u32);
                    ids.push(self.mask_token_id);
                    ids.extend(tokens);
                }
                ids.push(self.sep_token_id);
                ids.extend_from_slice(&state);
                ids.push(self.sep_token_id);
                Ok(EncodedDecisionQuestion {
                    question_id: qid.clone(),
                    input_ids: ids.into_boxed_slice(),
                    markers: markers.into_boxed_slice(),
                    question_type: kind,
                    temperature: self.temperature(kind, option_ids.len())?,
                    option_ids: option_ids.into_boxed_slice(),
                })
            })
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum MarkerAdapterError {
    #[error("Request encoding failed: {0}")]
    Request(#[from] DecisionRequestError),
    #[error("Marker inference failed: {0}")]
    Inference(String),
    #[error("Unsupported marker adapter configuration: {0}")]
    Unsupported(String),
}
/// Julia and Laya share this graph; their family-specific release policies
/// continue to control rendering, sequence budgets and calibration.
pub struct ModernBertMarkerAdapter<B: Backend> {
    model: DecisionModel<B>,
    policy: DecisionPolicy,
}
impl<B: Backend> ModernBertMarkerAdapter<B> {
    pub fn new(
        model: DecisionModel<B>,
        policy: DecisionPolicy,
    ) -> Result<Self, MarkerAdapterError> {
        if policy.architecture_id != MODERNBERT_MARKER_ARCHITECTURE {
            return Err(MarkerAdapterError::Unsupported(format!(
                "architecture {:?} requires its own adapter",
                policy.architecture_id
            )));
        }
        if policy.max_length != model.config().max_sequence_length as usize
            || policy.head_length != model.config().max_head_tokens as usize
        {
            return Err(MarkerAdapterError::Unsupported(
                "release policy and model context/head budget disagree".into(),
            ));
        }
        Ok(Self {
            model,
            policy,
        })
    }
    pub fn model(&self) -> &DecisionModel<B> {
        &self.model
    }
    pub fn policy(&self) -> &DecisionPolicy {
        &self.policy
    }
}
impl<B: Backend> DecisionAdapter for ModernBertMarkerAdapter<B> {
    type Error = MarkerAdapterError;
    fn score_questions(
        &self,
        request: &DecisionRequest,
    ) -> Result<Box<[ScoredDecisionQuestion]>, Self::Error> {
        self.policy
            .encode(self.model.tokenizer(), request)?
            .into_vec()
            .into_iter()
            .map(|question| {
                let logits = self
                    .model
                    .score(&question.input_ids, question.question_type, &question.markers)
                    .map_err(|error| MarkerAdapterError::Inference(error.to_string()))?;
                let kind = request
                    .questions
                    .get(&question.question_id)
                    .ok_or_else(|| MarkerAdapterError::Unsupported("encoded question is absent from request".into()))?
                    .kind();
                Ok(ScoredDecisionQuestion {
                    question_id: question.question_id,
                    kind,
                    option_ids: question.option_ids,
                    values: DecisionValues::Logits {
                        values: logits.into_boxed_slice(),
                        temperature: question.temperature,
                    },
                    input_tokens: Some(question.input_ids.len()),
                })
            })
            .collect()
    }
    fn adapter_id(&self) -> Option<&str> {
        self.policy.adapter_id.as_deref()
    }
    fn architecture_id(&self) -> Option<&str> {
        Some(&self.policy.architecture_id)
    }
}
