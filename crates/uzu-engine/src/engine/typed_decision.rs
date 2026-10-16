//! Architecture-independent typed decisions. Adapters own model-specific input
//! preparation and inference; this layer checks their output against the request.
use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
};

use serde::Serialize;
use thiserror::Error;

use super::decision_request::{DecisionQuestionKind, DecisionRequest};

/// Raw model outputs. Already-normalized probability models need not invent logits.
#[derive(Debug, Clone)]
pub enum DecisionValues {
    Logits {
        values: Box<[f32]>,
        temperature: f64,
    },
    Probabilities {
        values: Box<[f64]>,
    },
}
#[derive(Debug, Clone)]
pub struct ScoredDecisionQuestion {
    pub question_id: String,
    pub kind: DecisionQuestionKind,
    pub option_ids: Box<[String]>,
    pub values: DecisionValues,
    pub input_tokens: Option<usize>,
}
/// A new architecture supplies its own adapter instead of being coerced into a
/// marker head or a ModernBERT checkpoint layout.
pub trait DecisionAdapter {
    type Error: Error;
    fn score_questions(
        &self,
        request: &DecisionRequest,
    ) -> Result<Box<[ScoredDecisionQuestion]>, Self::Error>;
    fn adapter_id(&self) -> Option<&str> {
        None
    }
    fn architecture_id(&self) -> Option<&str> {
        None
    }
    fn model_id(&self) -> Option<&str> {
        None
    }
    fn model_revision(&self) -> Option<&str> {
        None
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct DecisionAnswer {
    #[serde(rename = "type")]
    pub kind: DecisionQuestionKind,
    pub probabilities: BTreeMap<String, f64>,
    /// Positional order of logits and adapter scores.
    pub option_ids: Box<[String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Box<[f32]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<usize>,
}
#[derive(Debug, Clone, Serialize)]
pub struct DecisionResponse {
    pub answers: BTreeMap<String, DecisionAnswer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture_id: Option<String>,
}
#[derive(Debug, Error)]
pub enum TypedDecisionError<E: Error> {
    #[error("Adapter inference failed: {0}")]
    Adapter(#[source] E),
    #[error("Invalid adapter output: {0}")]
    InvalidOutput(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

pub struct TypedDecisionEngine<A: DecisionAdapter> {
    adapter: A,
}
impl<A: DecisionAdapter> TypedDecisionEngine<A> {
    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
        }
    }
    pub fn adapter(&self) -> &A {
        &self.adapter
    }
    pub fn decide(
        &self,
        request: &DecisionRequest,
    ) -> Result<DecisionResponse, TypedDecisionError<A::Error>> {
        if request.questions.is_empty() {
            return Err(TypedDecisionError::InvalidRequest("at least one question is required".into()));
        }
        let scored = self.adapter.score_questions(request).map_err(TypedDecisionError::Adapter)?;
        if scored.len() != request.questions.len() {
            return Err(TypedDecisionError::InvalidOutput("question count differs from the request".into()));
        }
        let mut answers = BTreeMap::new();
        for question in scored {
            let expected = request.questions.get(&question.question_id).ok_or_else(|| {
                TypedDecisionError::InvalidOutput(format!("unexpected question {:?}", question.question_id))
            })?;
            if answers.contains_key(&question.question_id) {
                return Err(TypedDecisionError::InvalidOutput(format!(
                    "duplicate question {:?}",
                    question.question_id
                )));
            }
            if question.kind != expected.kind() {
                return Err(TypedDecisionError::InvalidOutput(format!(
                    "question {:?} changed type",
                    question.question_id
                )));
            }
            let expected_ids = expected.option_ids();
            let expected_set = expected_ids.iter().collect::<HashSet<_>>();
            let actual_set = question.option_ids.iter().collect::<HashSet<_>>();
            if question.option_ids.is_empty()
                || actual_set.len() != question.option_ids.len()
                || expected_set.len() != expected_ids.len()
                || actual_set != expected_set
            {
                return Err(TypedDecisionError::InvalidOutput(format!(
                    "question {:?} has missing, duplicate or unexpected options",
                    question.question_id
                )));
            }
            let (probabilities, logits, temperature) = match question.values {
                DecisionValues::Logits {
                    values,
                    temperature,
                } => {
                    if values.len() != question.option_ids.len() {
                        return Err(TypedDecisionError::InvalidOutput(format!(
                            "question {:?} has mismatched logits/options",
                            question.question_id
                        )));
                    }
                    let probabilities = decision_probabilities(&values, temperature)
                        .map_err(|error| TypedDecisionError::InvalidOutput(error.to_string()))?;
                    (probabilities, Some(values), Some(temperature))
                },
                DecisionValues::Probabilities {
                    values,
                } => {
                    if values.len() != question.option_ids.len()
                        || values.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                        || (values.iter().sum::<f64>() - 1.0).abs() > 1e-6
                    {
                        return Err(TypedDecisionError::InvalidOutput(format!(
                            "question {:?} must return finite probabilities in [0,1] summing to one (tolerance 1e-6)",
                            question.question_id
                        )));
                    }
                    (values, None, None)
                },
            };
            let option_ids = question.option_ids;
            let probabilities = option_ids.iter().cloned().zip(probabilities.into_vec()).collect::<BTreeMap<_, _>>();
            let noul = if question.kind == DecisionQuestionKind::Noul {
                Some(
                    *probabilities
                        .get("true")
                        .ok_or_else(|| TypedDecisionError::InvalidOutput("noul must contain a true option".into()))?,
                )
            } else {
                None
            };
            answers.insert(
                question.question_id,
                DecisionAnswer {
                    kind: question.kind,
                    probabilities,
                    option_ids,
                    noul,
                    logits,
                    temperature,
                    input_tokens: question.input_tokens,
                },
            );
        }
        Ok(DecisionResponse {
            answers,
            model: self.adapter.model_id().map(str::to_owned),
            model_revision: self.adapter.model_revision().map(str::to_owned),
            adapter_id: self.adapter.adapter_id().map(str::to_owned),
            architecture_id: self.adapter.architecture_id().map(str::to_owned),
        })
    }
}

#[derive(Debug, Error)]
#[error("expected nonempty finite logits and positive finite temperature")]
pub struct DecisionProbabilityError;

/// Stable softmax for logits models. Probability-only adapters bypass this
/// function and their supplied distributions are checked without renormalizing.
pub fn decision_probabilities(
    logits: &[f32],
    temperature: f64,
) -> Result<Box<[f64]>, DecisionProbabilityError> {
    if logits.is_empty()
        || !temperature.is_finite()
        || temperature <= 0.0
        || logits.iter().any(|value| !value.is_finite())
    {
        return Err(DecisionProbabilityError);
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let values = logits.iter().map(|value| ((*value as f64 - maximum) / temperature).exp()).collect::<Vec<_>>();
    let denominator = values.iter().sum::<f64>();
    Ok(values.iter().map(|value| value / denominator).collect())
}
