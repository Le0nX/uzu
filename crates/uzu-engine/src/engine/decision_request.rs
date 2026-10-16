//! Architecture-independent typed decision input. Adapters own model-specific encoding.
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DecisionQuestionKind {
    Choice,
    Score,
    Noul,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum DecisionQuestion {
    Choice {
        instructions: String,
        criteria: Map<String, Value>,
    },
    Score {
        instructions: String,
        criteria: Box<[Value]>,
    },
    Noul {
        instructions: String,
        criteria: Option<Map<String, Value>>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: IndexMap<String, DecisionQuestion>,
}

impl DecisionQuestion {
    pub fn kind(&self) -> DecisionQuestionKind {
        match self {
            Self::Choice {
                ..
            } => DecisionQuestionKind::Choice,
            Self::Score {
                ..
            } => DecisionQuestionKind::Score,
            Self::Noul {
                ..
            } => DecisionQuestionKind::Noul,
        }
    }
    pub fn option_ids(&self) -> Box<[String]> {
        match self {
            Self::Choice {
                criteria,
                ..
            } => criteria.keys().cloned().collect(),
            Self::Score {
                criteria,
                ..
            } => (0..criteria.len()).map(|i| i.to_string()).collect(),
            Self::Noul {
                ..
            } => ["false".into(), "true".into()].into(),
        }
    }
}
