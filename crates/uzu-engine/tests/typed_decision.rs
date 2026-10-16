use std::convert::Infallible;

use serde_json::json;
use uzu_engine::engine::{
    decision_request::{DecisionQuestionKind, DecisionRequest},
    typed_decision::{DecisionAdapter, DecisionValues, ScoredDecisionQuestion, TypedDecisionEngine},
};

// A probability-only adapter exercises the public contract without a BERT
// tokenizer, checkpoint, marker policy or inference backend.
struct ProbabilitySource {
    questions: Box<[ScoredDecisionQuestion]>,
}

impl DecisionAdapter for ProbabilitySource {
    type Error = Infallible;
    fn score_questions(
        &self,
        _: &DecisionRequest,
    ) -> Result<Box<[ScoredDecisionQuestion]>, Self::Error> {
        Ok(self.questions.clone())
    }
    fn architecture_id(&self) -> Option<&str> {
        Some("test_probability_source")
    }
}

fn request() -> DecisionRequest {
    serde_json::from_value(json!({"state":{"measured_value":1e-6},"questions":{"pick":{"type":"choice","instructions":"Choose","criteria":{"a":"first","b":"second"}}}})).unwrap()
}

fn scored(values: DecisionValues) -> ScoredDecisionQuestion {
    ScoredDecisionQuestion {
        question_id: "pick".into(),
        kind: DecisionQuestionKind::Choice,
        option_ids: vec!["a".into(), "b".into()].into_boxed_slice(),
        values,
        input_tokens: None,
    }
}

#[test]
fn native_probabilities_use_the_same_contract_without_marker_assumptions() {
    let engine = TypedDecisionEngine::new(ProbabilitySource {
        questions: vec![scored(DecisionValues::Probabilities {
            values: vec![0.0, 1.0].into_boxed_slice(),
        })]
        .into_boxed_slice(),
    });
    let result = engine.decide(&request()).unwrap();
    let answer = &result.answers["pick"];
    assert_eq!(answer.probabilities["a"], 0.0);
    assert_eq!(answer.probabilities["b"], 1.0);
    assert!(answer.logits.is_none() && answer.temperature.is_none());
    assert_eq!(result.architecture_id.as_deref(), Some("test_probability_source"));
}

#[test]
fn a_single_option_is_valid_in_the_generic_contract() {
    let request=serde_json::from_value(json!({"state":"evidence","questions":{"pick":{"type":"choice","instructions":"Choose","criteria":{"only":"the sole candidate"}}}})).unwrap();
    let question = ScoredDecisionQuestion {
        question_id: "pick".into(),
        kind: DecisionQuestionKind::Choice,
        option_ids: vec!["only".into()].into_boxed_slice(),
        values: DecisionValues::Probabilities {
            values: vec![1.0].into_boxed_slice(),
        },
        input_tokens: None,
    };
    let result = TypedDecisionEngine::new(ProbabilitySource {
        questions: vec![question].into_boxed_slice(),
    })
    .decide(&request)
    .unwrap();
    assert_eq!(result.answers["pick"].probabilities["only"], 1.0);
}

#[test]
fn logits_are_normalized_once_with_declared_temperature() {
    let engine = TypedDecisionEngine::new(ProbabilitySource {
        questions: vec![scored(DecisionValues::Logits {
            values: vec![10_000.0, 10_001.0].into_boxed_slice(),
            temperature: 2.0,
        })]
        .into_boxed_slice(),
    });
    let result = engine.decide(&request()).unwrap();
    assert!((result.answers["pick"].probabilities["b"] - 1.0 / (1.0 + (-0.5f64).exp())).abs() < 1e-12);
    assert_eq!(result.answers["pick"].temperature, Some(2.0));
}

#[test]
fn noul_uses_option_identity_instead_of_position() {
    let request =
        serde_json::from_value(json!({"state":"sample","questions":{"ok":{"type":"noul","instructions":"Valid?"}}}))
            .unwrap();
    let question = ScoredDecisionQuestion {
        question_id: "ok".into(),
        kind: DecisionQuestionKind::Noul,
        option_ids: vec!["true".into(), "false".into()].into_boxed_slice(),
        values: DecisionValues::Probabilities {
            values: vec![0.9, 0.1].into_boxed_slice(),
        },
        input_tokens: None,
    };
    let result = TypedDecisionEngine::new(ProbabilitySource {
        questions: vec![question].into_boxed_slice(),
    })
    .decide(&request)
    .unwrap();
    assert_eq!(result.answers["ok"].noul, Some(0.9));
    assert_eq!(&*result.answers["ok"].option_ids, &["true", "false"]);
}

#[test]
fn malformed_distributions_are_errors_instead_of_fabricated_predictions() {
    for values in [vec![0.0, 0.0], vec![0.3, 0.3], vec![-0.1, 1.1], vec![f64::NAN, 0.5], vec![0.2, 0.3, 0.5]] {
        let engine = TypedDecisionEngine::new(ProbabilitySource {
            questions: vec![scored(DecisionValues::Probabilities {
                values: values.into_boxed_slice(),
            })]
            .into_boxed_slice(),
        });
        assert!(engine.decide(&request()).is_err());
    }
}

#[test]
fn adapters_cannot_change_questions_types_or_option_identities() {
    let valid = scored(DecisionValues::Probabilities {
        values: vec![0.25, 0.75].into_boxed_slice(),
    });
    let mut wrong_id = valid.clone();
    wrong_id.question_id = "other".into();
    let mut wrong_options = valid.clone();
    wrong_options.option_ids = vec!["a".into(), "a".into()].into_boxed_slice();
    let mut wrong_kind = valid.clone();
    wrong_kind.kind = DecisionQuestionKind::Score;
    for question in [wrong_id, wrong_options, wrong_kind] {
        assert!(
            TypedDecisionEngine::new(ProbabilitySource {
                questions: vec![question].into_boxed_slice()
            })
            .decide(&request())
            .is_err()
        );
    }
    assert!(
        TypedDecisionEngine::new(ProbabilitySource {
            questions: vec![valid.clone(), valid].into_boxed_slice()
        })
        .decide(&request())
        .is_err()
    );
}
