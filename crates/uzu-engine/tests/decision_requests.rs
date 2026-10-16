use serde_json::json;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
use uzu_engine::engine::{
    decision_request::DecisionRequest,
    modernbert_marker::{DecisionFamily, DecisionPolicy, DecisionRequestError},
    typed_decision::decision_probabilities,
};

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab([(String::from("[UNK]"), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

fn policy(family: DecisionFamily) -> DecisionPolicy {
    DecisionPolicy {
        family,
        architecture_id: "modernbert_marker_v1".into(),
        adapter_id: None,
        max_length: 8192,
        head_length: 512,
        strict_encoding: true,
        temperatures: [1.0, 1.0, 1.0],
        temperatures_by_options: Default::default(),
        cls_token_id: 1,
        sep_token_id: 2,
        mask_token_id: 3,
        pad_token_id: 0,
        mask_token: "[MASK]".into(),
    }
}

#[test]
fn choice_preserves_option_order_and_marker_boundaries() {
    let request: DecisionRequest = serde_json::from_str(r#"{"state":"sample A","questions":{"stage":{"type":"choice","instructions":"Which stage?","criteria":{"later":"E10.5","earlier":"E8.5"}}}}"#).unwrap();
    let encoded = policy(DecisionFamily::Julia).encode(&tokenizer(), &request).unwrap();
    let q = &encoded[0];
    assert_eq!(&*q.option_ids, &["later", "earlier"]);
    assert_eq!(q.question_type, 0);
    for &marker in &q.markers {
        assert_eq!(q.input_ids[marker as usize], 3);
    }
    assert_eq!(q.input_ids.first(), Some(&1));
    assert_eq!(q.input_ids.last(), Some(&2));
}

#[test]
fn overflow_is_explicit_and_never_truncates() {
    let request: DecisionRequest = serde_json::from_value(
        json!({"state":"a b c d e f g h", "questions":{"ok":{"type":"noul","instructions":"Valid?"}}}),
    )
    .unwrap();
    let mut policy = policy(DecisionFamily::Laya);
    let complete = policy.encode(&tokenizer(), &request).unwrap();
    policy.max_length = complete[0].input_ids.len() - 1;
    assert!(
        matches!(policy.encode(&tokenizer(),&request),Err(DecisionRequestError::ContextExceeded{actual,maximum,..}) if actual==maximum+1)
    );
}

#[test]
fn head_budget_and_reserved_markers_are_not_silently_changed() {
    let request: DecisionRequest =
        serde_json::from_value(json!({"state":"evidence", "questions":{"ok":{"type":"noul","instructions":"Valid?"}}}))
            .unwrap();
    let mut policy = policy(DecisionFamily::Laya);
    policy.head_length = 8;
    assert!(policy.encode(&tokenizer(), &request).is_err());
    let injected: DecisionRequest = serde_json::from_value(
        json!({"state":"evidence [MASK]", "questions":{"ok":{"type":"noul","instructions":"Valid?"}}}),
    )
    .unwrap();
    assert!(policy.encode(&tokenizer(), &injected).is_err());
}

#[test]
fn native_laya_temperature_clamps_serving_but_preserves_raw_policy() {
    let request: DecisionRequest = serde_json::from_value(json!({"state":"evidence", "questions":{"pick":{"type":"choice","instructions":"Pick one","criteria":{"a":"first","b":"second","c":"third"}}}})).unwrap();
    let mut policy = policy(DecisionFamily::Laya);
    policy.temperatures_by_options.insert("choice:3-5".into(), 0.1);
    assert_eq!(policy.encode(&tokenizer(), &request).unwrap()[0].temperature, 0.5);
    let probabilities = decision_probabilities(&[1.0, 2.0], 0.1).unwrap();
    assert!(probabilities[1] > 0.999);
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(decision_probabilities(&[1.0, f32::NAN], 1.0).is_err());
    assert!(decision_probabilities(&[1.0, 2.0], 0.0).is_err());
}

#[test]
fn julia_rejects_incomplete_descriptions_and_laya_accepts_structured_criteria() {
    let request: DecisionRequest = serde_json::from_value(json!({"state":{"sample":"A"}, "questions":{"pick":{"type":"choice","instructions":"Pick one","criteria":{"a":null,"b":{"stage":"E8.5"}}}}})).unwrap();
    assert!(policy(DecisionFamily::Julia).encode(&tokenizer(), &request).is_err());
    assert!(policy(DecisionFamily::Laya).encode(&tokenizer(), &request).is_ok());
}

#[test]
fn unsupported_fields_and_float_serialization_are_explicit_errors() {
    assert!(serde_json::from_value::<DecisionRequest>(json!({"state":"sample", "questions":{"q":{"type":"noul","instructions":"Valid?","option_order":["true","false"]}}})).is_err());
    let request: DecisionRequest = serde_json::from_value(
        json!({"state":{"score":1e-6}, "questions":{"q":{"type":"noul","instructions":"Valid?"}}}),
    )
    .unwrap();
    assert!(policy(DecisionFamily::Julia).encode(&tokenizer(), &request).is_err());
    let incomplete: DecisionRequest = serde_json::from_value(json!({"state":"sample", "questions":{"q":{"type":"noul","instructions":"Valid?","criteria":{"true":"valid"}}}})).unwrap();
    assert!(policy(DecisionFamily::Julia).encode(&tokenizer(), &incomplete).is_err());
}
