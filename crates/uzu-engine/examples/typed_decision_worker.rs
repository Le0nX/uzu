//! Resident, non-generative worker for the local typed-decision service.
//! One ready event after loading, then one structured result per request line.
use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use serde_json::{Value, json};
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        code_readout::{CodeReadoutAdapter, CodeReadoutError, CodeReadoutPolicy},
        decision_model::DataType,
        decision_request::DecisionRequest,
        modernbert_marker::{DecisionPolicy, DecisionRequestError, MarkerAdapterError, ModernBertMarkerAdapter},
        nimble::{NimbleAdapter, NimbleEncodeError, NimbleError, NimblePolicy},
        typed_decision::{DecisionAdapter, TypedDecisionEngine, TypedDecisionError},
    },
};

fn failure(
    code: &str,
    message: impl ToString,
) -> Value {
    json!({"code":code,"message":message.to_string()})
}
fn context(
    question: &str,
    actual: usize,
    maximum: usize,
    message: impl ToString,
) -> Value {
    json!({"code":"context_exceeded","message":message.to_string(),"question_id":question,
        "actual_tokens":actual,"maximum_tokens":maximum,"truncated":false})
}
fn marker_error(error: &MarkerAdapterError) -> Value {
    match error {
        MarkerAdapterError::Request(DecisionRequestError::ContextExceeded {
            question_id,
            actual,
            maximum,
        }) => context(question_id, *actual, *maximum, error),
        MarkerAdapterError::Request(_) => failure("invalid_request", error),
        MarkerAdapterError::Unsupported(_) => failure("unsupported_architecture", error),
        MarkerAdapterError::Inference(_) => failure("inference_failed", error),
    }
}
fn d3_error(error: &CodeReadoutError) -> Value {
    match error {
        CodeReadoutError::ContextExceeded {
            question_id,
            actual,
            maximum,
        } => context(question_id, *actual, *maximum as usize, error),
        CodeReadoutError::Invalid(_) | CodeReadoutError::Tokenizer(_) => failure("invalid_request", error),
        CodeReadoutError::Inference(_) => failure("inference_failed", error),
    }
}
fn nimble_error(error: &NimbleError) -> Value {
    match error {
        NimbleError::Encoding(NimbleEncodeError::ContextExceeded {
            question_id,
            actual,
            maximum,
        }) => context(question_id, *actual, *maximum as usize, error),
        NimbleError::Encoding(_) => failure("invalid_request", error),
        NimbleError::Inference(_) => failure("inference_failed", error),
    }
}
fn serve<A: DecisionAdapter>(
    adapter: A,
    metadata: Value,
    classify: impl Fn(&A::Error) -> Value,
) -> Result<(), Box<dyn Error>> {
    let engine = TypedDecisionEngine::new(adapter);
    let mut out = io::stdout().lock();
    serde_json::to_writer(
        &mut out,
        &json!({"event":"ready","protocol":"uzu.typed_decision/1",
        "metadata":metadata,"execution":{"question_mode":"sequential","gpu_batching":false,"shared_prefix_cache":false}}),
    )?;
    writeln!(out)?;
    out.flush()?;
    for line in io::stdin().lock().lines() {
        let result = match serde_json::from_str::<DecisionRequest>(&line?) {
            Err(e) => json!({"ok":false,"error":failure("invalid_request",e)}),
            Ok(request) => match engine.decide(&request) {
                Ok(response) => json!({"ok":true,"response":response}),
                Err(TypedDecisionError::Adapter(e)) => json!({"ok":false,"error":classify(&e)}),
                Err(TypedDecisionError::InvalidRequest(e)) => json!({"ok":false,"error":failure("invalid_request",e)}),
                Err(TypedDecisionError::InvalidOutput(e)) => json!({"ok":false,"error":failure("invalid_output",e)}),
            },
        };
        serde_json::to_writer(&mut out, &result)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}
struct Run {
    directory: PathBuf,
    precision: DataType,
    architecture: String,
    policy: Value,
}
impl BackendSelection for Run {
    type Output = ();
    type Error = Box<dyn Error>;
    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        let engine = Engine::<B>::new().map_err(|e| e.to_string())?;
        let path = self.directory.join("decision_policy.json");
        // Each branch owns loading and encoding. Adding an ID alone does not
        // implement a new architecture; provide an actual DecisionAdapter.
        match self.architecture.as_str() {
            "modernbert_marker_v1" => {
                let policy = DecisionPolicy::from_file(&path)?;
                let metadata = json!({"architecture_id":self.architecture,"adapter_id":policy.adapter_id,
                    "family":policy.family,"max_context_tokens":policy.max_length});
                let model = engine
                    .load_decision_model_with_data_type(&self.directory, self.precision)
                    .map_err(|e| e.to_string())?;
                serve(ModernBertMarkerAdapter::new(model, policy)?, metadata, marker_error)
            },
            "qwen_code_readout_v1" => {
                if self.precision != DataType::BF16 {
                    return Err("code-readout worker requires BF16 backbone / native head precision".into());
                }
                let policy = CodeReadoutPolicy::from_file(&path)?;
                let metadata = json!({"architecture_id":self.architecture,"adapter_id":policy.adapter_id,
                    "model":policy.model_id,"model_revision":policy.revision,"max_context_tokens":policy.max_sequence_length});
                let model = engine.load_code_decision_model(&self.directory).map_err(|e| e.to_string())?;
                serve(CodeReadoutAdapter::new(model, policy)?, metadata, d3_error)
            },
            "causal_token_readout_v1" => {
                if self.precision != DataType::BF16 {
                    return Err("Nimble precision comes from the converted checkpoint".into());
                }
                let policy = NimblePolicy::from_file(&path)?;
                let metadata = json!({"architecture_id":self.architecture,"adapter_id":policy.adapter_id,
                    "max_context_tokens":policy.max_sequence_length});
                let model = engine.load_language_model(&self.directory).map_err(|e| e.to_string())?;
                serve(NimbleAdapter::new(model, policy)?, metadata, nimble_error)
            },
            _ => Err(format!(
                "unsupported_architecture: {:?}; explicit adapter required (policy: {:?})",
                self.architecture,
                self.policy.get("adapter_id")
            )
            .into()),
        }
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory: PathBuf = args.next().ok_or("usage: typed_decision_worker MODEL_DIRECTORY [--f32]")?.into();
    let precision = match args.next().as_deref() {
        None => DataType::BF16,
        Some("--f32") => DataType::F32,
        _ => return Err("expected --f32 or no precision override".into()),
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let policy: Value = serde_json::from_slice(&std::fs::read(directory.join("decision_policy.json"))?)?;
    let architecture = match policy.get("architecture_id").and_then(Value::as_str) {
        Some(s) => s.to_owned(),
        None if matches!(policy.get("family").and_then(Value::as_str), Some("julia" | "laya")) => {
            "modernbert_marker_v1".into()
        },
        _ => return Err("unsupported_architecture: no recognized architecture in release policy".into()),
    };
    select_backend(
        Run {
            directory,
            precision,
            architecture,
            policy,
        },
        "unknown backend".into(),
    )
}
