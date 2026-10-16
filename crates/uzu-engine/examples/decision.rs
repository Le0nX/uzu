//! JSON-lines inference for converted Julia/Laya checkpoints.
use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use serde::Deserialize;
use serde_json::{Value, json};
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        decision_model::DataType,
        decision_request::DecisionRequest,
        modernbert_marker::{DecisionPolicy, ModernBertMarkerAdapter},
        typed_decision::{TypedDecisionEngine, decision_probabilities},
    },
};

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Tokens {
        input_ids: Box<[u32]>,
        markers: Box<[u32]>,
        question_type: u32,
    },
    Request(DecisionRequest),
}

struct Run {
    model: PathBuf,
    precision: DataType,
}
impl BackendSelection for Run {
    type Output = ();
    type Error = Box<dyn Error>;

    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        let engine = Engine::<B>::new().map_err(|error| error.to_string())?;
        let model = engine
            .load_decision_model_with_data_type(&self.model, self.precision)
            .map_err(|error| error.to_string())?;
        let policy = DecisionPolicy::from_file(&self.model.join("decision_policy.json"))?;
        let decisions = TypedDecisionEngine::new(ModernBertMarkerAdapter::new(model, policy)?);
        let mut stdout = io::stdout().lock();
        for line in io::stdin().lock().lines() {
            let line = line?;
            let result = (|| -> Result<Value, Box<dyn Error>> {
                match serde_json::from_str::<Input>(&line)? {
                    Input::Tokens {
                        input_ids,
                        markers,
                        question_type,
                    } => {
                        let logits = decisions
                            .adapter()
                            .model()
                            .score(&input_ids, question_type, &markers)
                            .map_err(|error| error.to_string())?;
                        Ok(json!({"logits":logits, "probabilities":decision_probabilities(&logits, 1.0)?}))
                    },
                    Input::Request(request) => Ok(serde_json::to_value(decisions.decide(&request)?)?),
                }
            })();
            let output = match result {
                Ok(value) => value,
                Err(error) => json!({"error":error.to_string()}),
            };
            serde_json::to_writer(&mut stdout, &output)?;
            writeln!(stdout)?;
            stdout.flush()?;
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("usage: decision MODEL_DIRECTORY [--f32]")?.into();
    let precision = match args.next().as_deref() {
        None => DataType::BF16,
        Some("--f32") => DataType::F32,
        _ => return Err("expected --f32 or no precision argument".into()),
    };
    select_backend(
        Run {
            model,
            precision,
        },
        "unknown backend".into(),
    )
}
