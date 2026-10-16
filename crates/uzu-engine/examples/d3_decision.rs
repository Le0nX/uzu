//! JSON-lines typed or raw-candidate inference for a converted d3 code-readout model.
use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use serde::Deserialize;
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        d3::{D3Adapter, D3Policy},
        decision_request::DecisionRequest,
        typed_decision::{TypedDecisionEngine, decision_probabilities},
    },
};

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Tokens {
        input_ids: Box<[u32]>,
        readout_rows: Box<[u32]>,
    },
    Request(DecisionRequest),
}
fn lines(mut process: impl FnMut(&str) -> Result<Value, Box<dyn Error>>) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = process(&line).unwrap_or_else(|e| json!({"error":e.to_string()}));
        serde_json::to_writer(&mut stdout, &result)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
struct Run {
    directory: PathBuf,
}
impl BackendSelection for Run {
    type Output = ();
    type Error = Box<dyn Error>;
    fn select<B: Backend>(self) -> Result<(), Self::Error> {
        let engine = Engine::<B>::new().map_err(|e| e.to_string())?;
        let policy = D3Policy::from_file(&self.directory.join("decision_policy.json"))?;
        let model = engine.load_code_decision_model(&self.directory).map_err(|e| e.to_string())?;
        let decisions = TypedDecisionEngine::new(D3Adapter::new(model, policy)?);
        lines(|line| match serde_json::from_str::<Input>(line)? {
            Input::Tokens {
                input_ids,
                readout_rows,
            } => {
                let logits = decisions.adapter().model().score(&input_ids, &readout_rows).map_err(|e| e.to_string())?;
                Ok(
                    json!({"logits":logits,"probabilities":decision_probabilities(&logits,decisions.adapter().policy().temperature)?,"input_tokens":input_ids.len()}),
                )
            },
            Input::Request(request) => Ok(serde_json::to_value(decisions.decide(&request)?)?),
        })
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory: PathBuf = args.next().ok_or("usage: d3_decision MODEL_DIRECTORY [--encode-only]")?.into();
    match args.next().as_deref() {
        Some("--encode-only") => {
            let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
            let policy = D3Policy::from_file(&directory.join("decision_policy.json"))?;
            lines(|line| {
                Ok(serde_json::to_value(policy.encode(&tokenizer, &serde_json::from_str::<DecisionRequest>(line)?)?)?)
            })
        },
        None => select_backend(
            Run {
                directory,
            },
            "unknown backend".into(),
        ),
        _ => Err("expected --encode-only or no extra arguments".into()),
    }
}
