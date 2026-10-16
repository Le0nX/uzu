//! JSON-lines typed or raw-candidate inference for a converted Nimble v3 model.
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
        decision_request::DecisionRequest,
        nimble::{NimbleAdapter, NimblePolicy},
        typed_decision::{TypedDecisionEngine, decision_probabilities},
    },
};

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Tokens {
        input_ids: Box<[u32]>,
        candidate_token_ids: Box<[u32]>,
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
        let policy = NimblePolicy::from_file(&self.directory.join("decision_policy.json"))?;
        let model = engine.load_language_model(&self.directory).map_err(|e| e.to_string())?;
        let decisions = TypedDecisionEngine::new(NimbleAdapter::new(model, policy)?);
        lines(|line| match serde_json::from_str::<Input>(line)? {
            Input::Tokens {
                input_ids,
                candidate_token_ids,
            } => {
                let logits = decisions
                    .adapter()
                    .model()
                    .score_candidate_tokens(&input_ids, &candidate_token_ids)
                    .map_err(|e| e.to_string())?;
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
    let directory: PathBuf = args.next().ok_or("usage: nimble_decision MODEL_DIRECTORY [--encode-only]")?.into();
    match args.next().as_deref() {
        Some("--encode-only") => {
            let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
            let policy = NimblePolicy::from_file(&directory.join("decision_policy.json"))?;
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
