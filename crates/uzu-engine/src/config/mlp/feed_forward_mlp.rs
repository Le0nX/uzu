use uzu_engine_macros::uzu_config;

use crate::config::{activation::AnyActivation, linear::LinearConfig};
/// An ungated feed-forward block: down(activation(up(x))).
#[uzu_config(super::MLPConfig)]
pub struct FeedForwardMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: AnyActivation,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
}
