use uzu_engine_macros::uzu_config_abstract;

pub mod dense_mlp;
pub mod feed_forward_mlp;
pub mod mixture_of_experts;
pub mod routing_function;

#[uzu_config_abstract(
    dense_mlp::DenseMLPConfig,
    feed_forward_mlp::FeedForwardMLPConfig,
    mixture_of_experts::MixtureOfExpertsConfig
)]
pub struct MLPConfig;
