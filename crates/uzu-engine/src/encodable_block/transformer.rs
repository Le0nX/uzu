use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::TensorAddScaleKernel,
    },
    config::{rope::AnyRoPEConfig, transformer::TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{MixerState, attention::rope::PrecalculatedRoPE},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    Shared(u32),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: u32,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
}

impl<B: Backend> TransformerState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.prepare(context_length, suffix_length)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        {
            let command_buffer = &mut *command_buffer.debug_group("transformer accept");

            for layer_state in &mut self.layer_states {
                let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                    continue;
                };

                layer_state.encode_accept(accepted_indices, command_buffer)?;
            }

            self.context_length += accepted_indices.len() as u32;
        }

        command_buffer.debug_point(&format!("accepted {} tokens", accepted_indices.len()));

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TransformerNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

#[derive(Debug, Error)]
pub enum TransformerEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
}

pub struct Transformer<B: Backend> {
    ropes: Box<[AnyRoPEConfig]>,
    bf16_rope_tables: bool,
    layers: Box<[(TransformerLayer<B>, Option<usize>)]>,
    output_norm: Normalization<B>,
    model_dim: u32,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        data_type: DataType,
        output_norm_hadamard_factors: Option<B::GlobalBuffer>,
        transformer_config: &TransformerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, TransformerNewError<B>> {
        let mut ropes: Vec<AnyRoPEConfig> = Vec::new();

        let num_layers = transformer_config.layer_configs.len() as u32;
        let layers = transformer_config
            .layer_configs
            .iter()
            .zip(0u32..)
            .map(|(layer_config, layer_index)| {
                let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index));

                let rope = layer_config.rope_config.as_ref().map(|layer_rope_config| {
                    ropes.iter().position(|rope_config| rope_config == layer_rope_config).unwrap_or_else(|| {
                        ropes.push(layer_rope_config.clone());
                        ropes.len() - 1
                    })
                });

                let layer = TransformerLayer::new(
                    transformer_config.model_dim,
                    transformer_config.hidden_dim,
                    num_layers,
                    layer_index,
                    data_type,
                    layer_config,
                    &layer_loader,
                    context,
                )?;

                Ok((layer, rope))
            })
            .collect::<Result<Box<[_]>, TransformerNewError<B>>>()?;

        let output_norm = Normalization::new(
            transformer_config.model_dim,
            data_type,
            output_norm_hadamard_factors,
            if num_layers == 0 {
                ShortcutMode::None
            } else {
                ShortcutMode::Add
            },
            PostLayerScalar::None,
            &transformer_config.output_norm_config,
            &parameter_tree.subtree("output_norm"),
            context,
        )?;

        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(TransformerNewError::Backend)?;

        Ok(Self {
            ropes: ropes.into_boxed_slice(),
            bf16_rope_tables: false,
            layers,
            output_norm,
            model_dim: transformer_config.model_dim,
            residual_add,
        })
    }

    /// ModernBERT casts rotary tables to the activation dtype before applying
    /// rotations. Other transformer families retain the existing F32 tables.
    pub fn with_bf16_rope_tables(mut self) -> Self {
        self.bf16_rope_tables = true;
        self
    }

    fn capture_residual(
        &self,
        shortcut: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        batch_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch(hidden.size())?;
        let elements = batch_size * self.model_dim;
        self.residual_add.encode(Some(shortcut), hidden, &mut output, elements, elements, 1.0, command_buffer);
        Ok(output)
    }

    pub fn speculation_supported(&self) -> bool {
        self.layers.iter().all(|(layer, _rope)| layer.mixer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers.iter().map(|(layer, _rope_index)| layer.mixer.max_context_length()).fold(None, |acc, el| {
            match (acc, el) {
                (Some(a), Some(b)) => Some(u32::min(a, b)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }
        })
    }

    pub fn prefill_cache_layer_count(&self) -> usize {
        let num_layers = self.layers.len();
        let Some(last_owned_kv_layer_index) =
            self.layers.iter().rposition(|(layer, _rope_index)| layer.kv_source_layer_index.is_none())
        else {
            return num_layers;
        };

        last_owned_kv_layer_index + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.layers.len()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .layers
            .iter()
            .map(|(layer, _rope)| match layer.kv_source_layer_index {
                None => {
                    layer.mixer.create_empty_state(max_context_length, context).map(TransformerLayerStateType::Owned)
                },
                Some(kv_source_layer_index) => Ok(TransformerLayerStateType::Shared(kv_source_layer_index)),
            })
            .collect::<Result<_, B::Error>>()?;

        let context_length = 0;

        Ok(TransformerState {
            layer_states,
            context_length,
        })
    }

    /// Synchronous whole-prompt execution for the independent code-readout graph.
    /// Every layer still sees every token; only temporary allocation lifetimes
    /// change. Keeping just one layer's KV/state live avoids a prompt-length-
    /// dependent accumulation of wired Metal memory across all layers.
    fn encode_code_attention_layer(
        &self,
        layer: &TransformerLayer<B>,
        attention: &crate::encodable_block::mixer::attention::Attention<B>,
        input: &B::GlobalBuffer,
        n: u32,
        rope_index: Option<usize>,
        context: &B::Context,
    ) -> Result<B::GlobalBuffer, Box<dyn std::error::Error>> {
        use crate::backends::common::{CommandBufferExecutable, CommandBufferPending, Context};
        let row_bytes = input.size() / n as usize;
        let mut normalized = context.create_buffer(input.size()).map_err(|e| e.to_string())?;
        let mut cb = context.create_command_buffer(Some("code attention pre norm"), None).map_err(|e| e.to_string())?;
        if let Some(norm) = &layer.pre_mixer_norm {
            let mut shortcut = cb.allocate_scratch(input.size()).map_err(|e| e.to_string())?;
            cb.encode_fill(&mut shortcut, 0);
            let out = norm.encode(input, 0, n, Some(&mut shortcut), &mut cb).map_err(|e| e.to_string())?;
            cb.encode_copy(&out, &mut normalized);
            drop(out);
        } else {
            cb.encode_copy(input, &mut normalized);
        }
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        let attended = attention.encode_code_prompt(
            &normalized,
            n,
            rope_index.map(|i| &self.ropes[i]),
            self.bf16_rope_tables,
            context,
        )?;
        drop(normalized);
        let mut result = context.create_buffer(input.size()).map_err(|e| e.to_string())?;
        for start in (0..n).step_by(4096) {
            let count = (n - start).min(4096);
            let lo = start as usize * row_bytes;
            let hi = (start + count) as usize * row_bytes;
            context.release_cached_buffers();
            let mut cb = context
                .create_command_buffer(Some("code attention pointwise MLP tile"), None)
                .map_err(|e| e.to_string())?;
            let mut hidden = cb.allocate_scratch(hi - lo).map_err(|e| e.to_string())?;
            cb.encode_copy(attended.subrange(lo..hi), &mut hidden);
            let mut shortcut = cb.allocate_scratch(hi - lo).map_err(|e| e.to_string())?;
            cb.encode_copy(input.subrange(lo..hi), &mut shortcut);
            if let Some(norm) = &layer.post_mixer_norm {
                hidden = norm
                    .encode(&hidden, 0, count, None::<&mut B::ScratchBuffer>, &mut cb)
                    .map_err(|e| e.to_string())?;
            }
            let norm = layer
                .pre_mlp_norm
                .encode(&hidden, 0, count, Some(&mut shortcut), &mut cb)
                .map_err(|e| e.to_string())?;
            drop(hidden);
            let mut hidden = layer.mlp.encode(norm, count, &mut cb).map_err(|e| e.to_string())?;
            if let Some(norm) = &layer.post_mlp_norm {
                hidden = norm
                    .encode(&hidden, 0, count, None::<&mut B::ScratchBuffer>, &mut cb)
                    .map_err(|e| e.to_string())?;
            }
            let combined = self.capture_residual(&shortcut, &hidden, count, &mut cb).map_err(|e| e.to_string())?;
            cb.encode_copy(&combined, result.subrange_mut(lo..hi));
            drop(combined);
            drop(hidden);
            drop(shortcut);
            cb.end_encoding().submit().wait_until_completed().map_err(|e| format!("attention MLP {start}: {e}"))?;
        }
        Ok(result)
    }

    pub(crate) fn encode_code_prompt(
        &self,
        mut input: B::GlobalBuffer,
        batch: &BatchTopology,
        context: &B::Context,
    ) -> Result<B::GlobalBuffer, Box<dyn std::error::Error>> {
        use crate::backends::common::{CommandBufferExecutable, CommandBufferPending, Context};
        let n = batch.size();
        let positions = batch.heights().collect::<Vec<_>>();
        let row_bytes = input.size() / n as usize;
        for (layer, rope_index) in self.layers.iter() {
            assert!(layer.kv_source_layer_index.is_none() && layer.ple_projection.is_none());
            context.release_cached_buffers();
            if n >= 4352
                && let Some(attention) = (layer.mixer.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<crate::encodable_block::mixer::attention::Attention<B>>()
            {
                input = self
                    .encode_code_attention_layer(layer, attention, &input, n, *rope_index, context)
                    .map_err(|e| format!("code-decision layer {}: {e}", layer.layer_index))?;
                continue;
            }
            let mut state = layer.mixer.create_empty_state(Some(n), context).map_err(|e| e.to_string())?;
            let causal_recurrence = (layer.mixer.as_ref() as &dyn std::any::Any)
                .is::<crate::encodable_block::mixer::delta_net::DeltaNet<B>>();
            let mut next = context.create_buffer(input.size()).map_err(|e| e.to_string())?;
            let mut offset = 0;
            while offset < n {
                // This is an execution tile, not a context limit. Carry the
                // causal convolution/recurrent state through every tile; full
                // bidirectional attention always sees the complete sequence.
                // Align to the native 64-token GDN chunk and avoid a final
                // <256 tile switching to a different recurrent kernel.
                let remaining = n - offset;
                let count = if causal_recurrence && remaining >= 4352 {
                    4096
                } else {
                    remaining
                };
                let nodes = (0..count)
                    .map(|i| crate::backends::common::gpu_types::trie::TrieNode {
                        trie_start: i,
                        trie_end: count - 1,
                        height: i,
                    })
                    .collect::<Vec<_>>();
                let tile = BatchTopology::new(&nodes, true);
                state.prepare(offset, count).map_err(|e| e.to_string())?;
                context.release_cached_buffers();
                let mut cb =
                    context.create_command_buffer(Some("code decision layer tile"), None).map_err(|e| e.to_string())?;
                let start = offset as usize * row_bytes;
                let end = (offset + count) as usize * row_bytes;
                let mut hidden = cb.allocate_scratch(end - start).map_err(|e| e.to_string())?;
                cb.encode_copy(input.subrange(start..end), &mut hidden);
                let mut shortcut = cb.allocate_scratch(end - start).map_err(|e| e.to_string())?;
                cb.encode_fill(&mut shortcut, 0);
                let rope = rope_index
                    .map(|i| {
                        PrecalculatedRoPE::<B>::precalculate_with_precision(
                            &self.ropes[i],
                            &positions[offset as usize..(offset + count) as usize],
                            self.bf16_rope_tables,
                            &mut cb,
                        )
                    })
                    .transpose()
                    .map_err(|e| e.to_string())?;
                let hidden = layer
                    .encode(
                        hidden,
                        &mut shortcut,
                        None::<&B::ScratchBuffer>,
                        rope.as_ref(),
                        &tile,
                        Some(MaybeMut::Mut(state.as_mut())),
                        &mut cb,
                    )
                    .map_err(|e| e.to_string())?;
                let combined = self.capture_residual(&shortcut, &hidden, count, &mut cb).map_err(|e| e.to_string())?;
                cb.encode_copy(&combined, next.subrange_mut(start..end));
                if causal_recurrence {
                    state.encode_accept(&(0..count).collect::<Vec<_>>(), &mut cb).map_err(|e| e.to_string())?;
                }
                drop(combined);
                drop(hidden);
                drop(shortcut);
                drop(rope);
                cb.end_encoding().submit().wait_until_completed().map_err(|e| {
                    format!("code-decision layer {} tokens {offset}..{}: {e}", layer.layer_index, offset + count)
                })?;
                offset += count;
            }
            input = next;
            drop(state);
        }
        context.release_cached_buffers();
        let mut cb =
            context.create_command_buffer(Some("code decision final norm"), None).map_err(|e| e.to_string())?;
        let mut zeros = cb.allocate_scratch(input.size()).map_err(|e| e.to_string())?;
        cb.encode_fill(&mut zeros, 0);
        let norm = self
            .output_norm
            .encode(&input, n - 1, 1, (!self.layers.is_empty()).then_some(&mut zeros), &mut cb)
            .map_err(|e| e.to_string())?;
        let mut result = context.create_buffer(norm.size()).map_err(|e| e.to_string())?;
        cb.encode_copy(&norm, &mut result);
        drop(norm);
        drop(zeros);
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        drop(input);
        context.release_cached_buffers();
        Ok(result)
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        let mut hidden = input;
        let layer_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.layers.len()
        };

        let mut shortcut = command_buffer.allocate_scratch(hidden.size()).map_err(TransformerEncodeError::Backend)?;
        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        let context_length = state.as_ref().map(|state| state.context_length).unwrap_or(0);
        let token_positions = batch_dim.heights().map(|rel_pos| context_length + rel_pos).collect::<Box<[u32]>>();

        let precalculated_ropes = self
            .ropes
            .iter()
            .map(|rope_config| {
                PrecalculatedRoPE::precalculate_with_precision(
                    rope_config,
                    &token_positions,
                    self.bf16_rope_tables,
                    command_buffer,
                )
            })
            .collect::<Result<Box<[_]>, B::Error>>()
            .map_err(TransformerEncodeError::Backend)?;

        for (layer, layer_rope_index) in self.layers.iter().take(layer_count) {
            let precalculated_rope = layer_rope_index.map(|i| &precalculated_ropes[i]);

            let layer_state = if let Some(state) = &mut state {
                Some(match &mut state.layer_states[layer.layer_index as usize] {
                    TransformerLayerStateType::Owned(layer_state) => MaybeMut::Mut(layer_state.as_mut()),
                    TransformerLayerStateType::Shared(owned_layer_index) => {
                        let TransformerLayerStateType::Owned(owned_layer) =
                            &state.layer_states[*owned_layer_index as usize]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        MaybeMut::Const(owned_layer.as_ref())
                    },
                })
            } else {
                None
            };

            hidden = layer.encode(
                hidden,
                &mut shortcut,
                per_layer_inputs,
                precalculated_rope,
                batch_dim,
                layer_state,
                command_buffer,
            )?;

            if let (Some(hidden_features), Some(indices)) = (&mut hidden_features, hidden_feature_layer_indices) {
                for (feature_index, &layer_index) in indices.iter().enumerate() {
                    if layer_index == layer.layer_index {
                        let feature = self
                            .capture_residual(&shortcut, &hidden, batch_dim.size(), command_buffer)
                            .map_err(TransformerEncodeError::Backend)?;
                        hidden_features[feature_index] = Some(feature);
                    }
                }
            }
        }

        let hidden_features = hidden_features.map(|hidden_features| {
            hidden_features
                .into_iter()
                .enumerate()
                .map(|(feature_index, feature)| {
                    feature.unwrap_or_else(|| {
                        panic!(
                            "requested hidden feature for missing layer {}",
                            hidden_feature_layer_indices.unwrap()[feature_index]
                        )
                    })
                })
                .collect::<Box<[_]>>()
        });

        let Some(output_range) = output_range else {
            return Ok(TransformerEncodeOutput {
                output: None,
                hidden_features,
            });
        };

        let output_normalized = self
            .output_norm
            .encode(
                &hidden,
                output_range.start,
                output_range.end - output_range.start,
                (!self.layers.is_empty()).then_some(&mut shortcut),
                command_buffer,
            )
            .map_err(TransformerEncodeError::Backend)?;

        Ok(TransformerEncodeOutput {
            output: Some(output_normalized),
            hidden_features,
        })
    }
}
