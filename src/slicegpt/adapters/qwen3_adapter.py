# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config, Qwen3DecoderLayer, Qwen3RMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter

class CustomQwen3RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).sum(-1, keepdim=True) / self.hidden_size
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{[self.hidden_size]}, eps={self.variance_epsilon}"

class CompressedQwen3DecoderLayer(Qwen3DecoderLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_mlp_interface = torch.nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: tuple[Tensor] | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states
        # Fully Connected
        hidden_states = self.attn_mlp_interface(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states
        return hidden_states

class Qwen3LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen3DecoderLayer) -> None:
        super().__init__()
        self._layer: Qwen3DecoderLayer = layer
        if isinstance(layer.self_attn.q_norm, Qwen3RMSNorm):
            layer.self_attn.q_norm.skip_replace = True
        if isinstance(layer.self_attn.k_norm, Qwen3RMSNorm):
            layer.self_attn.k_norm.skip_replace = True

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen3ModelAdapter(ModelAdapter):
    _supported_models = [
        "Qwen/Qwen3",
    ]
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self._model: Qwen3ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen3Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Qwen3DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen3RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen3LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedQwen3DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Same as Llama
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not any(model_name.startswith(name) for name in cls._supported_models):
            return None

        from_pretrained_kwargs = {
            "dtype": dtype,
            "token": token,
            "local_files_only": local_files_only,
            "use_sliding_window": False,    # disables sliding window attention -> full attention only
        }
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, **from_pretrained_kwargs 
        )

        return Qwen3ModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not any(model_name.startswith(name) for name in cls._supported_models):
            return None

        class UninitializedQwen3ForCausalLM(Qwen3ForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        from_pretrained_kwargs = {
            "dtype": dtype,
            "token": token,
            "local_files_only": local_files_only,
            "use_sliding_window": False,    # disables sliding window attention -> full attention only
        }
        config = Qwen3Config.from_pretrained(
            model_path, **from_pretrained_kwargs
        )
        model = UninitializedQwen3ForCausalLM(config)
        model = model.to(dtype=dtype)

        return Qwen3ModelAdapter(model)

if __name__ == "__main__":
    print("Testing uninitialized Qwen3-0.6B")
    model = Qwen3ModelAdapter._from_uninitialized("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B")
    print(70*'*')
    print(model.config)
    print(70*'*')
    print(model.model)