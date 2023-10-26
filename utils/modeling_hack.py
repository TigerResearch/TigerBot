import math
from typing import Optional, Tuple

import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaLinearScalingRotaryEmbedding, \
    LlamaRotaryEmbedding


def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


# Copied from
# https://github.com/huggingface/text-generation-inference/blob/7402a355dcbf9ffe7a0b2a788f2062aa9e0a3ed5/server/text_generation_server/utils/layers.py#L764-L806
class LlamaYaRNRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaYaRNRotaryEmbedding extended with YaRN NTK scaling from [paper](https://arxiv.org/pdf/2309.00071.pdf)"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=8.0,
                 extrapolation_factor=1., attn_factor=1., beta_fast=32, beta_slow=1):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.extrapolation_factor = extrapolation_factor
        self.scaling_factor = scaling_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        super().__init__(self.dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)

        if seq_len <= self.max_position_embeddings:
            seq_len = self.max_position_embeddings

        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            inv_freq_extrapolation = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )

            freqs = 1.0 / inv_freq_extrapolation
            inv_freq_interpolation = 1.0 / (self.scaling_factor * freqs)
            low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base,
                                              self.max_position_embeddings)
            inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(
                device)) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

            # Get n-d magnitude scaling corrected for interpolation
            mscale = ((0.1 * math.log(self.scaling_factor) + 1.0) if self.scaling_factor > 1 else 1) * self.attn_factor
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            mscale = 1

        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        # Don't do einsum, it converts fp32 to fp16
        # freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) * mscale

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Return cos and sin for the asked position ids
        """
        if not (seq_len <= self.max_seq_len_cached == self.max_position_embeddings) or \
                self.cos_cached.device != x.device:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        print('_set_cos_sin_cache', device)
        if seq_len <= self.max_position_embeddings:
            seq_len = self.max_position_embeddings

        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
        else:
            base = self.base
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if not (seq_len <= self.max_seq_len_cached == self.max_position_embeddings):
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# hack llama attention init rope

def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == 'yarn':
            self.rotary_emb = LlamaYaRNRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


LlamaAttention._init_rope = _init_rope


def get_model(model_path: str, rope_scaling: Optional[str] = None, rope_factor: float = 8.0, ) -> Tuple[
    transformers.AutoModelForCausalLM, transformers.AutoTokenizer, transformers.GenerationConfig]:
    if rope_scaling is None:
        rope_config = None
    else:
        rope_config = {"type": rope_scaling, "factor": rope_factor}

    print(f"Loading model from {model_path}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto',
                                                              rope_scaling=rope_config)
    print(model.model.layers[0].self_attn.rotary_emb)
    print("Done")

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    print("Done")

    print(f"Loading generation config from {model_path}...")
    generation_config = transformers.GenerationConfig.from_pretrained(model_path)
    print("Done")

    return model, tokenizer, generation_config
