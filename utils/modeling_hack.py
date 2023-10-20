import math
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaLinearScalingRotaryEmbedding, \
    LlamaDynamicNTKScalingRotaryEmbedding, LlamaRotaryEmbedding


# Copied from
# https://github.com/huggingface/text-generation-inference/blob/7402a355dcbf9ffe7a0b2a788f2062aa9e0a3ed5/server/text_generation_server/utils/layers.py#L764-L806
class LlamaYaRNRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaYaRNRotaryEmbedding extended with YaRN NTK scaling from [paper](https://arxiv.org/pdf/2309.00071.pdf)"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, attn_factor=1.):
        self.scaling_factor = scaling_factor
        self.attn_factor = attn_factor
        self.max_seq_len_cached = 0
        super().__init__(dim, max_position_embeddings, base, device)

    def find_correction_dim(self, num_rotations):
        return (self.dim * math.log(self.max_position_embeddings / (num_rotations * 2 * math.pi))) / (
                2 * math.log(self.base))

    # Find dim range bounds based on rotations
    def find_correction_range(self, low_rot, high_rot):
        low = math.floor(self.find_correction_dim(low_rot))
        high = math.ceil(self.find_correction_dim(high_rot))
        return max(low, 0), min(high, self.dim - 1)  # Clamp values just in case

    def linear_ramp_mask(self, min, max):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(self.dim // 2, dtype=torch.float) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def get_mscale(self):
        if self.scaling_factor <= 1:
            return 1.0
        return 0.1 * math.log(self.scaling_factor) + 1.0

    def _set_cos_sin_cache(self, seq_len, device, dtype):

        if seq_len > self.max_seq_len_cached:
            if seq_len > self.max_position_embeddings:
                inv_freq_extrapolation = 1.0 / (
                        self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
                freqs = 1.0 / inv_freq_extrapolation
                inv_freq_interpolation = 1.0 / (self.scaling_factor * freqs)
                low, high = self.find_correction_range(self.beta_fast, self.beta_slow)
                inv_freq_mask = (1 - self.linear_ramp_mask(low, high).float().to(
                    device)) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
                inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
                # Get n-d magnitude scaling corrected for interpolation
                if self.scaling_factor <= 1:
                    mscale = 1.0
                else:
                    mscale = 0.1 * math.log(self.scaling_factor) + 1.0
                mscale *= self.attn_factor
            else:
                base = self.base * (
                        (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
                mscale = 1.

            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", (emb.cos() * mscale)[None, None, :, :].to(dtype).to(device),
                                 persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * mscale)[None, None, :, :].to(dtype).to(device),
                                 persistent=False)


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
