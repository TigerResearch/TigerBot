import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half
import transformers


class LlamaRotaryEmbeddingFixed(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

def _set_cos_sin_cache_Linear_fixed(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
    t = t / self.scaling_factor

    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)

    self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

def _set_cos_sin_cache_DynamicNTK_fixed(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len

    if seq_len > self.max_position_embeddings:
        base = self.base * (
            (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(q.dtype)
    k_embed = k_embed.to(k.dtype)
    return q_embed, k_embed

def replace_embedding():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbeddingFixed
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding._set_cos_sin_cache = _set_cos_sin_cache_Linear_fixed
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding._set_cos_sin_cache = _set_cos_sin_cache_DynamicNTK_fixed
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
