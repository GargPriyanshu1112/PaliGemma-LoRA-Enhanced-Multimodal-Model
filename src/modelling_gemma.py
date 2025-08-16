import math
import torch
from torch import nn
import torch.nn.functional as F

class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10_000):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.attention_scaling = 1.0 # TODO: Verify value from official impl
        # Pre-compute frequencies
        #     θ = { θi = 10000^(-2(i-1)/d), i ∈ [1, 2, ..., d/2] }    ; from research paper (RoFormer)
        #  => θ = { θi = 1 / (10000^(2i/d), i ∈ [0, 1, ..., d//2] }   ; subtract 1
        #  => θ = { θi = 1 / (10000^(i/d),  i ∈ [0, 2, ..., d] }      ; multiple and divide by 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) # [b, head_dim//2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [b, 1, seq_len]

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [b, head_Dim//2, 1] @ [b, 1, seq_Len] -> [b, head_dim//2, seq_Len] -> [b, seq_Len, head_dim//2]
            emb = torch.cat((freqs, freqs), dim=-1) # [b, seq_Len, head_dim] ; for half-rotation
            cos = emb.cos() * self.attention_scaling # [b, seq_Len, head_dim//2]
            sin = emb.sin() * self.attention_scaling # [b, seq_Len, head_dim//2]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
    
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    unsqueeze_dim (`int`, *optional*, defaults to 1):
        The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
        sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
        that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
        k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
        cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
        the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x, n_rep):
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    x = x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    return x

   
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * F.gelu(self.gate_proj(x), approximate='tanh'))
    

# TODO: 3:38:00 KV cache + GQA.
# https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/gemma/modeling_gemma.py#L191
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # maps each decoder layer to its dedicated KV-cache instance
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        assert self.hidden_size % self.head_dim == 0
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias)
    
    def forward(self, hidden_states, attention_mask, position_embeddings, kv_cache):
        input_shape = hidden_states.shape[: -1] # [b, seq_len]
        hidden_shape = (*input_shape, -1, self.head_dim) # [b , seq_len, -1, head_dim]

        # [b, q_len, hidden_size] -> [b, q_len, num_attention_heads * head_dim] -> [b, num_attention_heads, q_len, head_dim]
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [b, q_len, hidden_size] -> [b, q_len, num_key_value_heads * head_dim] -> [b, num_key_value_heads, q_len, head_dim]
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [b, q_len, hidden_size] -> [b, q_len, num_key_value_heads * head_dim] -> [b, num_key_value_heads, q_len, head_dim]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # kv-cache isn't used during training (is set to None) but initialized at inference time
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)
        
        # Repeat K and V heads to match no. of heads in Q
        k = repeat_kv(k, self.num_attention_heads // self.num_key_value_heads)
        v = repeat_kv(v, self.num_attention_heads // self.num_key_value_heads)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) # [b, num_attention_heads, q_len, q_len]
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k.shape[-2]]
            attn_weights += causal_mask # adds -inf to masked positions to prevent attention
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_outputs = torch.matmul(attn_weights, v) # [b, num_attention_heads, q_len, head_dim]
        attn_outputs = attn_outputs.transpose(1, 2).contiguous() # [b, q_len, num_attention_heads, head_dim]
        attn_outputs = attn_outputs.reshape(*input_shape, -1).contiguous() # [b, q_len, num_attention_heads * head_dim]
        
        # Each attention head captures different aspects of the input.
        # Use a linear layer to fuse these diverse representations. 
        attn_outputs = self.o_proj(attn_outputs) # [b, q_len, hidden_size]
        return attn_outputs, attn_weights
              

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(self, hidden_states, attention_mask, position_embeddings, kv_cache):
        residual = hidden_states # [b, q_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states) # [b, q_len, hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states, attention_mask, position_embeddings, kv_cache
        ) # [b, q_len, hidden_size]
        hidden_states = residual + hidden_states # [b, q_len, hidden_size]
        residual = hidden_states # [b, q_len, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states) # [b, q_len, hidden_size]
        hidden_states = self.mlp(hidden_states) # [b, q_len, hidden_size]
        hidden_states = residual + hidden_states # [b, q_len, hidden_size]
        return hidden_states
    
# TODO: Go through impl
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


 # --------------------------------------------------------------------------------------------------------------------------------------- # 
def _preprocess_mask_arguments(input_embeds, attention_mask, kv_cache, cache_position, layer_idx):
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)
    if kv_cache is not None:
        kv_length, kv_offset = kv_cache.get_mask_sizes(cache_position, layer_idx)
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0
    return False, attention_mask, kv_length, kv_offset
 # --------------------------------------------------------------------------------------------------------------------------------------- #


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id # embedding vector associated with padding_idx isn't updated during training
        )
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
    
    def forward(self, input_embeds, attention_mask, position_ids, kv_cache):            
        hidden_states = input_embeds # [b, seq_len, hidden_size]

        # Create position embeddings to be shared across decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)      

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer # [b, seq_len, hidden_size]

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask, position_embeddings, kv_cache
            ) # [b, seq_len, hidden_size] ; contextualized embeddings
        hidden_states = self.norm(hidden_states) # [b, seq_len, hidden_size]
        return hidden_states, kv_cache

# https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/gemma/modeling_gemma.py#L476
class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight # point both modules to the same memory

    def get_input_embeddings(self, input_ids):
        return self.model.embed_tokens(input_ids)
    
    def forward(self, input_embeds, attention_mask, position_ids, kv_cache):
        hidden_states, kv_cache = self.model(
            input_embeds, attention_mask, position_ids, kv_cache
        ) # [b, seq_len, hidden_size] ; contextualized embeddings
        logits = self.lm_head(hidden_states) # [b, seq_len, vocab_size]
        
        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data['kv_cache'] = kv_cache
        return return_data
