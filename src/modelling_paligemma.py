import torch
from torch import nn
from typing import List
from modelling_gemma import GemmaConfig, GemmaForCausalLM
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel 

class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100, 
        pad_token_id=None,
        image_token_id=256000,
        vocab_size=257152,
        projection_dim=2048,
        **kwargs,
    ):
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.vocab_size = self.text_config.vocab_size
        self.projection_dim = projection_dim
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    x = x.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)
    return x # [b, num_attention_heads, seq_len, head_dim]

# https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L437
class KVCache:
    """
    Stores the key and value states as a list of tensors, one for each layer. Grows 
    dynamically as more tokens are generated.
    """
    def __init__(self):
        self.k_cache: List[torch.Tensor] = [] # [k_state_layer{0}, k_state_layer{1}, ..., k_state_layer{N-1}]
        self.v_cache: List[torch.Tensor] = [] # [v_state_layer{0}, v_state_layer{1}, ..., v_state_layer{N-1}]
    
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.k_cache):
            return (self.k_cache[layer_idx], self.v_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self.k_cache)} layers, attempted to access layer with index {layer_idx}")

    def get_seq_length(self, layer_idx=0) -> int:
        return 0 if len(self.k_cache)==0 else self.k_cache[layer_idx].shape[-2]
        
    def update(self, k_state: torch.Tensor, v_state: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.k_cache) <= layer_idx:
            self.k_cache.append(k_state)
            self.v_cache.append(v_state)
        else:
            self.k_cache[layer_idx] = torch.cat( [self.k_cache[layer_idx], k_state], dim=-2 ) 
            self.v_cache[layer_idx] = torch.cat( [self.v_cache[layer_idx], v_state], dim=-2 ) 
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
   

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.embed_dim, config.projection_dim, bias=True)
    
    def forward(self, hidden_states):
        return self.linear(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, hidden_size]
        

# https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/modeling_paligemma.py#L231
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config) 
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def tie_weights(self):
        # The decoder's output projection layer maps the hidden states (contextualized embedding)
        # to the token-ids/vocabulary. This is exactly opposite of what the embedding layer does.
        # We can share parameters of these layers (weight tying) to reduce overall model parameters.
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, input_ids, input_embeds, image_embeds, attention_mask, labels, token_type_ids, cache_position
    ):
        _, _, hidden_size = image_embeds.shape # [b, num_patches, hidden_size]
        batch_size, seq_len = input_ids.shape  # [b, seq_len]
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min

        image_embeds = image_embeds / (hidden_size**0.5) # [b, num_patches, hidden_size] 

        txt_mask = (input_ids != self.config.image_token_id) & (input_ids != self.config.pad_token_id) # [b, seq_len]
        img_mask  = input_ids == self.config.image_token_id # [b, seq_len]
        pad_mask  = input_ids == self.config.pad_token_id # [b, seq_len]

        # Expand masks to match embeding dimension
        txt_mask_expanded = txt_mask.unsqueeze(-1).expand(-1, -1, hidden_size).to(device) # [b, seq_len] -> [b, seq_len, 1] -> [b, seq_len, hidden_size]
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, hidden_size).to(device) # [b, seq_len] -> [b, seq_len, 1] -> [b, seq_len, hidden_size]
        img_mask_expanded = img_mask.unsqueeze(-1).expand(-1, -1, hidden_size).to(device) # [b, seq_len] -> [b, seq_len, 1] -> [b, seq_len, hidden_size]

        merged_embeds = torch.zeros((batch_size, seq_len, hidden_size), dtype=dtype, device=device) # [b, seq_len, hidden_size]
        merged_embeds = torch.where(txt_mask_expanded, input_embeds, merged_embeds)
        merged_embeds = torch.where(pad_mask_expanded, torch.zeros_like(merged_embeds), merged_embeds)
        merged_embeds = merged_embeds.masked_scatter(img_mask_expanded, image_embeds.to(device=device, dtype=dtype))
        merged_embeds = torch.where(pad_mask_expanded, torch.zeros_like(merged_embeds), merged_embeds)   

        ## --------------------- Create position-ids for Rotary Positional Embedding (RoPE) --------------------- ## 
        if attention_mask is not None:
            # Create position-ids using attention mask. For padding tokens use 1 as position id.
            # Process:  
            #   attention_mask == [[0, 0, 1, 1, 1, 1, 1]]                               ; [b, seq_len]
            #   attention_mask.cumsum(-1) -> [[0, 0, 1, 2, 3, 4, 5]]                    ; [b, seq_len]
            #   attention_mask.cumsum(-1).masked_fill_(...) -> [[1, 1, 1, 2, 3, 4, 5]]  ; [b, seq_len]
            #  
            # Note: This does not introduce extra/garbage information, since padding tokens are
            #       anyways masked out and not used during the attention process/loss calculation.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1) # [b, seq_len]
        else:
            position_ids = None  

        ## ------------------------------------------ Create masks --------------------------------------------- ##
        # If *training-with-teacher-forcing* stage, create a mask allowing full attention on
        # image tokens + prefixes while simultaneously allowing causal attention on suffixes.
        if token_type_ids is not None and labels is not None:
            target_len = cache_position[-1] + 1 # length of the current input text slice
            causal_mask = torch.full(
                (seq_len, target_len), fill_value=min_dtype, dtype=dtype, device=device
            ) # min_dtype at all positions
            # Enforce that each token attends to only the prior tokens and itself
            if seq_len != 1: # TODO: Figure out when seq_len == 1
                causal_mask = torch.triu(causal_mask, diagonal=1) # -inf at positions of future tokens
            causal_mask *= torch.arange(target_len, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1) # add batch and head dimension ; min_dtype at positions of future tokens, 0 at rest
            
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1] # seq_len
                # Merge attention_mask (with 0 at positions of padding tokens) with the causal_mask (look-ahead mask).
                # Wherever attention_mask == 1 (real token), 1 is added to that position, irrespective of what the  
                # causal_mask's value may be at that position (0 or min_dtype). Valid positions become 1, 
                # padding positions remain 0 and already‐masked positions remain very negative.
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                # Wherever token_type_ids == 0, i.e. at positions of image/text tokens, 
                # assign 0 as value at those positions. 
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    mask=token_type_ids[:, None, None, :].to(causal_mask.device) == 0, value=0
                )
                # Wherever padding tokens, set those position's value to be min_dtype
                padding_mask = padding_mask == 0 # get positions of padding tokens
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                ) # min_dtype: at positions of padding tokens and future tokens |  0: at positions of valid real tokens
            
            # Tokens with id == ignore_index won't contribute to the loss
            final_labels = torch.full(
                (batch_size, seq_len), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
            final_labels = torch.where(input_ids != self.pad_token_id, labels, final_labels) # (b, seq_len) ; ignore_index at places of pad token
        else: # if pre-filling stage of inference
            # Create a mask that ensures attention is only computed between valid (non-pad) tokens      
            causal_mask = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(-1) # [b, 1, seq_len, seq_len] ; 1*1=1 for two real tokens; anything with a pad yields 0.
            causal_mask = torch.where(causal_mask == 0, min_dtype, 0) # [b, 1, seq_len, seq_len] ; min_dtype value at padding positions
            causal_mask = causal_mask.to(dtype).expand(-1, self.config.text_config.num_key_value_heads, -1, -1) # [b, num_key_value_heads, seq_len, seq_len]
            # We don't require labels during inference
            final_labels = None
        return merged_embeds, causal_mask, final_labels, position_ids
    
    def forward(self, input_ids, pixel_values, attention_mask, labels, token_type_ids, cache_position, kv_cache):
        """
        input_ids         [b, seq_len] ; tokenized sequences
        pixel_values      [b, c, h, w] ;
        attention_mask    [b, seq_len] ; returned by tokenizer (0: padding tokens, 1: rest)
        labels            [b, seq_len] ;
        token_type_ids    [b, seq_len] ; (0: image/prefix token, 1: suffix token) 
        cache_position                 ; None initially      
        kv_cache          self.k_cache==[], v_cache==[]
        """         
        input_attention_mask = attention_mask # store the original attention mask. 

        # Gemma's embedding module wasn't trained on the image placeholder token, i.e. <image>
        # Therefore this token's generated embedding will be junk. These embeddings will later
        #  be replaced by the contextualized image patch features returned by image encoder.
        input_embeds = self.language_model.get_input_embeddings(input_ids) # [b, seq_len] -> [b, seq_len, hidden_size]
        
        # The following if-block executes during *training* and *pre-filling stage of inference*
        if pixel_values is not None and input_ids.shape[1] != 1: 
            # Get contextualized image patch embedding. Then resize them to match the text token's embedding dimension
            image_embeds = self.vision_tower(pixel_values.to(input_embeds.dtype)) # [n, c, h, w] -> [b, num_patches, embed_dim]
            image_embeds = self.multi_modal_projector(image_embeds) # [b, num_patches, embed_dim] -> [b, num_patches, hidden_size] 
            assert input_embeds.shape[-1] == image_embeds.shape[-1], f"Mismatch b/w image patch embedding dim and text token embedding dim: {input_embeds[-1]} != {image_embeds[-1]}"
            
            if cache_position is None:
                cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device) # (seq_len,)

            input_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                input_ids,
                input_embeds, 
                image_embeds, 
                attention_mask, 
                labels, 
                token_type_ids, 
                cache_position
            )
        else: # if token generation phase (after prefilling)
            if pixel_values is not None and kv_cache is not None and input_ids.shape[1] == 1:
                first_layer_past_key_value = kv_cache[0][0][:, :, :, 0] # [b, num_heads, seq_len_cached] ; first layer's key tensor
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)
                # Get cached sequence length (image tokens + prefix token + output_tokens_produced_so_far)
                target_seqlen = kv_cache.get_seq_length() + 1 # cache_position[-1] + 1 
                
                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], target_seqlen - attention_mask.shape[1] + 1), 
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ) # [b, (seq_len_cached - current_seq_len + 1)] -> [b, Δ] ; usually (Δ == 1)
                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses PaliGemma+ Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1) # [b, seq_len_cached + Δ] ; binary mask

                # Compute the position ID for the current input query token. We do this by counting
                # how many tokens are "active" (i.e. not masked) in the attention mask. Since position 
                # indices are 0-based, we need to subtract 1.
                # Ex.
                #   attention_mask = [
                #     [ 1, 1, 1, 1 ],    → 4 active tokens → position_ids = [3]
                #     [ 0, 0, 1, 1 ]     → 2 active tokens → position_ids = [1]
                #   ]            \
                #                ↓ 
                #  (for current input query token)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1 # [b, 1]

                # 1. Since tokens are generated autoregressively, at each step t, only one query token 
                #    (most recent generated token) is passed in. 
                # 2. KV-cache contains all past tokens. 
                # 3. The current token can only attend to past tokens (those already in the cache). 
                #
                # => We don't need a causal mask with -inf here because the model is only attending to
                #    the past tokens (masking, if required would already have been done at appropriate
                #    positions, mostly at padding indexes, during the pre-filling stage) and there are
                #    no future tokens to mask. 
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [b, 1, 1, seq_len_cached + Δ]

        attention_mask = attention_mask.to(input_embeds.dtype)
    
        outputs = self.language_model(
            input_embeds, attention_mask, position_ids, kv_cache, cache_position
        ) # dict (loss, logits, kv_cache)

        logits = outputs['logits'] # [b, seq_len, vocab_size]
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Discard last prediction as there is no next token to compare it to
            shift_logits = logits[..., :-1, :]
            # Discard first label token since the model begins predicting from the second token, using the first as context.
            shift_labels = labels[..., 1:]
            
            if input_attention_mask is not None:
                shift_attention_mask = input_attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()

            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(flat_logits, flat_labels)
        return (loss,) + outputs if loss is not None else outputs
    