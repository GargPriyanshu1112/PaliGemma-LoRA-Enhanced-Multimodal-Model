import torch
from modelling_paligemma import KVCache

def _sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # [b, vocab_size]
    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_cumsum > p
    mask[..., 0] = False # keep at least the highest-prob token
    probs_sort = probs_sort.masked_fill(mask, 0.0) # zero out masked probs
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True) # renormalize
    # Sample a token from the top p distribution
    next_token_idx = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token_idx)
    return next_token

@torch.inference_mode()
def generate(
    model,
    processor,
    pixel_values,
    input_ids,
    attention_mask,
    max_tokens_to_generate,
    device,
    temperature=1.0,
    top_p=1.0,
    do_sample=False,
):
    if do_sample and temperature <= 0:
        raise ValueError("Temperature must be > 0 for sampling")
    
    model.eval()

    kv_cache = KVCache()
    pad_token  = processor.tokenizer.pad_token_id
    stop_token = processor.tokenizer.eos_token_id
    unfinished_seqs = torch.ones(input_ids.shape[0], dtype=torch.long, device=device) # to keep track of which sequences in the batch are still being generated

    generated_tokens = []    

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids, pixel_values, attention_mask, None, None, kv_cache
        )
        kv_cache = outputs['kv_cache'] # updated kv-cache
        next_token_logits = outputs['logits'][:, -1, :]
        if do_sample:
            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(probs, top_p) # use top_p sampling
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # greedy sampling
        
        # If prediction for some sequences in the batch is finished early, replace their next tokens with pad token.
        next_token = next_token * unfinished_seqs.view(-1, 1) + pad_token * (1 - unfinished_seqs).view(-1, 1)

        generated_tokens.append(next_token)
        
        input_ids = next_token
        unfinished_seqs = unfinished_seqs & (next_token != stop_token).long().view(-1)

        if unfinished_seqs.max() == 0: # if all sequences have finished, stop generation
            break

    generated_tokens = torch.cat(generated_tokens, dim=1)
    decoded_seqs = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded_seqs