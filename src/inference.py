import torch
from PIL import Image
from processing_paligemma import PaliGemmaProcessor
from modelling_paligemma import KVCache
from utils import get_torch_device, move_to_device, load_tokenizer, load_model

def _sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # [b, vocab_size]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (probs_sum - probs_sort) returns the cumulative sum before each token.
    # E.g.,
    #   probs_sort = [0.4, 0.3, 0.2, 0.1]
    #   probs_sum  = [0.4, 0.7, 0.9, 1.0]
    #   probs_sum - probs_sort => [0.0, 0.4, 0.5, 0.9]
    #
    # The tokens that would cause cumulative sum to exceed 'p' are then masked.
    mask = (probs_sum - probs_sort) > p
    probs_sort[mask] = 0.0 # wherever False, assign 0.0
    # Redistribute probabilities such that they sum to 1 by dividing each 
    # element of the tensor by `probs_sort.sum(dim=-1, keepdim=True)`
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # in-place
    # Sample a token (its index) from the top p distribution
    next_token_idx = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token_idx)
    return next_token

@torch.inference_mode()
def generate(
    model,
    processor,
    prompt_l,
    image_path_l,
    max_tokens_to_generate,
    temperature,
    top_p,
    do_sample,
    device=None
):
    if not device:
        device = get_torch_device()

    model.eval()

    model_inputs = processor(
        [Image.open(path) for path in image_path_l],
        prompt_l, 
    )
    model_inputs = move_to_device(model_inputs, device)
    
    input_ids = model_inputs['input_ids']
    pixel_values = model_inputs['pixel_values']
    attention_mask = model_inputs['attention_mask']
    labels = None
    token_type_ids = None
    cache_position = None
    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    
    generated_tokens = []
    for _ in range(max_tokens_to_generate):
        # print("\ncache length", kv_cache.get_seq_length())
        outputs = model(
            input_ids, pixel_values, attention_mask, labels, token_type_ids, cache_position, kv_cache
        )
        if kv_cache is not None: 
            kv_cache = outputs['kv_cache'] # updated kv-cache
        next_token_logits = outputs['logits'][:, -1, :]
        # Sample the next token
        if do_sample: # use sampling to get next token instead of the greedy strategy
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1) # use temperature
            next_token = _sample_top_p(next_token_logits, top_p) # use top_p sampling
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0) # remove batch dim
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded_seqs = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_seqs

if __name__ == '__main__':
    model_path = r"C:\Users\ADMIN\Downloads\paligemma_3b_pt_224_model_files"
    prompt_l = ["Name the monument."]
    image_path_l = [r"C:\Users\ADMIN\Downloads\Taj-Mahal-Agra-India.png"]
    max_tokens_to_generate=100 
    temperature=0.8
    top_p=0.9
    do_sample=True
    
    device = get_torch_device()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)

    print("Loading model...")
    model = load_model(model_path, device)
    model.eval()

    image_size = model.config.vision_config.image_size
    num_image_tokens = model.config.vision_config.num_image_tokens
    processor = PaliGemmaProcessor(image_size, num_image_tokens, tokenizer)

    print('Generating output...')
    with torch.no_grad():
        outputs = generate(
            model,
            processor,
            prompt_l,
            image_path_l,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            device
        )
    print(outputs)