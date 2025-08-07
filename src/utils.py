import re
import os
import json
import torch
from glob import glob
from safetensors import safe_open
from transformers import AutoTokenizer
from modelling_paligemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration

def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, device, framework='pt'):
    if not device:
        device = get_torch_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right')
    
    safetensors_file_l = glob(os.path.join(model_path, '*.safetensors'))
    tensors = {}
    for safetensors_file in safetensors_file_l:
        with safe_open(safetensors_file, framework=framework, device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        print(f"Successfully loaded {os.path.basename(safetensors_file)}")
    
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config_file = json.load(f)
        config = PaliGemmaConfig(**config_file)
    
    model = PaliGemmaForConditionalGeneration(config)
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()
    return model, tokenizer

def json2token(obj: dict):
    if isinstance(obj, dict):
        output = ''
        for k, v in obj.items():
            output += (rf"<s_{k}>" + json2token(v) + rf"</s_{k}>")
        return output
    else:
        obj = str(obj)
    return obj

def token2json(tokens):
    output = {}
    while tokens:
        start_match = re.search(r"<s_(.*?)>", tokens)
        if start_match is None:
            break
        key = start_match.group(1)
        start_token = f"<s_{key}>"
        end_token = f"</s_{key}>"
        pattern = re.compile(
            re.escape(start_token) + r"(.*?)" + re.escape(end_token),
            re.DOTALL
        )
        match = pattern.search(tokens)
        
        if not match:
            tokens = tokens[start_match.end(): ]
            continue
        content = match.group(1).strip()

        if r"<s_" in content and r"/s_" in content: 
            value = token2json(content) # recursive parse
        else:
            value = content
        output[key] = value
        tokens = tokens[match.end(): ].strip()
    return output

# https://www.youtube.com/watch?v=XYi2-LPrwm4&t=439s
def calc_levenshtein_distance(word1: str, word2: str) -> int:
        cache = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        # Pre-fill minimum distance for case when len(word1) == 0
        for j in range(len(word2) + 1):
            cache[-1][j] = len(word2) - j
        # Pre-fill minimum distance for case when len(word2) == 0
        for i in range(len(word1) + 1):
            cache[i][-1] = len(word1) - i
        # Calculate distance
        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2) -1, -1, -1):
                if word1[i] == word2[j]:
                    cache[i][j] = cache[i+1][j+1] 
                else:
                    cache[i][j] = 1 + min(cache[i+1][j], cache[i][j+1], cache[i+1][j+1])
        return cache[0][0]



# def token2json(tokens, is_inner_value=False):
#     output = {}
#     while tokens:
#         start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
#         if start_token is None:
#             break
#         key = re.escape(start_token.group(1)) # key name of start token
#         start_token = start_token.group() 
#         end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
#         if end_token is not None:
#             end_token = end_token.group()
#             start_token_escaped = re.escape(start_token)
#             end_token_escaped = re.escape(end_token)
#             content = re.search(
#                 f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
#             )
#             if content is not None:
#                 content = content.group(1).strip()
#                 if r"<s_" in content and r"/s_" in content: # dict withen a dict
#                     value = token2json(content, True)
#                     if value:
#                         if len(value) == 1:
#                             value = value[0]
#                         output[key] = value
#                 else: # leaf dict
#                     output[key] = content
#                 tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
#         else: # if not closing tag (malformed token)
#             tokens = tokens.replace(start_token, '')
#     if len(output):
#         return [output] if is_inner_value else output
#     else:
#         return []


