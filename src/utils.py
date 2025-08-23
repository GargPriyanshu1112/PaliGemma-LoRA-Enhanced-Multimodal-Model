import os
import json
import torch
from glob import glob
from safetensors import safe_open
from transformers import AutoTokenizer, PaliGemmaConfig
from modelling_paligemma import PaliGemmaForConditionalGeneration

def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, device=None, framework='pt'):
    if not device:
        device = get_torch_device()
    
    safetensors_file_l = glob(os.path.join(model_path, '*.safetensors'))
    tensors = {}
    for safetensors_file in safetensors_file_l:
        with safe_open(safetensors_file, framework=framework, device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        print(f"Successfully loaded {os.path.basename(safetensors_file)}")
    
    config = PaliGemmaConfig.from_pretrained(os.path.join(model_path, 'config.json'))
    config.vision_config.hidden_size = 1152
    config.vision_config.embed_dim = config.vision_config.hidden_size # for this specific custom siglip impl only 
    # TODO
    # with open(os.path.join(model_path, 'config.json'), 'r') as f:
    #     config_file = json.load(f)
    #     # config_file['vision_config']['embed_dim'] = config_file['vision_config']['hidden_size'] # for this specific custom siglip impl only 
    #     config = PaliGemmaConfig(**config_file)
    
    model = PaliGemmaForConditionalGeneration(config)
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()
    model.to(device)
    return model

def load_tokenizer(pretrained_model_name_or_path):
    tokenizer =  AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, padding_side='left'
    )
    return tokenizer

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(e, device) for e in obj)
    return obj

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