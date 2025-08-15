import re
import json
import random
from datasets import load_dataset
from torch.utils.data import Dataset

def json2token(obj: dict | str):
    if isinstance(obj, dict):
        output = ''
        for k, v in obj.items():
            output += (fr"<s_{k}>" + json2token(v) + fr"</s_{k}>")
        return output
    elif isinstance(obj, list):
        return r"<sep/>".join([json2token(item) for item in obj])
    else:
        return obj

# TODO: Refactor
def token2json(tokens, is_inner_value=False, added_vocab=None):
    if added_vocab is None:
        added_vocab = {}

    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


class HFDatasetWrapper(Dataset):
    def __init__(self, dataset_name_or_path, split='train'):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split)
    
    def process_ground_truth(self, ground_truth: str):
        gt = json.loads(ground_truth) # dict
        if 'gt_parses' in gt:
            assert isinstance(gt['gt_parses'], list)
            gt_jsons = gt['gt_parses']
        else:
            assert 'gt_parse' in gt and isinstance(gt['gt_parse'], dict)
            gt_jsons = [gt['gt_parse']]
        return [json2token(gt_json) for gt_json in gt_jsons]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        labels = self.process_ground_truth(sample['ground_truth'])
        label = random.choice(labels) # select at random if more than one
        return image, label


def train_collate_fn(samples, processor, prompt, padding='longest', max_length=None, truncation=False):
    images  = [sample[0].convert('RGB') for sample in samples] 
    prefix = [prompt for _ in samples]
    suffix = [sample[1] for sample in samples]
     
    inputs = processor(
        images=images, 
        texts=prefix, 
        suffix=suffix,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
    )
    return inputs

def eval_collate_fn(samples, processor, prompt, padding='longest', max_length=None, truncation=False):
    images = [sample[0].convert('RGB') for sample in samples] 
    prefix = [prompt for _ in samples]
    outputs = [sample[1] for sample in samples]
     
    inputs = processor(
        images=images, 
        texts=prefix,
        padding=padding,
        max_length=max_length,
        truncation=truncation,
    )
    return inputs, outputs