import torch
import numpy as np
from PIL import Image

def process_images(images, size, resample, rescale_factor, img_mean, img_std, dtype='float32'):
    imgs_arr = np.array([image.resize(size, resample=resample) for image in images], dtype=dtype)
    imgs_arr *= rescale_factor
    imgs_arr = (imgs_arr - np.array(img_mean, dtype=dtype)) / np.array(img_std, dtype=dtype)
    imgs_arr = imgs_arr.transpose(0, 3, 1, 2) # [b, h, w, c] -> [b, c, h, w]
    return torch.tensor(imgs_arr) # .dtype == 'float32'

def build_input_string(img_token, num_img_tokens, bos_token, prefix):
    return f"{img_token * num_img_tokens}{bos_token}{prefix}\n"

class PaliGemmaProcessor:
    IMG_PLACEHOLDER_TOKEN = "<image>"
    OBJ_DET_TOKENS = [f"<loc{idx:04d}>" for idx in range(1024)]
    OBJ_SEG_TOKENS = [f"<seg{idx:03d}>" for idx in range(128) ]
    IMAGENET_MEAN = [0.5, 0.5, 0.5] # one for each channel ; TODO: Sanity check values from paper
    IMAGENET_STD  = [0.5, 0.5, 0.5] # one for each channel ; TODO: Sanity check values from paper
    
    def __init__(self, image_size, num_image_tokens, tokenizer):
        super().__init__()
        self.image_size = image_size # (h, w)
        self.image_seq_length = num_image_tokens
        # PaliGemma uses the Gemma tokenizer, which wasn't originally designed for multimodal input.
        # We need to add special tokens for it to support image-related information.
        NEW_TOKENS = self.OBJ_DET_TOKENS + self.OBJ_SEG_TOKENS
        tokenizer.add_special_tokens({'additional_special_tokens': [self.IMG_PLACEHOLDER_TOKEN]})
        tokenizer.add_tokens(NEW_TOKENS)
        tokenizer.add_bos_token = False # disable automatic addition of BOS token
        tokenizer.add_eos_token = False # disable automatic addition of EOS token
        self.tokenizer = tokenizer
        self.img_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_PLACEHOLDER_TOKEN)

    def __call__(self, images, texts, padding="longest", truncation=True):
        assert len(images) == 1 and len(texts) == 1, "Currently works only on a single image-text pair."
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            img_mean=self.IMAGENET_MEAN,
            img_std=self.IMAGENET_STD,
        )
        input_strings = [
            build_input_string(
                img_token=self.IMG_PLACEHOLDER_TOKEN,
                num_img_tokens=self.image_seq_length,
                bos_token=self.tokenizer.bos_token,
                prefix=text,
            ) 
            for text in texts
        ]
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        ) # returns `input_ids` and `attention_mask`
        return {"pixel_values": pixel_values, **inputs}
