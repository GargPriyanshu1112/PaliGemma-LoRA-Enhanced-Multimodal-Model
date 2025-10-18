import torch
from PIL import Image
from transformers import AutoProcessor
from generation_utils import generate
from dataset_utils import token2json
from utils import get_torch_device, load_model


if __name__ == '__main__':
    model_path = r"/home/finetuned_paligemma"

    prompt_l = ["<image> extract JSON."]
    image_path_l = [r"/home/test.jpg"] 
   
    device = get_torch_device()

    print("Loading model...")
    model = load_model(model_path, device)
    model.eval()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

    model_inputs = processor(
        [Image.open(path) for path in image_path_l],
        prompt_l, 
        return_tensors="pt"
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    print('Generating output...')
    with torch.no_grad():
        outputs = generate(
            model,
            processor,
            model_inputs['pixel_values'],
            model_inputs['input_ids'],
            model_inputs['attention_mask'],
            1024,
            device,
            0.8,
            0.9,
            False
        )
    print(token2json(outputs[0], processor))
