import re
import os
import time
import torch
import wandb
import random
from functools import partial
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from huggingface_hub import create_repo
from nltk.metrics.distance import edit_distance
from transformers import AutoProcessor

from utils import get_torch_device, load_model
from dataset_utils import HFDatasetWrapper, train_collate_fn
from generation_utils import generate
from lora import attach_lora_weights, merge_and_unwrap_lora

load_dotenv()
_hf_token = os.getenv("HF_TOKEN")
assert _hf_token is not None, "Load HF_TOKEN in env."

_wandb_token = os.getenv("WANDB_API_KEY")
assert _wandb_token is not None, "Load WANDB_API_KEY in env."

os.environ["WANDB_API_KEY"] = _wandb_token
wandb.login()

# ---------------base dir----------------
HOME = r"/home"
# ---------------------------------------

# ------------- hyperparams -------------
NUM_EPOCHS = 10
IMAGE_SIZE = 224
NUM_IMG_TOKENS = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0
NUM_PROC = 4
NUM_WORKERS = 4
GRAD_ACCUMULATION_STEPS = 4
ENABLE_AUTOCAST = False
MAX_LENGTH = 1024
USE_LORA = False
# ---------------------------------------

# --------------hf hub setup-------------
HF_DATASET_ID = "naver-clova-ix/cord-v2"
PALIGEMMA_PROCESSOR_ID = "google/paligemma-3b-pt-224"
FINETUNED_MODEL_ID = "PriyHF/paligemma-3b-pt-224-finetuned"
create_repo(FINETUNED_MODEL_ID, exist_ok=True, token=_hf_token)
# ---------------------------------------

# --------------ckpt setup-------------
output_dir = rf"{HOME}/finetuned_paligemma"  # local folder where checkpoints go
os.makedirs(output_dir, exist_ok=True)
# ---------------------------------------

# --------------input prompt-------------
PROMPT = "extract JSON."
# ---------------------------------------


if __name__ == '__main__':  
    model_path = rf"{HOME}/paligemma-3b-pt-224-model-files"

    device = get_torch_device()
    print(f"Device: {device}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(PALIGEMMA_PROCESSOR_ID)
    
    print("Loading model...")
    model = load_model(model_path, device, 'pt')
    if USE_LORA:
        # Freeze all params
        for param in model.parameters():
            param.requires_grad = False
        # Attach lora params
        attach_lora_weights(
            model, 
            ["q_proj", "o_proj", "out_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            r=8
        )
        # Ensure all weights (base + LoRA) are on same device
        model.to(device)
    else:
        for name, param in model.named_parameters():
            if "attn" in name or "norm" in name or "bias" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    print("Loading datasets...")
    train_dataset = HFDatasetWrapper(HF_DATASET_ID, 'train', num_proc=NUM_PROC)
    val_dataset = HFDatasetWrapper(HF_DATASET_ID, 'validation', num_proc=NUM_PROC)
    print(f"Datasets loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")
    
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset, 
        BATCH_SIZE, 
        True, 
        collate_fn=partial(train_collate_fn, processor=processor, prompt=PROMPT),
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset, 
        BATCH_SIZE, 
        False, 
        collate_fn=partial(train_collate_fn, processor=processor, prompt=PROMPT),
        num_workers=NUM_WORKERS
    )
    print(f"Dataloaders ready: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    # -----------------initialize wandb-----------------
    wandb.init(
        project="paligemma-finetune",
        name="paligemma-3b-pt-224",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE,
            "grad_accum_steps": GRAD_ACCUMULATION_STEPS
        }
    )
    # ----------------------------------------------------
    
    best_val_loss = float("inf")
    print("Training...")
    total_start = time.time()  # start total training timer
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()  # start epoch timer
        model.train()
        train_loss_epoch = 0
        for idx, train_batch in enumerate(train_dataloader):
            train_batch = {k: v.to(device) for k, v in train_batch.items()} 
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=ENABLE_AUTOCAST):
                train_outputs = model(
                    input_ids=train_batch["input_ids"], 
                    pixel_values=train_batch["pixel_values"], 
                    attention_mask=train_batch["attention_mask"], 
                    labels=train_batch["labels"], 
                    token_type_ids=train_batch["token_type_ids"], 
                    kv_cache=None
                )
                train_loss_batch = train_outputs['loss']
            unscaled_loss = train_loss_batch.item()
            scaled_loss = train_loss_batch / GRAD_ACCUMULATION_STEPS

            if idx % 10 == 0:
                print(f"Epoch: {epoch+1} Iter: {idx} Train Loss: {unscaled_loss:.4f}")
            train_loss_epoch += unscaled_loss
            
            scaled_loss.backward()

            if (idx + 1) % GRAD_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}           
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=ENABLE_AUTOCAST):
                    val_outputs = model(
                        input_ids=val_batch["input_ids"],
                        pixel_values=val_batch["pixel_values"],
                        attention_mask=val_batch["attention_mask"],
                        labels=val_batch["labels"],
                        token_type_ids=val_batch["token_type_ids"],
                        kv_cache=None
                    )
                    val_loss_batch = val_outputs['loss']
                val_loss_epoch += val_loss_batch.item()
                
        # Calculate average losses
        avg_train_loss = train_loss_epoch / len(train_dataloader)
        avg_val_loss = val_loss_epoch / len(val_dataloader)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Run generation on one random validation batch
        val_iter = iter(val_dataloader)
        rand_idx = random.randint(1, len(val_dataloader)-1)
        for _ in range(rand_idx):
            batch = next(val_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        answers = batch["labels"]

        predictions = generate(
            model=model,
            processor=processor,
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_tokens_to_generate=MAX_LENGTH,
            device=device,
        )

        # Compute normalized edit distance
        scores = []
        for pred, answer in zip(predictions, answers):
            if torch.is_tensor(answer):
                answer_ids = answer.tolist()
                # keep only valid IDs
                answer_ids = [tid for tid in answer_ids if 0 <= tid < processor.tokenizer.vocab_size]
                answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            denom = max(len(pred), len(answer), 1)
            scores.append(edit_distance(pred, answer) / denom)
            print(f"    Answer: {answer}")
            print(f"Prediction: {pred}")

        avg_edit_distance = sum(scores) / len(scores)
        print(f"[Validation Generation] Avg NormED (random batch): {avg_edit_distance:.4f}")

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | "
                  f"Avg Val Loss: {avg_val_loss:.4f} | "
                  f"Learning Rate: {param_group['lr']:.4f} | "
                  f"Epoch Time: {epoch_time/60:.2f} min\n")
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time_min": round(epoch_time / 60, 2)
        })

        # Save best model locally based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model found with validation loss: {best_val_loss:.4f}")
            print(f"Saving model to {output_dir}")

            if USE_LORA:
                adapter_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k
                }
                torch.save(adapter_state, os.path.join(output_dir, "lora_adapter.pt"))

            # Workaround to prevent weight tying error (https://github.com/kazuar/Phi3-Vision-ft/issues/2)
            state_dict = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if not k.endswith("lm_head.weight")}
            # Save model and processor
            model.save_pretrained(output_dir, state_dict=filtered)
            processor.save_pretrained(output_dir) 
    
    total_end = time.time()
    total_time = total_end - total_start
    print(f"\nTotal training time: {total_time/60:.2f} minutes.")
    
    wandb.finish()

    # TODO: Hack; needs refactoring
    if USE_LORA:
        print("\nMerging LoRA and saving final model...")
        model = merge_and_unwrap_lora(model)
        model.to("cpu")
        # Workaround to prevent weight tying error (https://github.com/kazuar/Phi3-Vision-ft/issues/2)
        state_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if not k.endswith("lm_head.weight")}
        # Save model and processor
        model.save_pretrained(output_dir, state_dict=filtered)
        processor.save_pretrained(output_dir) 

    print("\nTraining finished. Pushing the best model to the Hugging Face Hub.")
    api = HfApi()
    api.upload_folder(
        repo_id=FINETUNED_MODEL_ID, folder_path=output_dir, token=_hf_token
    )
    print(f"Best model successfully pushed to {FINETUNED_MODEL_ID}")