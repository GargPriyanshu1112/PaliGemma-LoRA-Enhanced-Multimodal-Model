import os
import time
import torch
import wandb
from functools import partial
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from huggingface_hub import create_repo

from utils import get_torch_device, load_model, load_tokenizer, move_to_device
from dataset_utils import HFDatasetWrapper, train_collate_fn
from processing_paligemma import PaliGemmaProcessor
from lora import attach_lora_to_layers

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
NUM_EPOCHS = 5
IMAGE_SIZE = 224
NUM_IMG_TOKENS = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0
NUM_PROC = 4
NUM_WORKERS = 4
GRAD_ACCUMULATION_STEPS = 4
USE_LORA = False
# ---------------------------------------

# --------------hf hub setup-------------
HF_DATASET_ID = "naver-clova-ix/cord-v2"
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
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)

    print("Loading model...")
    model = load_model(model_path, device, 'pt')
    if USE_LORA:
        # Freeze all params
        for param in model.parameters():
            param.requires_grad = False
        # Attach lora params
        attach_lora_to_layers(
            model, 
            ["q_proj", "o_proj", "out_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            r=8
        )
        # Ensure all weights (base + LoRA) are on same device
        model.to(device)
    else:
        # Freezing everything but the attention layers, layernorm and bias params
        for name, param in model.named_parameters():
            if "attn" in name or "norm" in name or "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Loading processor...")
    processor = PaliGemmaProcessor(IMAGE_SIZE, NUM_IMG_TOKENS, tokenizer)
    
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
            train_batch = move_to_device(train_batch, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
                val_batch = move_to_device(val_batch, device)          
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
            # Workaround to prevent weight tying error (https://github.com/kazuar/Phi3-Vision-ft/issues/2)
            state_dict = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if not k.endswith("lm_head.weight")}
            model.save_pretrained(output_dir, state_dict=filtered)
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            # Save processor
            processor.save_pretrained(output_dir)
    
    total_end = time.time()
    total_time = total_end - total_start
    print(f"\nTotal training time: {total_time/60:.2f} minutes.")
    
    artifact = wandb.Artifact("paligemma_best_model", type="model")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)
    wandb.finish()

    print("\nTraining finished. Pushing the best model to the Hugging Face Hub.")
    api = HfApi()
    api.upload_folder(
        repo_id=FINETUNED_MODEL_ID, folder_path=output_dir, token=_hf_token
    )
    print(f"Best model successfully pushed to {FINETUNED_MODEL_ID}")