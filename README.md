# PaliGemma+ : LoRA-Enhanced Multimodal VLM

• Built a **trainable PaliGemma from scratch**, integrating **SigLIP** (vision encoder), **Gemma** (language model), multimodal fusion modules and **LoRA** for parameter-efficient fine-tuning.

• Enhanced efficiency with **Rotary Positional Embeddings (RoPE)** for richer context representation and **KV caching** for faster inference.

<h4>Full Fine-Tuning Performance (10 epochs)</h4>
<p align="center">
  <img src="assets/finetune_results.jpg" width="700">
</p>

<h4>LoRA Fine-Tuning Performance (10 epochs)</h4>
<p align="center">
  <img src="assets/finetuning_results_lora.jpg" width="700">
</p>