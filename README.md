# PaliGemma+ : LoRA-Enhanced Multimodal VLM

• Built a **trainable PaliGemma from scratch**, integrating **SigLIP** (vision encoder), **Gemma** (language model), multimodal fusion modules and **LoRA** for parameter-efficient fine-tuning.

• Enhanced efficiency with **Rotary Positional Embeddings (RoPE)** for richer context representation and **KV caching** for faster inference.

<h3>Full Fine-Tuning Performance (10 epochs)</h3>
<p align="center">
  <img src="assets/finetune_results.jpg" width="700">
</p>


<h3>LoRA Fine-Tuning Performance (10 epochs)</h3>
<p align="center">
  <img src="assets/finetuning_results_lora.jpg" width="700">
</p>

### Sample Model Output

<div align="center">
<table>
<tr>
  <th align="center">Input Receipt</th>
  <th align="center">With LoRA (raw JSON)</th>
  <th align="center">Without LoRA (raw JSON)</th>
</tr>

<tr>
<td align="center" valign="top">
  <img src="assets/test_img1.jpg" width="250"/>
</td>
<td valign="top"><pre>
{
  "total": {
    "total_price": "50,000",
    "creditcardprice": "50,000"
  },
  "sub_total": {
    "tax_price": "4,545",
    "subtotal_price": "45,455",
    "etc": "0"
  },
  "menu": {
    "nm": "[REG] BLACK SAKURA",
    "price": "45,455",
    "cnt": "1",
    "sub": [
      {
        "nm": "COOKIE DON SAUCES",
        "price": "0",
        "cnt": "1"
      },
      {
        "nm": "NATA DE COCO",
        "price": "0",
        "cnt": "1"
      }
    ]
  }
}
</pre></td>
<td valign="top"><pre>
{
  "total": {
    "total_price": "50,000",
    "creditcardprice": "50,000"
  },
  "sub_total": {
    "tax_price": "4,545",
    "subtotal_price": "45,455",
    "etc": "0"
  },
  "menu": {
    "nm": "[REG] BLACK SAKURA",
    "price": "45,455",
    "cnt": "1",
    "sub": [
      {
        "nm": "COOKIE DON SAUCES",
        "price": "0",
        "cnt": "1"
      },
      {
        "nm": "NATA DE COCO",
        "price": "0",
        "cnt": "1"
      }
    ]
  }
}
</pre></td>
</tr>

<tr>
<td align="center" valign="top">
  <img src="assets/test_img2.jpg" width="250"/>
</td>
<td valign="middle"><pre>
{
  "total": {
    "total_price": "54,000",
    "changeprice": "50,000",
    "cashprice": "104,000"
  },
  "menu": [
    {
      "nm": "TWIST DONUT",
      "price": "18,000",
      "cnt": "1"
    },
    {
      "nm": "BANANA DONUT",
      "price": "11,000",
      "cnt": "1"
    },
    {
      "nm": "CREAMCHEESE BREAD",
      "price": "13,000",
      "cnt": "1"
    },
    {
      "nm": "FRANKFRUT SAUSAGE ROLL",
      "price": "12,000",
      "cnt": "1"
    }
  ]
}
</pre></td>
<td valign="middle"><pre>
{
  "total": {
    "total_price": "54,000",
    "changeprice": "50,000",
    "cashprice": "104,000"
  },
  "menu": [
    {
      "nm": "TWIST DONUT",
      "price": "18,000",
      "cnt": "1"
    },
    {
      "nm": "BANANA DONUT",
      "price": "11,000",
      "cnt": "1"
    },
    {
      "nm": "CREAMCHEESE BREAD",
      "price": "13,000",
      "cnt": "1"
    },
    {
      "nm": "FRANKFRUT SAUSAGE ROLL",
      "price": "12,000",
      "cnt": "1"
    }
  ]
}
</pre></td>
</tr>

</table>
</div>