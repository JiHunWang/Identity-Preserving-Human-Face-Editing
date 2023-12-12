# Identity-Preserving Human Face Editing


## Project Description

We explore the task of **human face editing** using multimodal diffusion models, which is a challenging task that previous models do not excel at. In particular, prior works suffer from the **identity preservation of the subject**, only focusing on editing images as prescribed by the textual prompts. To address this issue, they often rely on an extensive fine-tuning of the model, hence affected by computational constraints. 

In light of this, we aim to close this gap and investigate various ways to further improve the performance of diffusion models as multimodal human face editing tools in a resource-aware setting. Specifically, we focus on the possibility of parameter-efficient fine-tuning (PEFT), which has been widely adopted for an efficient adaptation of pre-trained language models without extensive fine-tuning, often yielding a comparable performance to full fine-tuning. We extend PEFT in text-based conditional image generation tasks by experimenting with instruction tuning approaches that lead to better performance under the injection of more subtle semantic prompts, mixed precisions, and memory-efficient attention using instruction-tuning motivated by Alpaca and FLAN V2.



## Directory Structure

This repo consists of three subdirectories:
1. The first is baseline `instruct-pix2pix` (without fine-tuning), which is the model that we utilize for fine-tuning.
2. The second is `instruction-tuned-sd`, which provides a motivation and basic bash prompts that we modify to fine-tune InstructPix in a parameter-efficient way.
3. The third is `Collaborative-Diffusion`, which we originally explored as a (better) baseline model, but ended up not utilizing (**deprecated**).




## Code Snippets

1. Run `pip install -r requirements.txt` to install all the dependencies. (**Note**: `xformers` and `accelerate` by default installs PyTorch 2.1.0, which may have some issues with activating the CUDA and training on GPUs. Depending on the driver installed, we recommend installing a different version of `torch` and `torchvision`.)
2. For the dataset preparation, refer to `instruction-tuned-sd/data_preparation` for further details. This requires a HuggingFace developer ID.
3. Launch the following `accelerate` commands:

```bash
export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="instruction-tuning-sd/CUSTOM_DATASET"
export OUTPUT_DIR="face-image-editing-finetuned"
export VALIDATION_PROMPT="PROMPT OF ONE'S CHOICE"

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --resolution=256 --random_flip \
  --dataset_name=$DATASET_ID \
  --train_batch_size=2 \
  --max_train_steps=15000 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --validation_prompt=$VALIDATION_PROMPT \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision=fp16 \
```

The last three lines are relevant to the parameters that one can set in the setting of PEFT, which can be further tuned based on needs.



