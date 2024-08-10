#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Llama 3.1 8B
# 
# Adapted from source: https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=aTaDCGTe78bK
# and
# https://huggingface.co/docs/trl/sft_trainer

# ## 1. Imports

# In[1]:

import os
os.environ['HF_TOKEN']='***REMOVED***'

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported


# ## 2. Load Model

# In[2]:


# Load model
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

# Prepare model for PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

model.print_trainable_parameters()


# ## 3. Prepare data and tokenizer

# In[3]:


tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
)

i=1
def apply_template(examples):
    global i
    messages = examples['conversations']
    for m in messages:
        if i == 1:
            # print(f"messages len is {len(messages)}")
            # print(f"messages is {messages}")
            print(f"message is {m} and type is {type(m)}")
            i -= 1
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

# TODO: Edit this dataset to be "thesherrycode/gen-z-slangs-translation"
# dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = load_dataset("json", data_files={'train': 'gen_z_slangs_translation.json'})
dataset = dataset.map(apply_template, batched=True)


# ## 4. Training

# In[ ]:


trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    # train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    # packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()


# ## 5. Inference

# In[ ]:


# Load model for inference
FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)


# ## 6. Save Model

# In[ ]:


model.save_pretrained_merged("trained_model", tokenizer, save_method="merged_16bit")


# In[ ]:





# In[ ]:




