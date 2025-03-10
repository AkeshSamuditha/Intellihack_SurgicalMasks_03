from transformers import AutoTokenizer

model_name = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer=AutoTokenizer.from_pretrained(model_name)

from datasets import load_dataset
data = load_dataset('squad')
print(data)

def preprocess(examples):
    inputs = tokenizer(
      examples["question"],
      examples["context"],
      truncation=True,
      padding="max_length",
      return_offsets_mapping=True,
      max_length = 512,
      stride = 128
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        context_start = idx
        context_end = idx
        try:
            while sequence_ids[idx] != 1:
                idx += 1
                context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
                context_end = idx - 1
        except:
            pass

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


gpt2_squad = data.map(preprocess, batched=True, remove_columns=data["train"].column_names)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="gpt2_qa", 
    logging_steps=1, 
    report_to="none", 
    per_device_train_batch_size=2,            
    gradient_accumulation_steps=2,        
    per_device_eval_batch_size=2,
    )


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True, 
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, 
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], 
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=gpt2_squad["train"],
    eval_dataset=gpt2_squad["validation"],
    data_collator=data_collator,
    processing_class=tokenizer
)

trainer.train()