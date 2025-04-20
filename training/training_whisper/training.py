# based on https://huggingface.co/blog/fine-tune-whisper
# https://huggingface.co/docs/peft/task_guides/lora_based_methods

import argparse
import os
import time
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics
from peft import get_peft_model, LoraConfig, TaskType  # LoRA apply
from personalized_whisper import PersonalizedWhisper
from torch.utils.data import DataLoader

# parse command-line arguments
parser = argparse.ArgumentParser(description="Train Whisper-small with LoRA on decoder MLP W1 layer.")
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (e.g., 4, 8, 16). Default is 8.")
args = parser.parse_args()

# model size: Whisper-small
model_id = "openai/whisper-small"

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load dataset
dataset_path = "/scratch/wang.yan8/dataset_dict_small"
dataset_dict = load_from_disk(dataset_path)

print("Dataset loaded.")

columns_to_keep = ["input_features", "labels", "xvector"]
dataset_dict["train"] = dataset_dict["train"].with_format("torch", columns=columns_to_keep)
dataset_dict["eval"] = dataset_dict["eval"].with_format("torch", columns=columns_to_keep)
dataset_dict["test"] = dataset_dict["test"].with_format("torch", columns=columns_to_keep)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]

for split in ["train", "eval", "test"]:
    dataset = dataset_dict[split]
    for idx in range(min(10, len(dataset))):  
        sample = dataset[idx]
        if 'labels' not in sample:
            print(f"Error: Split {split}, sample {idx} missing 'labels' key.")

sample = dataset_dict["train"][0]
print("Sample keys:", sample.keys())


# load base Whisper model
base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
base_model.to(device)

# Apply LoRA in decoder's MLP W1 layer
decoder_fc1_modules = []
for name, module in base_model.named_modules():
    if "decoder" in name and "fc1" in name:
        decoder_fc1_modules.append(name)

lora_config = LoraConfig(
    r=args.lora_rank,  # LoRA (4, 8, 16)
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=decoder_fc1_modules,
    # if apply LoRA to all MLP W1 layers(encoder + decoder), set target_modules=["fc1"]
)

base_model = get_peft_model(base_model, lora_config)  # Apply LoRA
print(f"Applied LoRA to decoder MLP W1 layer with rank {args.lora_rank}.")

# Wrap base model with PersonalizedWhisper to fuse x-vector with log-Mel features
xvec_dim = 512
projection_dim = 64
mel_dim = 3000
model = PersonalizedWhisper(base_model, xvec_dim, projection_dim, mel_dim)
model.to(device)

model.generation_config.language = "English"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

# initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
print("Data collator finished.")

collated_batch = data_collator([dataset_dict["train"][0], dataset_dict["train"][1]])
print("Collated batch keys:", collated_batch.keys())

# define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"../../trained_models/whisper-small-lora-w1-{args.lora_rank}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=14000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
    remove_unused_columns=False,
    save_safetensors=False,
    
)

# check if a checkpoint exists in the output directory
def get_latest_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
            return os.path.join(output_dir, checkpoints[0])
    return None

checkpoint = get_latest_checkpoint(training_args.output_dir)
if checkpoint:
    print(f"Resuming from checkpoint: {checkpoint}")
else:
    print("No checkpoint found. Starting fresh training.")


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer),
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


print(f"Starting LoRA fine-tuning on decoder MLP W1 (rank={args.lora_rank})...")
start_time = time.time()
trainer.train(resume_from_checkpoint=checkpoint)
end_time = time.time()
session_time = end_time - start_time
      
time_file = "total_training_time.txt"
if os.path.exists(time_file):
    with open(time_file, "r", encoding="utf-8") as f:
        total_time = float(f.read().strip())
else:
    total_time = 0

total_time += session_time

with open(time_file, "w", encoding="utf-8") as f:
    f.write(str(total_time))

print(f"Total training time across sessions: {total_time:.2f} seconds.")
trainer.save_model(training_args.output_dir)

# assessment
print("Evaluating on the test dataset...")
predictions = trainer.predict(test_dataset=test_dataset)
print("Predictions:", predictions)
print(predictions.metrics)

