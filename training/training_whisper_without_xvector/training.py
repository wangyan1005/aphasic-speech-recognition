# based on https://huggingface.co/blog/fine-tune-whisper
# based on https://huggingface.co/docs/peft/task_guides/lora_based_methods

import argparse
import os
import time
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics
from peft import get_peft_model, LoraConfig, TaskType  # LoRA apply
import glob
from torch.utils.data import DataLoader

# parse command-line arguments
parser = argparse.ArgumentParser(description="Train Whisper-small with LoRA on decoder MLP W1 layer.")
parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (e.g., 4, 8, 16). Default is 8.")
args = parser.parse_args()

# model size: Whisper-small
model_id = "openai/whisper-small"

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load dataset
dataset_path = "/scratch/wang.yan8/dataset_dict_small_without_xvector"
dataset_dict = load_from_disk(dataset_path)

print("Dataset loaded.")

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]

# load base Whisper model
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

# Apply LoRA in decoder's MLP W1 layer
decoder_fc1_modules = []
for name, module in model.named_modules():
    if "decoder" in name and "fc1" in name:
        decoder_fc1_modules.append(name)

lora_config = LoraConfig(
    r=args.lora_rank,  # LoRA (4, 8, 16)
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=decoder_fc1_modules,   
)

model = get_peft_model(model, lora_config) # Apply LoRA
model.enable_input_require_grads()
print(f"Applied LoRA to decoder MLP W1 layer with rank {args.lora_rank}.")

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


# define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"/scratch/wang.yan8/trained_models/whisper-small-LORA-w1-{args.lora_rank}",
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
for name, p in model.named_parameters():
    if p.requires_grad:
        print(" LoRA param:", name)

print(f"Starting LoRA fine-tuning on decoder MLP W1 (rank={args.lora_rank})...")
start_time = time.time()
if checkpoint:
    print(f"Resuming from checkpoint: {checkpoint}")

    # Remove old optimizer and scheduler state so they won't be loaded
    for pattern in ["optimizer*.pt", "scheduler*.pt"]:
        for path in glob.glob(os.path.join(checkpoint, pattern)):
            try:
                os.remove(path)
                print(f"Deleted {path}")
            except OSError:
                print(f"Failed to delete {path}")

else:
    print("No checkpoint found. Starting fresh training.")

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

