# based on https://huggingface.co/blog/fine-tune-whisper

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
dataset_path = "../../data_processed/dataset_dict_small"
dataset_dict = load_from_disk(dataset_path)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]
print("Dataset loaded.")

# load base Whisper model
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

# Wrap base model with PersonalizedWhisper to fuse x-vector with log-Mel features
xvec_dim = 512
projection_dim = 64
mel_dim = 80
model = PersonalizedWhisper(base_model, xvec_dim, projection_dim, mel_dim)
model.to(device)

# Apply LoRA in decoder's MLP W1 layer
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  
    r=args.lora_rank,  # LoRA (4, 8, 16)
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=["fc1"],   # apply only to decoder MLP W1 layer
)

model = get_peft_model(model, lora_config)
print(f"Applied LoRA to decoder MLP W1 layer with rank {args.lora_rank}.")


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
    output_dir=f"../../trained_models/whisper-small-lora-w1-r{args.lora_rank}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    warmup_steps=500,
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
    save_total_limit=5,
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


print(f"Starting LoRA fine-tuning on decoder MLP W1 (rank={args.lora_rank})...")
start_time = time.time()
trainer.train(resume_from_checkpoint=checkpoint)
end_time = time.time()
training_duration = end_time - start_time
print(f"Training completed in {training_duration // 3600} hours, "
      f"{(training_duration % 3600) // 60} minutes, and {training_duration % 60:.2f} seconds.")

trainer.save_model(training_args.output_dir)

# assessment
print("Evaluating on the test dataset...")
predictions = trainer.predict(test_dataset=test_dataset)
print(predictions.metrics)
