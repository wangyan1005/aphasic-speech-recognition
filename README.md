# Aphasic Speech Recognition

## Overview
This project explores integrating speaker-specific embeddings (x-vectors) with Low-Rank Adaptation (LoRA) to adapt the Whisper-small ASR model, aiming to better recognize and transcribe speech from individuals with aphasia.

## Dataset
- **AphasiaBank Dataset**: A collection of audio recordings and transcriptions from individuals with aphasia, widely used for research in speech and language processing.
- The dataset provides metadata such as patient names, age, sex, and conversation transcriptions.

## Model
- **Whisper-small Model**: A speech recognition model developed by OpenAI.

## Environment Setup
Set up the environment by loading modules and installing necessary packages:
 ```bash
   module load python/3.8.1
   pip install transformers datasets torchaudio peft jiwer pandas numpy
 ```

## Data Processing
Data preprocessing (Step 1 to Step 5) uses scripts from https://github.com/Liting-Zhou/Aphasic_speech_recognition. My approach extends the original pipeline by integrating speaker-specific embeddings (x-vectors).
### Step6: extract xvector from audio chunks
 ```bash
   python extract_xvector_for_chunk.py
 ```
### Step7: Split the dataset into training (60%), validation (20%), and test (20%) sets
```bash
   data_splitting.py.py
 ```

## Baseline Model Transcribe
Evaluate the baseline Whisper-small model without speaker adaptation:
- **transcribe.py**: The script transcribes the test set audio using Whisper-small model from Hugging Face's pipeline, and writes the detailed transcription results (including predictions and references) to a CSV file.
- **wer_calculation.py**: This script reads the detailed transcription results and calculates the overall Word Error Rate (WER).
```bash
   transcribe.py
   wer_calculation.py
 ```

## Training Models 
Training scripts are organized in two directories:
- **With x-vectors (LoRA + x-vector integration): change directory:**
```bash
   cd training_whisper/
 ```
- **Without x-vectors (LoRA-only): change directory:**
```bash
   cd training_whisper_without_xvector/
 ```

### Step1:  Data Preparation
- **data_preparation.py**:Prepares and processes the dataset by reading a CSV file, loading audio files, extracting log-Mel features, and parsing x-vectors(if x-vector is invloved).
```bash
   python data_preparation.py
 ```
### Step2: Data Collation
- **data_collator.py**: Provides a data collator that pads input features and labels for speech recognition, and also processes x-vectors(if x-vector is invloved).
  
### Step3: Personalized Whisper Model(only for x-vector integration)
- **personalized_whisper.py**: Implements the personalized Whisper model. This module integrates speaker-specific x-vector information with the log-Mel spectrogram features.
  
### Step4: Model Training 
- **training.py**: The main training script: fine-tunes Whisper-small using LoRA on specific layers (default is decoder's MLP W1 layers).
   python training.py
 ```
Key parameters used in experiments:
- **LoRA rank (r)**: 8, 64, 128, 256
- **LoRA alpha scaling factor (α)**: 32
- **Training steps**: 14,000
- **Evaluation intervals**: Every 1,000 steps
- **Learning rate**: 5e-6
- **Batch size**: 8 (with gradient accumulation steps = 4, effective batch size = 32)

### Step5: Evlaution
- **compute_metrics.py**: Defines functions to compute the Word Error Rate (WER).

## Expected Outputs and Results
Upon completion, results include:
- Fine-tuned model checkpoints (trained_models/)
- WER metrics

