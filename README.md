# Aphasic Speech Recognition

## Overview
This project focuses on aphasic speech recognition using the AphasiaBank dataset and the Whisper model. The goal is to preprocess and fine-tune speech recognition models to better understand and transcribe speech from individuals with aphasia.

## Dataset
- **AphasiaBank Dataset**: A collection of audio recordings and transcriptions from individuals with aphasia, widely used for research in speech and language processing.
- The dataset provides metadata such as patient names, age, sex, and conversation transcriptions.

## Model
- **Whisper Model**: A state-of-the-art speech recognition model developed by OpenAI.
- The model is fine-tuned using LoRA (Low-Rank Adaptation) to efficiently adapt to aphasic speech patterns.

## Environment Setup
 ```bash
   module load python/3.8.1
   pip install transformers datasets torchaudio peft jiwer pandas numpy
 ```

## Data Processing
Data preprocessing (Step 1 to Step 5) is performed using the same scripts provided by the [Aphasic Speech Recognition](https://github.com/Liting-Zhou/Aphasic_speech_recognition). My approach extends the original pipeline by integrating speaker-specific embeddings (x-vectors).
### Step6: extract xvector from audio chunks
 ```bash
   python extract_xvector_for_chunk.py
 ```
### Step7: Split the dataset into training (60%), validation (20%), and test (20%) sets
```bash
   data_splitting.py.py
 ```

## Baseline Model Transcribe
- **transcribe.py**: The script transcribes the test set audio using Whisper model from Hugging Face's pipeline, and writes the detailed transcription results (including predictions and references) to a CSV file.
- **wer_calculation.py**: This script reads the detailed transcription results and calculates the overall Word Error Rate (WER).
```bash
   transcribe.py
   wer_calculation.py
 ```

## Training Models 
For training with LoRA and x-vectors, change directory:
```bash
   cd training_whisper/
 ```
For training without x-vectors, change directory:
```bash
   cd training_whisper_without_xvectors/
 ```

### Step1:  Data Preparation
- **data_preparation.py**:Prepares and processes the dataset by reading a CSV file, loading audio files, extracting log-Mel features, and parsing x-vectors(if x-vector is invloved).
```bash
   python data_preparation.py
 ```
### Step2: Data Collation
- **data_collator.py**: Provides a data collator that pads input features and labels for speech recognition, and also processes x-vectors(if x-vector is invloved).
  
### Step3: Personalized Whisper Model(only for integration with x-vectors)
- **personalized_whisper.py**: Implements the personalized Whisper model. This module integrates speaker-specific x-vector information with the log-Mel spectrogram features.
### Step4: Training
- **training.py**: The main training script. It loads the prepared dataset, applies LoRA to the Whisper-Small.
```bash
   python training.py
 ```
### Step5: Evlaution
- **compute_metrics.py**: Defines functions to compute the Word Error Rate (WER).


