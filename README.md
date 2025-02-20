# Aphasic Speech Recognition

## Overview
This project focuses on aphasic speech recognition using the AphasiaBank dataset and the Whisper model. The goal is to preprocess and fine-tune speech recognition models to better understand and transcribe speech from individuals with aphasia.

## Dataset
- **AphasiaBank Dataset**: A collection of audio recordings and transcriptions from individuals with aphasia, widely used for research in speech and language processing.
- The dataset provides metadata such as patient names, age, sex, and conversation transcriptions.

## Model
- **Whisper Model**: A state-of-the-art speech recognition model developed by OpenAI.
- The model is fine-tuned using LoRA (Low-Rank Adaptation) to efficiently adapt to aphasic speech patterns.

## Data Processing
The data preprocessing is based on https://github.com/Liting-Zhou/Aphasic_speech_recognition.
I have added two new files:
- **test.py**:The provided scripts test the dataset and remove non-standard audio segments.
- **hours_analysis.py**: The scripts extract and analyze the total speech duration for each speaker.
  
