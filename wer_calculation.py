# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/wer_calculation.py
import pandas as pd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation
import inflect

def convert_numbers_to_words(text):
    """
    Convert digit numbers into words (e.g., 2002 -> two thousand and two).
    """
    p = inflect.engine()
    words = text.split()
    
    converted_text = []
    
    for word in words:
        if word.isdigit():  
            try:
                word = p.number_to_words(word, andword='', group=2)
            except (inflect.NumOutOfRangeError, ValueError):
                pass
        converted_text.append(word)
    
    return " ".join(converted_text)


def calculate_overall_wer_from_csv(detailed_csv):
    """
    Calculate the overall WER from the detailed CSV file after all predictions are made.
    """
    df = pd.read_csv(detailed_csv)
    references = df['reference'].tolist()
    predictions = df['prediction'].tolist()

    # Define text transformation
    transformation = Compose([ToLowerCase(), RemovePunctuation()])

    # Normalize the prediction and transcription text
    references_normalized = [transformation(ref) for ref in references]
    predictions_normalized = [transformation(pred) for pred in predictions]

    # Convert numbers in the predicted text to words
    predictions_normalized = [convert_numbers_to_words(pred) for pred in predictions_normalized]

    # Calculate overall WER using the list of references and predictions
    overall_wer = wer(references_normalized, predictions_normalized)
    return overall_wer

def run_all_wer_calculations(models, detailed_results_folder, summary_csv):
    """
    Run WER calculations for models and save the summary results.
    """
    summary_data = []

    for model_name in models:
        file_name = model_name.replace("/", "_")
        detailed_csv = f"{detailed_results_folder}/detailed_{file_name}_results.csv"
        
        overall_wer = calculate_overall_wer_from_csv(detailed_csv)
        print(f"Overall WER for {model_name}: {overall_wer}")

        summary_data.append({"model": model_name, "wer": overall_wer})

    # Save summary WER results
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary WER results saved to {summary_csv}")


detailed_csv = "../data_processed/detailed_wer_results/detailed_openai_whisper-medium_results.csv"
print(calculate_overall_wer_from_csv(detailed_csv))


