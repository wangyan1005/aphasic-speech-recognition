# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/data_cleaning_and_statistics.py
import pandas as pd
import os
import soundfile as sf

def get_audio_length(file_path):
    try:
        with sf.SoundFile(file_path) as audio_file:
            return len(audio_file) / audio_file.samplerate*1000  # length in miliseconds
    except Exception as e:
        print(f"Error reading audio file {file_path}: {e}")
        return None

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # get total number of rows before cleaning up
    num_rows = df.shape[0]
    print(f"Total rows: {num_rows}")

    df['difference'] = df.iloc[:, 1] - df.iloc[:, 0]
    # get the total duration
    sum_duration = df['difference'].sum()/1000/60/60 
    print(f"The total duration of all audios: {sum_duration} hours")

    #----------------------------------------------------------------
    # newly added, filter missing or corrupted audios
    
    # remove date with missing audios in the original dataset: UMD/MMA20a, UNH/UNH10b, UNH/UNH11b
    df = df[~df['file'].isin(['MMA20a.wav', 'UNH10b.wav', 'UNH11b.wav'])]

    # extract unique folder_name/file combinations
    unique_files = df[['folder_name', 'file']].drop_duplicates()
    
    # create a dictionary to store the original file lengths
    file_lengths = {}
    
    # calculate the length of each unique audio file
    for index, row in unique_files.iterrows():
        file_path = os.path.join("../data_processed/audios", row['folder_name'], row['file'])
        length = get_audio_length(file_path)
        file_key = f"{row['folder_name']}/{row['file']}"
        file_lengths[file_key] = length

    # map the lengths back to the original DataFrame
    df['original_file_length'] = df.apply(lambda row: file_lengths.get(f"{row['folder_name']}/{row['file']}", None), axis=1)

    # get rows with "mark_start" larger or equal than "file_length"
    invalid_rows = df[df['mark_start'] >= df['original_file_length']]
    if not invalid_rows.empty:
        print(f"Rows with mark_start greater than or equal to file length:\n{invalid_rows[['folder_name', 'file', 'mark_start', 'original_file_length']]}")
        df = df.drop(invalid_rows.index)

    #----------------------------------------------------------------

    # get the rows with differences larger than 30,000, or smaller than 300, or empty transcripts
    large_differences = df[df['difference'] > 30000]
    small_differences = df[df['difference'] < 300]
    empty_transcripts = df[df['transcriptions'].isnull() | (df['transcriptions'].str.strip() == '')]
    # unkown_speaker_names = df[(df['name']=="Unknown") | (df['name']=="Participant")]

    # print number of rows with audios longer than 30 seconds
    if not large_differences.empty:
        num_large_differences = large_differences.shape[0] 
        print(f"Number of rows with differences larger than 30,000: {num_large_differences}") 
    else:
        print("No differences larger than 30,000 found.")
    
    # print number of rows with audios shorter than 0.3 seconds
    if not small_differences.empty:
        num_small_differences = small_differences.shape[0]
        print(f"Number of rows with differences smaller than 300: {num_small_differences}") 
    else:
        print("No differences smaller than 300 found.")
    
    # print number of rows with empty transcripts
    if not empty_transcripts.empty:
        num_empty_transcripts = empty_transcripts.shape[0]
        print(f"Number of rows with empty_transcripts: {num_empty_transcripts}")
    else:
        print("No empty transcripts found.")


    rows_to_drop = pd.concat([large_differences, small_differences, empty_transcripts]).drop_duplicates()
    if not rows_to_drop.empty:
        num_rows_dropped = rows_to_drop.shape[0]
        print(f"Dropping {num_rows_dropped} rows.") 
        df = df.drop(rows_to_drop.index)
    else:
        print("No rows to drop based on the given criteria.")

    # extract the speaker name from the filename
    df['name_extracted_from_filename'] = df['file'].str.split('.wav', expand=True)[0]
    # 1. change the last letter to "a" if it's not already "a", for example, 'kurland01a' and 'kurland01b' belong to the same speaker
    # 2. if the last character is a digit, change the last digit to "1" if it's not already "1",for example, '68-1' and '68-2' belong to the same speaker
    def modify_name(name):
        if name:
            if name[-1].isdigit():
                return name[:-1] + '1'  # change last digit to '1'
            elif name[-1] != 'a':
                return name[:-1] + 'a'  # change last character to 'a'
        return name

    df['name_unique_speaker'] = df['name_extracted_from_filename'].apply(modify_name)

    # calculate the distinct speaker names
    distinct_speaker_names = df['name_unique_speaker'].unique()
    num_distinct_speaker_names = len(distinct_speaker_names)
    print(f"Distinct speaker names: {num_distinct_speaker_names}") 
    
    
    df.to_csv("final_clean_dataset.csv", index=False)

    # get total number of rows after cleaning
    num_rows_cleaned = df.shape[0]
    print(f"Total rows after cleaning: {num_rows_cleaned}") 

    # get the total duration after cleaning
    sum_duration_cleaned = df['difference'].sum()/1000/60/60 
    print(f"The total duration of all audios after cleaning: {sum_duration_cleaned} hours")


def main():
    csv_file = "../data_processed/clean_dataset.csv"
    clean_dataset(csv_file)

if __name__ == "__main__":
    main()
