# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/convert_to_wav.sh
#!/bin/bash

# define useful directory
ORIGIN_AUDIOS_DIR="/work/van-speech-nlp/aphasia/English/Aphasia"
AUDIOS_DIR="../data_processed/audios"

# create the audios folder if it does not exist
if [ ! -d "$AUDIOS_DIR" ]; then
    mkdir "$AUDIOS_DIR"
    echo "Created directory: $AUDIOS_DIR"
fi


# check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found, please install ffmpeg first"
    exit 1
fi

# Get all subfolders from the base path
folders=$(find "$ORIGIN_AUDIOS_DIR" -type d)

for folder in $folders; do
    # Calculate the relative folder structure for the output
    relative_folder="${folder#$ORIGIN_AUDIOS_DIR/}"
    output_folder="$AUDIOS_DIR/$relative_folder"

    # Create the output folder if it doesn't exist
    mkdir -p "$output_folder"
    
    # run the python script to convert audio files
    echo "Processing folder: $folder"
    echo "Output folder: $output_folder"
    
    python3 "convert_to_wav.py" "$folder" "$output_folder"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Conversion failed for folder: $folder"
    else
        echo "Conversion successful for folder: $folder"
    fi
done