# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/data_processing.sh
#!/bin/bash

BASE_DIR="../data_processed/transcripts"

# Final CSV where all processed CSV files will be combined
OUTPUT_CSV="../data_processed/clean_dataset.csv"

# Temporary directory for storing individual CSV files
TEMP_DIR="../data_processed/temp_csvs"

# Create the temporary directory if it doesn't exist
mkdir -p "$TEMP_DIR"

# Remove any old output CSV file
rm -f "$OUTPUT_CSV"

# Process each folder in the base directory one by one
for folder in "$BASE_DIR"/*/; do
    echo "Processing folder: $folder"
·
    # Run the Python script for each folder
    python3 data_processing.py "$folder"
    
    OUTPUT_FILE="$folder/clean_dataset.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        # Move the CSV to the temp directory
        cp "$OUTPUT_FILE" "$TEMP_DIR/$(basename "$folder")_clean_dataset.csv"
    else
        echo "No CSV found in $folder"
    fi
done

# Combine all the individual CSVs into one
echo "Combining all CSV files into one..."
# Add header from the first file
head -n 1 "$(ls "$TEMP_DIR"/*_clean_dataset.csv | head -n 1)" > "$OUTPUT_CSV"
# Combine the rest without headers
for file in "$TEMP_DIR"/*_clean_dataset.csv; do
    tail -n +2 "$file" >> "$OUTPUT_CSV"
done

# Clean up temporary CSV files
rm -rf "$TEMP_DIR"

echo "All CSV files have been combined into: $OUTPUT_CSV"