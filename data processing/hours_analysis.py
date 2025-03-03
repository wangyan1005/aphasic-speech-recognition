'''
check per unique speaker's audio duration for further data processing
'''

import pandas as pd

# Load CSV file
file_path = "final_clean_dataset.csv"
df = pd.read_csv(file_path)

# Compute the duration for each entry (end - start)
df["duration"] = df["mark_end"] - df["mark_start"]  # Duration in milliseconds

# Aggregate total duration per unique speaker
speaker_durations = df.groupby("name_unique_speaker")["duration"].sum().reset_index()

# Convert milliseconds to hours
speaker_durations["total_duration_hours"] = speaker_durations["duration"] / (1000 * 60 * 60)

# Find speakers with the longest and shortest total speech time
max_speaker = speaker_durations.loc[speaker_durations["total_duration_hours"].idxmax()]
min_speaker = speaker_durations.loc[speaker_durations["total_duration_hours"].idxmin()]

# Print results
print(f"Speaker with most speech: {max_speaker['name_unique_speaker']}, Total Duration: {max_speaker['total_duration_hours']:.2f} hours")
print(f"Speaker with least speech: {min_speaker['name_unique_speaker']}, Total Duration: {min_speaker['total_duration_hours']:.2f} hours")

# Save results to CSV
speaker_durations.to_csv("speaker_durations.csv", index=False)
