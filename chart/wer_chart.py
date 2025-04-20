import matplotlib.pyplot as plt

# LoRA ranks and corresponding WER values
lora_ranks = [8, 64, 128, 256]
wer_values_decoder = [49.76, 48.12, 48.89, 47.64]
wer_values_encoder_decoder = [49.59, 45.81, 48.16, 46.02]
baseline_wer = 46.94

# Create the figure
plt.figure(figsize=(6, 4))

# Plot the line graph with black markers
plt.plot(lora_ranks, wer_values_encoder_decoder, marker='o', color='black', label='MLP W1 Layers(Encoder+Decoder) Fine-Tuned')

# Add the baseline as a dashed horizontal line
plt.axhline(y=baseline_wer, linestyle='--', color='black', label='Baseline (46.94%)')
plt.plot(lora_ranks, wer_values_decoder, marker='s', color='black', label='MLP W1 Layers(Decoder) Fine-Tuned')

# Set the axes labels
plt.xlabel('LoRA Rank')
plt.ylabel('WER (%)')

# Display only the specified x-axis ticks
plt.xticks(lora_ranks)

# Set the y-axis ticks from 10% to 70% with 10% intervals
plt.yticks(range(10, 71, 10))
plt.ylim(10, 70)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()
plt.tight_layout()
plt.show()
