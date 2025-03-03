# to test if utterances and transcriptions of the following audio folder are aligned

import pylangacq as pla

file_path = "../data_processed/transcripts/Fridriksson-2/"
ds = pla.read_chat(file_path)

print(ds.utterances())  
