# based on https://huggingface.co/blog/fine-tune-whisper

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
   

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # compute the attention mask for the input features
        batch["attention_mask"] = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)
        
        label_features = []
        for i, feature in enumerate(features):
            if 'labels' not in feature:
                available_keys = list(feature.keys())
                raise KeyError(f"Feature {i} missing 'labels' key. Available keys: {available_keys}")
            label_features.append({"input_ids": feature["labels"]})
        
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # Process x-vector:
        if "xvector" in features[0]:
            xvector_list = [feature["xvector"] for feature in features]
            batch["xvector"] = torch.stack(xvector_list)
        return batch
       
