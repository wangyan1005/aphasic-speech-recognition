from datasets import load_from_disk
from torch.utils.data import DataLoader


# Load  DatasetDict from disk
dataset_dict_path = "/scratch/wang.yan8/dataset_dict_small"
dataset_dict = load_from_disk(dataset_dict_path)

# Print a summary of the dataset
print(dataset_dict)


train_sample = dataset_dict["train"][0]
print("Checking dataset before training starts:")
print(f"Sample keys: {train_sample.keys()}")  
print(f"Labels: {train_sample.get('labels', 'Missing!')}")




