import torch

pth_file_path = "weights.pth"

# load
state_dict = torch.load(pth_file_path, map_location=torch.device('cpu'))

# print
# print("Model parameters:")
# for key, value in state_dict.items():
#     print(f"{key}: {value.shape}")
total_params = sum(value.numel() for value in state_dict.values())
print(f"Total number of parameters: {total_params}")