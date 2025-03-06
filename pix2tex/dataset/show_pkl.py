import pickle
from pix2tex.dataset.dataset import Im2LatexDataset
from os.path import join
import numpy as np

# Path to pkl file
pkl_file_path = "data/mini_test.pkl"

# Load the dataset from the pkl file
with open(pkl_file_path, 'rb') as f:
    dataset = pickle.load(f)

# Ensure the dataset is iterable
#iter(dataset)

# Print info and tensors
print("Data structure:")
count = 0

for dimensions, pairs in list(dataset.data.items())[:13]:  # Process a few dimensions
    print(f"Images with dimensions {dimensions}:")
    for i, (latex, image_path) in enumerate(pairs[:1]):  # Process one pair per dimension
        count += 1
        print(f"  Item {i+1}:")
        print(f"    LaTeX Code: {latex}")
        print(f"    Image Path: {image_path}")

        # Prepare data to get tensors
        batch = np.array([(latex, image_path)])  # Convert to NumPy array
        tok, images = dataset.prepare_data(batch)

        # Check if tensors are successfully prepared
        if tok is not None and images is not None:
            print("    Tokenized LaTeX (input_ids):", tok['input_ids'])
            print("    Attention Mask:", tok['attention_mask'])
            print("    Image Tensor Shape:", images.shape)
        else:
            print("    Failed to prepare tensors for this item.")

print(f"Total items processed: {count}")