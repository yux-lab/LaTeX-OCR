import pickle
from pix2tex.dataset.dataset import Im2LatexDataset
from os.path import join

# path to pkl
pkl_file_path = "data/test.pkl"

# load pkl
with open(pkl_file_path, 'rb') as f:
    dataset = pickle.load(f)

#iter(dataset)

# print info.
print("Data structure:")
for dimensions, pairs in list(dataset.data.items())[:100]:  # total images
    print(f"Images with dimensions {dimensions}:")
    for i, (latex, image_path) in enumerate(pairs[:1]):  # one by one
        print(f"  Item {i+1}:")
        print(f"    LaTeX Code: {latex}")
        print(f"    Image Path: {image_path}")