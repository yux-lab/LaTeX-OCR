import pickle
from pix2tex.dataset.dataset import Im2LatexDataset
from transformers import PreTrainedTokenizerFast
from torchvision.transforms import ToPILImage

# path to pkl
pkl_file_path = "train.pkl"

# path to tokenizer.json
tokenizer_path = "../../model/dataset/tokenizer.json"

# load
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# load pkl
with open(pkl_file_path, 'rb') as f:
    dataset = pickle.load(f)

# by index
first_item = dataset[0]  # the first formula, but dataset is not the same as original path

#
formula_data, image_tensor = first_item

#
input_ids = formula_data["input_ids"]

#
if hasattr(input_ids, 'shape'):
    if len(input_ids.shape) > 1:
        input_ids = input_ids.squeeze(0)

# decode
latex_code = tokenizer.decode(input_ids, skip_special_tokens=True)

# replace
latex_code = latex_code.replace("Ġ", "")

print(f"LaTeX Code: {latex_code}")

# show
to_pil = ToPILImage()
image = to_pil(image_tensor.squeeze(0))
image.show()

print("-" * 40)
