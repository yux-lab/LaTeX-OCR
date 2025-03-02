from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('dataset/test_img/3.png')
model = LatexOCR()
print(model(img))