import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = r"D:/Desktop/lab/dataset/formulae/mini_test/0000391.png"
output_path = r"D:/Desktop/lab/dataset/formulae/ppt_processed.png"

# original img
image = cv2.imread(input_path)  # default RGB
if image is None:
    raise FileNotFoundError(f"can't not find: {input_path}")

# to gray img
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# save
cv2.imwrite(output_path, gray_image)

print(f"save to: {output_path}")


unique_values = np.unique(gray_image)
print("Unique Values in Grayscale Image:", unique_values)

# show original & gray
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # to RGB
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(gray_image, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Grayscale Image")
ax[1].axis("off")

plt.show()

