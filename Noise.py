# Create artificial noise on each image
# The resulting images are in the R channel
# But other channels can also be chosen

import os, random
import csv
from PIL import Image

PATH = os.path.dirname(__file__) + "/Patches/"
INPUT_PATH = PATH + "Clear_Patches/"
OUTPUT_PATH = PATH + "Noisy_Patches/R/"
try: os.makedirs(OUTPUT_PATH)
except FileExistsError: pass

STRIP_CHANCE = 0.02 # Affects the amount of strips in an image


def makeNoise(img):
    imgMatrix = img.load()
    strip_ids = []

    for i in range(img.width):
        if random.random() < STRIP_CHANCE:
            maxPix = max([imgMatrix[i, j] for j in range(img.height)])
            a = random.uniform(0, 255 / maxPix if maxPix else 0)
            b = random.randrange(0, 255 - int(a * maxPix))
            for j in range(img.height): imgMatrix[i, j] = int(a * imgMatrix[i, j] + b)
            strip_ids.append(str(i))

    strip_ids = ",".join(strip_ids)
    return strip_ids


Y = []
for file in os.listdir(INPUT_PATH):
    patch = Image.open(INPUT_PATH + file)
    patch, _, _ = patch.convert("RGB").split() # Only consider the R channel
    strip_ids = makeNoise(patch)
    Y.append(strip_ids)
    patch.save(OUTPUT_PATH + file, 'bmp')
    patch.close()

# The ordinary number of each strip in each image gets stored into a file
# The data is used as target data for a neural network
with open(PATH + "Noisy_Patches/strip_ids_R.csv", "w", newline = "\n") as csvfile:
    fields = "Image", "Ids"
    writer = csv.DictWriter(csvfile, fieldnames = fields, delimiter = ";")
    writer.writeheader()
    for i in range(len(Y)):
        writer.writerow({"Image": os.listdir(INPUT_PATH)[i], "Ids": Y[i]})