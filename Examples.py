# An example to show how the trained network works

import os, dill
import numpy as np
from PIL import Image

TRESHOLD = 0.2 # Affects sensitivity of the network
PATH = os.path.dirname(__file__)
DATA_PATH = PATH + "/Patches/Noisy_Patches/R/"

img = Image.open(DATA_PATH + "0, 2828.bmp")
img_matrix = np.asarray(img).reshape(img.width * img.height)
img.show()
img.close()

with open(PATH + "/trained_nn.pkl", "rb") as pklfile:
    nn = dill.load(pklfile)

nn.forward_prop(img_matrix)
print("The sigmoid function value in the output layer:")
for i in range(0, nn.shape[-1], nn.shape[-1] // 4):
    print(*nn.activations[-1][i : i + 7].round(2))
print("Corrupted strips:", *np.nonzero(nn.activations[-1] > TRESHOLD)[0])