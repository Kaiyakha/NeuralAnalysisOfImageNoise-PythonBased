import os, dill
import numpy as np
from PIL import Image

TRESHOLD = 0.2
PATH = os.path.dirname(__file__) + "/Patches/Noisy_Patches/B/"

img = Image.open(PATH + "2548, 504.bmp")
img_matrix = np.asarray(img).reshape(img.width * img.height)
img.show()
img.close()

with open("Practice/trained_nn.pkl", "rb") as pklfile:
    nn = dill.load(pklfile)

nn.forward_prop(img_matrix)
print("Активность выходных нейронов:")
for i in range(0, nn.shape[-1], nn.shape[-1] // 4):
    print(*nn.activations[-1][i : i + 7].round(2))
print("Зашумление в полосах:", *np.nonzero(nn.activations[-1] > TRESHOLD)[0])