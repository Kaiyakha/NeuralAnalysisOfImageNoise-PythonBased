import os, dill
from GetData import *

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + "/Patches/Noisy_Patches/"
TRAIN_PATH = PATH + "R/"
TARGET_PATH = PATH + "strip_ids_R.csv"

X, Y = getData(TRAIN_PATH, TARGET_PATH)

with open("Practice/trained_nn.pkl", "rb") as pklfile:
    nn = dill.load(pklfile)

nn.test(X, Y)