from PIL import Image
import os

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + '/'
PATCHES_PATH = PATH + "Patches/Clear_Patches/"
FILE = "vd1.bmp"
try: os.makedirs(PATCHES_PATH)
except FileExistsError: pass

scene = Image.open(PATH + FILE)

for i in range(0, scene.width - WIDTH, WIDTH):
    for j in range(0, scene.height - HEIGHT, HEIGHT):
        patch = scene.crop((i, j, i + WIDTH, j + HEIGHT))
        patch.save(PATCHES_PATH + f"{i}, {j}.bmp", 'bmp')
        patch.close()

scene.close()