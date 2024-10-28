import os
import shutil

path = "/cver/dcjxc/Dataset/RGBnPIL1/Edge"
new_path = "/cver/dcjxc/Dataset/RGBnPIL1/Infr"
images = [ f for f in os.listdir(path) if f.endswith('_nir.png')]
for image in images:
    os.remove(path + '/' + image)
# print(len(images))
