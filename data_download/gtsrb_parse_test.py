import os
import numpy as np
import shutil as sh

root = "/home/fcdl/dataset/GTSRB"
test = root+'/Final_Test'
images = test+"/Images"

# open file
annotations = root + "/GT-final_test.csv"

for i in range(43):
    os.makedirs(os.path.join(test, f"{i:05d}"), exist_ok=True)

with open(annotations, "r") as ann:
    i = 0
    for line in ann.readlines():
        if i > 0:
            fields = line.split(";")
            image_name = fields[0]
            class_id = f"{int(fields[-1]):05d}"
            print(f"{i:5d}: Fetching " + image_name + " of class " + class_id)
            sh.copy(os.path.join(images, image_name), os.path.join(test, class_id, image_name))
        i += 1
