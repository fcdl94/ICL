import os
import numpy as np
import shutil as sh

root = '/home/fcdl/dataset/sketchy'
dataset = root + "/sketch"
dest_train = dataset + "_train/"
dest_test = dataset + "_test/"

os.makedirs(dest_test, exist_ok=True)
os.makedirs(dest_train, exist_ok=True)


if __name__ == '__main__':
    for dir in os.listdir(dataset):
        dir_path = os.path.join(dataset, dir)

        dest_dir_test = os.path.join(dest_test, dir)
        dest_dir_train = os.path.join(dest_train, dir)
        os.makedirs(dest_dir_test, exist_ok=True)
        os.makedirs(dest_dir_train, exist_ok=True)

        images = list(os.listdir(dir_path))
        im_len = len(images)

        indices = np.arange(im_len)
        np.random.shuffle(indices)
        train, test = indices[100:], indices[:100]

        print(f"Parsing images of {dir}, # {len(train)} train images, # {len(test)} test images")

        for i in train:
            os.link(os.path.join(dir_path, images[i]), os.path.join(dest_dir_train, images[i]))
        for i in test:
            os.link(os.path.join(dir_path, images[i]), os.path.join(dest_dir_test, images[i]))
