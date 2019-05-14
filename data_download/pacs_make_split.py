import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', default="train", help='The split to use')
parser.add_argument('--domain', default="cartoon", help='The domain to split')

args = parser.parse_args()

root = '/home/fcdl/dataset/pacs/'
source = root + "kfold/"

dest = root + args.split + "/" + args.domain

split_name = "crossval" if args.split == 'val' else args.split

split_file = root + "official_splits/" + args.domain + "_" + split_name + "_kfold.txt"

os.makedirs(dest, exist_ok=True)

with open(split_file, "r") as file:
    for line in file.readlines():
        img = source + line.split(" ")[0]
        dest_cl = dest + "/" + line.split("/")[1] + "/"

        if not os.path.exists(dest_cl):
            os.makedirs(dest_cl)

        print(f"Parsing image {img}")

        os.link(img, dest_cl + line.split(" ")[0].split("/")[2])