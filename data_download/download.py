import os
from torchvision.datasets.utils import download_url
import shutil


def download_gtsrb(dir):
    # extract file
    url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
    filename = "gtsrb.zip"
    download_url(url, dir, filename)

    import zipfile
    with zipfile.ZipFile(os.path.join(dir, filename), "r") as zip_ref:
        zip_ref.extractall(dir)

    url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
    download_url(url, dir, filename)

    with zipfile.ZipFile(os.path.join(dir, filename), "r") as zip_ref:
        zip_ref.extractall(dir)


def download_syn_sign(dir):
    """
    We cannot download directly because it's hosted on dropbox.
    Please, download it from here
    https://www.dropbox.com/s/7izi9lccg163on1/synthetic_data.zip?dl=0
    and then use this script
    """
    # extract file
    import zipfile
    with zipfile.ZipFile(os.path.join(dir, "synthetic_data.zip"), "r") as zip_ref:
        zip_ref.extractall(dir)

    root = os.path.join(dir, "synthetic_data")
    for i in range(43):
        os.makedirs(os.path.join(root, f"{i:05d}"), exist_ok=True)

    with open(os.path.join(dir, "synthetic_data", "train_labelling.txt"), "r") as train_label:
        for line in train_label.readlines():
            fname, label, _ = line.split(" ")
            print(fname, label)
            shutil.move(os.path.join(root, fname), os.path.join(root, f"{int(label):05d}"))

    os.rmdir(os.path.join(root, "train"))


if __name__ == "__main__":
    download_syn_sign("/home/fcdl/dataset")
