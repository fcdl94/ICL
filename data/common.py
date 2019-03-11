from PIL import Image
from torch.utils.data import Dataset


class DatasetPrototypes(Dataset):

    def __init__(self, x, y, transform=None):
        super().__init__()
        assert len(x) == len(y), "Error, the size of x and y must match"
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):

        sample = Image.fromarray(self.x[index])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.y[index]

    def __len__(self):
        return len(self.y)
