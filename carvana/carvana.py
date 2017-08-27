import os

from torch.utils.data.dataset import Dataset
from PIL import Image

class CARVANA(Dataset):
    def __init__(self, root, subset = 'train', transform = None):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.transform = transform

        self.data_path = list()
        self.image_dir = os.path.join(root, subset)
        for f in os.listdir(self.image_dir):
            ff = os.path.join(self.image_dir, f)
            if os.path.isfile(ff):
                self.data_path.append(ff)
        self.data_path.sort()

        self.label_path = list()
        self.label_dir = os.path.join(root, subset + '_masks')
        for f in os.listdir(self.label_dir):
            ff = os.path.join(self.label_dir, f)
            if os.path.isfile(ff):
                self.label_path.append(ff)
        self.label_path.sort()

    def __getitem__(self, index):
        img = Image.open(self.data_path[index])
        mask = Image.open(self.label_path[index])

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.data_path)

ds = CARVANA('/data/noud/kaggle/carvana/')
