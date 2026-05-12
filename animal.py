from jupyter_core.migrate import security_file
from sympy import transpose
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Resize
import numpy as np
from PIL import Image

transformer = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

class Animaldataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.images_path = []
        self.labels_path = []
        self.transform = transform
        if train:
            datafiles = os.path.join(root, "train")
        else:
            datafiles = os.path.join(root, "test")

        files = [f for f in os.listdir(datafiles) if not f.endswith((".DS_Store", "Must read.txt"))]
        for index, f in enumerate(files):
            filename = os.path.join(datafiles, f)
            for i in [f for f in os.listdir(filename) if not f.endswith((".DS_Store", "Must read.txt"))]:
                filepath = os.path.join(filename, i)
                self.images_path.append(filepath)
                self.labels_path.append(index)

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        # image_path = cv2.imread(self.images_path[idx])
        image_path = Image.open(self.images_path[idx]).convert("RGB")
        # image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        if self.transform:
            image_path = self.transform(image_path)
        label = self.labels_path[idx]
        return image_path, label

if __name__ == "__main__":

    train_dataset = Animaldataset(root="/Users/tavantai/Developer/PycharmProjects/vietnguyencourse/BasicDeepLearning/animals", train=True, transform=transformer)
    # image, label = train_dataset.__getitem__(8028)
    # transform = transforms.ToTensor()
    # image = transform(image)
    # print(image.shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )
    for image, label in train_loader:
        print(image.shape)
        print(type(label))