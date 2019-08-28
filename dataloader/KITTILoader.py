import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
# imagenet_stats = {'mean': [0.485, 0.456, 0.406],
#                    'std': [0.229, 0.224, 0.225]}
imagenet_stats = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**imagenet_stats)])
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))

            dataL = dataL.crop((w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**imagenet_stats)])
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
