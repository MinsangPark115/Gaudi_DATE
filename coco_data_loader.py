import torch
import torch.utils
import pandas as pd
import os
import open_clip
from PIL import Image
import clip
from torchvision import transforms
import numpy as np

class text_image_pair(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        if resolution != 1:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', force_image_size=resolution)
        else:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        # _, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):

        img_path = os.path.join(self.dir_path, f'{idx}.png')
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()
        # image2 = self.preprocess2(raw_image).squeeze().float()
        text = self.text_description[idx]
        return image, image, text

class text_image_pair_pred(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        if resolution != 1:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', force_image_size=resolution)
        else:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        # _, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, f'{idx}.png')
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()

        lst_pred = []
        for k in range(54):
            if k not in [0, 11, 22, 33, 44]:
                img_path_k = os.path.join(self.dir_path, f'{idx}_{k}.png')
                raw_image_k = Image.open(img_path_k)
                image_k = self.preprocess(raw_image_k).squeeze().float()
                lst_pred.append(image_k)
        # image2 = self.preprocess2(raw_image).squeeze().float()
        text = self.text_description[idx]
        return image, lst_pred, text


class text_image_pair_train(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        self.image_path = df['file_name']
        if resolution != 1:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', force_image_size=resolution)
        else:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        # _, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):

        img_path = os.path.join(self.dir_path, self.image_path[idx])
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()
        # image2 = self.preprocess2(raw_image).squeeze().float()
        text = self.text_description[idx]
        return image, image, text


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class text_image_pair_org(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        self.image_path = df['file_name']
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.resolution)),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):

        img_path = os.path.join(self.dir_path, self.image_path[idx])
        text = self.text_description[idx]
        raw_image = Image.open(img_path)
        # Ensure the image is in RGB mode (3 channels)
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')

        # Apply the transformations
        tensor_image = self.transform(raw_image)

        return tensor_image, tensor_image, text


class text_image_pair_gen_org(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.resolution)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, f'{idx}.png')
        text = self.text_description[idx]
        raw_image = Image.open(img_path)
        # Ensure the image is in RGB mode (3 channels)
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')

        # Apply the transformations
        tensor_image = self.transform(raw_image)
        return tensor_image, tensor_image, text