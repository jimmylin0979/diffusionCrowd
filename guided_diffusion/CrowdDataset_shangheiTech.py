#
import torch
import torch.utils.data as data
import torchvision.transforms as T
import random

#
import os
import numpy as np
import h5py
from PIL import Image


class CrowdDataset_shangheiTech(data.Dataset):
    def __init__(self, root: str, image_size: int, test_flag=False) -> None:
        super(CrowdDataset_shangheiTech, self).__init__()

        # Attributes
        self.root = root
        self.image_size = image_size
        self.test_flag = test_flag
        # self.transforms = None

        # If not transforms is inputed, then suggest the most basic transforms in default
        # if self.transforms is None:
        #     self.transforms = T.Compose(
        #         [
        #             # TODO: Resize or RandomResizeCrop
        #             T.Resize((self.image_size, self.image_size)),
        #             T.ToTensor(),
        #         ]
        #     )

        #
        self.len = len(os.listdir(f"{self.root}/images"))

    def __len__(self):
        return self.len

    def transforms(self, image, mask):

        # Resize
        resize = T.Resize(size=(self.image_size * 2, self.image_size * 2))
        # print(image.size)
        if image.size[0] < self.image_size or image.size[1] < self.image_size:
            image = resize(image)
            mask = resize(mask)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size)
        )
        image = T.functional.crop(image, i, j, h, w)
        mask = T.functional.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)

        # Transform to tensor
        image = T.functional.to_tensor(image)
        mask = T.functional.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):

        # paths
        path_image = f"{self.root}/images/IMG_{index + 1}.jpg"
        path_h5 = f"{self.root}/ground-truth-h5/IMG_{index + 1}.h5"
        path_mat = f"{self.root}/ground-truth/GT_IMG_{index + 1}.mat"

        # Load original image
        image = Image.open(path_image)

        # Load density map
        densityMap = h5py.File(path_h5, "r")["density"]
        densityMap = np.asarray(densityMap)
        densityMap = Image.fromarray(densityMap)
        # densityMap = torch.from_numpy(densityMap)
        # densityMap = densityMap.unsqueeze(0)

        # Apply transforms
        # image = self.transforms(image)
        # densityMap = self.transforms(densityMap)
        image, densityMap = self.transforms(image, densityMap)

        if image.shape[0] == 1:  # Deal with the case when input image is gray scale
            image = image.repeat(3, 1, 1)

        # print(image.shape, densityMap.shape)

        #
        if self.test_flag:
            return (image, path_image)

        #
        return (image, densityMap)


if __name__ == "__main__":

    #
    root = "../datasets/ShanghaiTech/part_A/train_data"
    dataset = CrowdDataset_shangheiTech(root=root)
    print("len", len(dataset))

    #
    index = 0
    print("shape", dataset[index][0].shape)
    print("shape", dataset[index][1].shape)

    # len 300
    # shape torch.Size([3, 768, 1024])
    # shape torch.Size([1, 768, 1024])
