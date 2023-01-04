#
import torch
import torch.utils.data as data
import torchvision.transforms as T

#
import os
import numpy as np
import h5py
from PIL import Image


class CrowdDataset_shangheiTech(data.Dataset):
    def __init__(self, root: str, image_size: int) -> None:
        super(CrowdDataset_shangheiTech, self).__init__()

        # Attributes
        self.root = root
        self.image_size = image_size
        self.transforms = None

        # If not transforms is inputed, then suggest the most basic transforms in default
        if self.transforms is None:
            self.transforms = T.Compose(
                [
                    # TODO: Resize or RandomResizeCrop
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                ]
            )

        #
        self.len = len(os.listdir(f"{self.root}/images"))

    def __len__(self):
        return self.len

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
        image = self.transforms(image)
        if image.shape[0] == 1:     # Deal with the case when input image is gray scale
            image = image.repeat(3, 1, 1)
        densityMap = self.transforms(densityMap)

        # print(image.shape, densityMap.shape)

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
