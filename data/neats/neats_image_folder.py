"""
 A PyTorch dataset for loading NEATS data.
    
"""

import json
import numpy as np
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def _get_transform(mode):
    """ Returns an image transform pipeline.
    """
    # augmentation params
    im_size = [299, 299]  # can change this to train on higher res
    mu_data = [0.485, 0.456, 0.406]
    std_data = [0.229, 0.224, 0.225]
    brightness, contrast, saturation, hue = 0.4, 0.4, 0.4, 0.25

    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=im_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness, contrast, saturation, hue),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu_data, std=std_data),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu_data, std=std_data),
            ]
        )



class NEATS(data.Dataset):
    def __init__(self, root, ann_file, mode='train', transform=None, class_index_offset=0, full_info=False):
        """ A Dataset for NEATS data.
        
        Args:
            data ([type]): Parent class.
            root (str or Path): Path to the root folder.
            mode (str, optional): Defaults to "train". Establishing if the
                dataset is of type `train`, `validation` or `test` and loads
                the coresponding data.
            transform (torchvision.transforms.Transform, optional): Defaults
                to None. A transform function fore preprocessing and
                augmenting images.
            full_info (bool, optional): Defaults to False. If `True` the
                loader will return also the `taxonomic_class` and the `img_id`.
        """
        self._full_info=full_info
        
        try:
            self._root = root
            self.annotations_path = root / ann_file
        except TypeError:
            self._root = root = Path(root)
            self._ann_file = ann_file = root / ann_file

        # load annotations
        print(f"iNaturalist: loading annotations from: {ann_file}.")
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # A list of dicts with all the genus and family names and ids associated with a species id
        self.categories = ann_data["categories"]
        print(f"Annotations categories: {self.categories}")
        # A dictionary of family ids with their child genus and species.
        self.topdown_mapper = ann_data["topdown"][0]
        print(f"Topdown hierarchy mapper: {self.topdown_mapper}")

        # set up the filenames and annotations
        self._img_paths = [root / aa["file_name"] for aa in ann_data["images"]]  # ** modified

        # if we dont have class labels set them to '0'
        if "annotations" in ann_data.keys():
            self._classes = [a["category_id"] for a in ann_data["annotations"]]
        else:
            self._classes = [0] * len(self._img_paths)

        self._num_classes = len(set(self._classes))
        self.class_index_offset = class_index_offset

        if full_info:
            # get image id
            self._img_ids = [aa["id"] for aa in ann_data["images"]]

        # image loading, preprocessing and augmentations
        self.loader = default_loader
        if transform:
            self.transform = transform
        else:
            self.transform = _get_transform(mode)

        # print out some stats
        print(f"NEATS: found {len(self._img_paths)} images.")
        print(f"NEATS: found {len(set(self._classes))} classes.")

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        img = self.loader(self._img_paths[index])
        species_id = self._classes[index]  # class
        # species_id -= 1  # offsets it by -1 since cat_id starts at 1 but category indexing starts at 0

        if self.transform:
            img = self.transform(img)

        if self._full_info:
            # we can also return some additionl info
            img_id = self._img_ids[index]
            genus_id = self.categories[species_id]['genus_id']
            family_id = self.categories[species_id]['family_id']
            # # Debugging
            # print(f"Image name = {self._img_paths[index]}")
            # print(f"Decoded taxon = {self.categories[species_id]['family']}_{self.categories[species_id]['genus']}_{self.categories[species_id]['species']}")

            return img, species_id+self.class_index_offset, genus_id+self.class_index_offset, family_id+self.class_index_offset, img_id
        return img, species_id
    
    def __str__(self):
        details = f"len={len(self)}, mode={self._mode}, root={self._root}"
        return f"NEATSDataset({details})"

    def __len__(self):
        return len(self._img_paths)


if __name__ == "__main__":
    import torch

    dset = NEATS(root="./data/", mode="train")
    train_loader = torch.utils.data.DataLoader(
        dset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )

    print(f"Iterating through {dset} ...")
    for i, (imgs, target) in enumerate(train_loader):
        print(f"batch={i:5d}, img={imgs.shape}, target={target.shape}")
        # print("Img ids: ", [idx.item() for idx in imgs_ids[:5]])
