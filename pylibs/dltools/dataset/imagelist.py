#!/usr/bin/env python

import torch.utils.data as data

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    import Image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    return pil_loader(path)

def find_classes(image_list):
    classes = list()
    for _, cls in image_list:
        if not cls in classes:
            classes.append(cls)
    class_to_idx = {classes[i] : i for i in range(len(classes))}

    return classes, class_to_idx

def make_dataset(image_list, class_to_idx):
    return [(image, class_to_idx[cls]) for image, cls in image_list]

class ImageList(data.Dataset):
    def __init__(self, image_list, transform = None, target_transform = None,
                 loader = default_loader):
        '''A generic data loader that requires an image list arranged in the following way: ::

            [
                ('path/to/image1.extention', 'class-label-image1'),
                ('path/to/image2.extention', 'class-label-image1'),
                ...,
                ('path/to/imagen.extention', 'class-label-imagen')
            ]

        Args:
            image_list (list): list of images with class label.
            transform (callable, optional): function/transform that takes a PIL
                image and returns a transformed version.
            target_transform (callable, optional): function/transform that takes
                the target and transforms it.
            loader (callable, optional): function to load an image given its
                path.
        '''

	classes, class_to_idx = find_classes(image_list)
        imgs = make_dataset(image_list, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in image list'))

        self.image_list = image_list
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
