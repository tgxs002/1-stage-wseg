import os, torch, json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from .utils import colormap
# import datasets.transforms as tf
import torchvision.transforms as tf
import torchvision.transforms.functional as F

class WikiScenes(Dataset):

    dataset_classes = [
        'facade', 'apse', 'roof', 'statue', 'altar', 'choir', 
        'nave', 'transept', 'view', 'portal', 'tower', 'ambulatory', 
        'ceiling', 'chapel', 'painting', 'monument', 'dome', 
        'sculpture', 'cloister', 'crypt'
    ]

    # dataset_classes = [
    #     'apse', 'roof', 'statue', 'altar', 'choir', 
    #     'nave', 'transept', 'view', 'portal', 'tower', 'ambulatory', 
    #     'ceiling', 'chapel', 'painting', 'monument', 'dome', 
    #     'sculpture', 'cloister', 'crypt'
    # ]

    dataset_classes = [
        'nave', 'tower', 'transept', 'apse', 'portal'
    ]

    # dataset_classes = [
    #     'interior', 'exterior'
    # ]

    CLASSES = ["background"]
    CLASSES += dataset_classes
    CLASSES.append("ambiguous")

    CLASS_IDX = {}
    CLASS_IDX_INV = {}

    for i, label in enumerate(CLASSES):
        if label != "ambiguous":
            CLASS_IDX[label] = i
            CLASS_IDX_INV[i] = label
        else:
            CLASS_IDX["ambiguous"] = 255
            CLASS_IDX_INV[255] = "ambiguous"
    NUM_CLASS = len(CLASSES) - 1

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class WikiSegmentation(WikiScenes):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('../wiki_dataset')):
        super(WikiSegmentation, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        assert self.split in ['train', 'val'], "Only support train and val, but get {}".format(self.split)

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'split/train_5classes_2103.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'split/val_5classes_2103.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_voc.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.images = []
        self.labels = []
        self.captions = []
        self.tags_list = []
        # we do not have ground truth
        self.masks = None
        image_path = set()
        with open(_split_f, "r") as lines:
            for line in lines:
                _image, label, caption, tags = line.strip("\n").split(':')
                image_path.add(_image)
                _image = os.path.join(self.root, _image)
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)
                self.labels.append(label)
                self.captions.append(caption)
                self.tags_list.append(tags)

        with open("correspondence.json", 'r', encoding='utf-8') as f:
            corr = json.load(f)
        


        self.transform = tf.Compose([tf.RandomResizedCrop(cfg.DATASET.CROP_SIZE, \
            scale=(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)), \
                                     tf.RandomHorizontalFlip(), \
                                     tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), \
                                     tf.ToTensor(),
                                     tf.Normalize(self.MEAN, self.STD)])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        label  = self.labels[index]

        label_index = self.CLASS_IDX[label]

        # ignoring BG
        label_tensor = torch.zeros(self.NUM_CLASS - 1)
        label_index -= 1 # shifting since no BG class

        label_tensor[label_index] = 1

        # general resize, normalize and toTensor
        image = self.transform(image)

        # return image, label_tensor, os.path.basename(self.images[index])
        # change to caption and tags
        return image, label_tensor, "captions: {}, tags: {}".format(self.captions[index], self.tags_list[index])

    @property
    def pred_offset(self):
        return 0
