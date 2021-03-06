import os, torch, json, random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from .utils import colormap
import datasets.transforms as tf
import torchvision.transforms.functional as F
from torchvision.utils import save_image as sv
from collections import Counter

class WikiScenes_corr(Dataset):

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
    #     'facade', 'nave', 'altar', 'tower', 'glass', 'transept', 'portal', 'monument', 'cloister', 'dome', 'apse', 'chancel'
    # ]
    # 5023  1873    928     1099    4654    1624    1721    1419    376     283     581     804
    # 0.83  0.57    0.09    0.18    0.95    0.21    0.28    0.16    0.03    0.04    0.12    0.34
    # 18499

    # 0.83  0.95`   0.57    0.28    0.21    0.16    0.18    0.09    0.34    0.12    0.03    0.02


    # dataset_classes = [
    #     'facade', 'nave', 'altar', 'tower', 'glass', 'transept', 'portal', 'monument', 'chancel'
    # ]

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


class WikiSegmentation_corr(WikiScenes_corr):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('../wiki_dataset')):
        super(WikiSegmentation_corr, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        assert self.split in ['train', 'val'], "Only support train and val, but get {}".format(self.split)

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'split/train_5classes_3269_corr.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'split/val_5classes_1000_corr.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_voc.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        with open("correspondence.json", 'r', encoding='utf-8') as f:
            corr = json.load(f)
        # corr = dict()

        self.images = []
        self.labels = []
        self.captions = []
        self.tags_list = []
        self.keypoints = []
        # we do not have ground truth
        self.masks = None
        self.corr_index = dict()
        with open(_split_f, "r") as lines:
            print("building graph")
            count = 0
            for line in lines:
                _image, label, caption, tags = line.strip("\n").split(':')
                image_corr = _image[22:]
                if image_corr in corr:
                    k = corr[image_corr]
                    for c in k:
                        p = k[c]
                        if c in self.corr_index:
                            self.corr_index[c].add(count)
                        else:
                            self.corr_index[c] = set([count])
                        k[c] = (p[1], p[0])
                    self.keypoints.append(k)
                else:
                    self.keypoints.append(None)
                _image = os.path.join(self.root, _image)
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)
                self.labels.append([x.strip("',") for x in label.strip("[]").split()])
                self.captions.append(caption)
                self.tags_list.append(tags)
                count += 1

        print("Filtering")
        self.image_graph = dict()
        for i, keypoint in enumerate(self.keypoints):
            if keypoint != None:
                t = Counter()
                for p in keypoint:
                    t.update(self.corr_index[p])
                t = [key for key, cnt in t.items() if cnt >= 10]
                if t:
                    self.image_graph[i] = t

        self.transform = tf.Compose([tf.RandResizedCrop_corr(self.cfg.DATASET), \
                                     tf.HFlip_corr(), \
                                     tf.ColourJitter_corr(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), \
                                     tf.Normalise_corr(self.MEAN, self.STD)
                                     ])

        print("{}/{} images have keypoints, {} keypoints in total, {} pairs per image.".format(len(self.image_graph), len(self.keypoints), len(self.corr_index), sum([len(k) for k in self.image_graph.values()]) / (len(self.image_graph) + 0.001)))
        self.cnt = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        keypoints = self.keypoints[index]
        pair_index = None
        # find a image with correspondence
        if keypoints != None and index in self.image_graph:
            p = random.choice(self.image_graph[index])
            if p != index:
                pair_index = p
        # if no correspondence, randomly pick one
        if pair_index == None:
            pair_index = random.randint(0, len(self.images) - 1)

        images = list()
        label_tensors = list()
        captions = list()
        kp = list()
        for i in [index, pair_index]:
            image = Image.open(self.images[i]).convert('RGB')
            label  = self.labels[i]
            keypoints = self.keypoints[i].copy() if self.keypoints[i] != None else None

            # ignoring BG
            label_tensor = torch.zeros(self.NUM_CLASS - 1)
            for l in label:
                label_index = self.CLASS_IDX[l]
                label_index -= 1 # shifting since no BG class
                label_tensor[label_index] = 1

            # general resize, normalize and toTensor
            image, keypoints = self.transform(image, keypoints)
            caption = "captions: {}, tags: {}".format(self.captions[i], self.tags_list[i])

            images.append(image)
            label_tensors.append(label_tensor)
            captions.append(caption)
            kp.append(keypoints)
        # get correspondence
        common = set(kp[0]).intersection(set(kp[1])) if kp[0] != None and kp[1] != None else None
        if common != None:
            corr = list()
            for i in common:
                corr.append([*kp[0][i], *kp[1][i]])
            corr = json.dumps(corr)
            # sv(images[0], "./images/{}.jpg".format(self.cnt))
            # sv(images[1], "./images/{}.jpg".format(self.cnt+1))
            # self.cnt += 2
            # if self.cnt > 100:
            #     quit()
        else:
            corr = "[]"
        images = {"1": images[0], "2": images[1], "corr": corr}
        return images, label_tensors, captions

    @property
    def pred_offset(self):
        return 0
