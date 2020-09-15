from __future__ import print_function
from core.config import cfg, cfg_from_file, cfg_from_list
import os, json, argparse, sys, math, torch, copy, random

import torch.nn as nn
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import matplotlib.cm as mpl_color_map
import numpy as np
from PIL import Image, ImagePalette
from models import get_model
from opts import get_arguments
from datasets.utils import Colorize
from skimage.transform import resize

def load_image(image_path):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = tf.Compose([tf.ToTensor(),
                            tf.Normalize(MEAN, STD)])
    raw = Image.open(image_path).convert('RGB')
    image = transform(raw)
    return image.unsqueeze(0), raw

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).resize(org_im.size, Image.ANTIALIAS)
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image

def concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class Compare():

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.image_path = None
        self.model = None
        self.model2 = None
        self.current_model_name = None
        self.backup_model_name = None
        self.raw = None
        self.feature = None
        self.backup_feature = None
        self.rx = None
        self.ry = None

    def load_model(self, model_path):
        assert os.path.exists(model_path), "please provide a valid model path"
        # load model
        self.model = get_model(self.cfg.NET)
        weight = torch.load(self.args.weight, map_location='cpu')
        weight.pop("last_conv.0.weight")
        weight.pop("last_conv.0.bias")
        weight.pop("_aff.aff_x.kernel")
        weight.pop("_aff.aff_m.kernel")
        weight.pop("_aff.aff_std.kernel")
        self.model.load_state_dict(weight)
        self.model.eval()
        self.current_model_name = self.args.weight
        if self.args.weight2 != None:
            self.model2 = get_model(self.cfg.NET)
            weight2 = torch.load(self.args.weight2, map_location='cpu')
            weight2.pop("last_conv.0.weight")
            weight2.pop("last_conv.0.bias")
            weight2.pop("_aff.aff_x.kernel")
            weight2.pop("_aff.aff_m.kernel")
            weight2.pop("_aff.aff_std.kernel")
            self.model2.load_state_dict(weight2)
            self.model2.eval()
            self.backup_model_name = self.args.weight2
        if self.image_path != None:
            self.load_image(self.image_path[0], self.image_path[1])

    def load_image(self, image_path1, image_path2):
        assert os.path.exists(image_path1), "{} does not exist".format(image_path1)
        assert os.path.exists(image_path2), "{} does not exist".format(image_path2)
        self.image_path = [image_path1, image_path2]
        image1, raw1 = load_image(image_path1)
        image2, raw2 = load_image(image_path2)
        self.raw = [raw1, raw2]

        assert self.model != None, "you need to load model before run this method"
        # run model, get feature
        with torch.no_grad():
            feature1 = self.model(image1)
            feature2 = self.model(image2)
            feature1 = feature1 / feature1.norm(dim=1, keepdim=True)
            feature2 = feature2 / feature2.norm(dim=1, keepdim=True)
            self.feature = [feature1, feature2]

        rx1 = feature1.shape[2] / image1.shape[2]
        ry1 = feature1.shape[3] / image1.shape[3]
        rx2 = feature2.shape[2] / image2.shape[2]
        ry2 = feature2.shape[3] / image2.shape[3]
        self.rx = [rx1, rx2]
        self.ry = [ry1, ry2]

    def change_model(self):
        assert self.model2 != None, "No second model specified"
        self.model, self.model2 = self.model2, self.model
        self.current_model_name, self.backup_model_name = self.backup_model_name, self.current_model_name
        if self.backup_feature == None:
            self.backup_feature = self.feature
            self.load_image(self.image_path[0], self.image_path[1])
        else:
            self.backup_feature, self.feature = self.feature, self.backup_feature

    def query(self, x, y, pic):
        assert pic in [0, 1], "you need to query picture1 or picture2"
        assert self.feature != None, "you need to load image first"
        selected_feature = self.feature[pic][0,:,int(y * self.rx[pic]), int(x * self.ry[pic])]
        heat = [torch.norm((self.feature[i][0] - selected_feature[:,None,None]), dim=0) for i in [0,1]]

        # normalize separately
        min_ = torch.min(heat[0].min(), heat[1].min())
        range_ = torch.max(heat[0].max(), heat[1].max()) - min_
        heat = [(heat[i] - min_) / range_ for i in [0,1]]

        # [0,1] -> [0,1], 0 -> 1, 1 -> 0
        heat = [1 - ((heat[i] * 2 - 1) * (heat[i] * 2 - 1) * (heat[i] * 2 - 1) / 2 + 0.5) for i in [0,1]]

        # put color
        colormap = [apply_colormap_on_image(self.raw[i], heat[i], 'jet') for i in [0,1]]

        # draw a cross
        green = (0, 255, 0)
        mark_size = 5
        for i in range(x - mark_size, x + 1 + mark_size):
            if 0 <= i < colormap[pic].size[0]:
                colormap[pic].putpixel((i, y), green)
        for j in range(y - mark_size, y + 1 + mark_size):
            if 0 <= j < colormap[pic].size[1]:
                colormap[pic].putpixel((x, j), green)

        return colormap

def gui(compare):
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    tk_image1 = ImageTk.PhotoImage(compare.raw[0])
    tk_image2 = ImageTk.PhotoImage(compare.raw[1])
    label1 = tk.Label(root, image=tk_image1, borderwidth=0, highlightthickness=0, padx=0, pady=0)
    label2 = tk.Label(root, image=tk_image2, borderwidth=0, highlightthickness=0, padx=0, pady=0)
    label1.image = tk_image1
    label2.image = tk_image2
    label1.pack(side=tk.LEFT)
    label2.pack(side=tk.LEFT)
    last_operation = dict()
    cache = None

    def update(event, pic):

        colormap1, colormap2 = compare.query(event.x, event.y, pic)

        tk_image1 = ImageTk.PhotoImage(colormap1)
        tk_image2 = ImageTk.PhotoImage(colormap2)

        label1.configure(image=tk_image1)
        label2.configure(image=tk_image2)
        label1.image = tk_image1
        label2.image = tk_image2

    def mouseClick(event):
        update(event, 0)
        last_operation["event"] = event
        last_operation["anchor"] = 0
        cache = None

    def mouseClick2(event):
        update(event, 1)
        last_operation["event"] = event
        last_operation["anchor"] = 1
        cache = None

    def switch():
        compare.change_model()
        text.set(compare.current_model_name)
        if last_operation:
            update(last_operation["event"], last_operation["anchor"])

    label1.bind("<Button>", mouseClick)
    label2.bind("<Button>", mouseClick2)

    if args.weight2 != None:
        button = tk.Button(root, 
                       text="switch", command=switch)
        button.pack(side=tk.TOP)
        text = tk.StringVar()
        info = tk.Label(root, textvariable=text)
        info.pack(side=tk.TOP)
        text.set(compare.current_model_name)

    root.mainloop()

def random_generate(compare):
    from pathlib import Path
    image_list = list(Path(args.image_folder).rglob("*.[jJpP][pPnN][gG]"))
    info = dict()
    assert not os.path.exists(args.id), "the --id folder already exists, please assign a new one."
    os.mkdir(args.id)
    for i in range(args.sample_size):
        # sample two images and the location
        images = random.sample(image_list, 2)
        compare.load_image(str(images[0]), str(images[1]))
        index = random.choice([0,1])
        width, height = compare.raw[index].size
        px = random.randint(0, width - 1)
        py = random.randint(0, height - 1)
        # record info
        info[i] = {"image1": str(images[0]), "image2": str(images[1]), "anchor": index, "x": px, "y": py}
        # compute and concate
        im1, im2 = compare.query(px, py, index)
        result = concat_h(im1, im2)
        result.save(os.path.join(args.id, "{}.png".format(i)))
    with open(os.path.join(args.id, "record.json"), 'w') as f:
        json.dump(info, f)


def from_file(compare):
    assert os.path.exists(args.file), "config file doesn't exist"
    assert not os.path.exists(args.id), "the --id folder already exists, please assign a new one."
    os.mkdir(args.id)
    with open(args.file, 'r') as f:
        info = json.load(f)
    for i, sp in info.items():
        compare.load_image(sp["image1"], sp["image2"])
        im1, im2 = compare.query(sp["x"], sp["y"], sp["anchor"])
        result = concat_h(im1, im2)
        result.save(os.path.join(args.id, "{}.png".format(i)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='compare features of two images')
    parser.add_argument('--mode', type=str, required=True, choices=["interactive", "random_generate", "from_file"], help='''
        interactive: in this mode you can interactively compare features.
        random_generate: in this mode you need to provide a folder with images, and a set of parameters, the program will randomly sample some image pairs and points to compare. The images colored by heatmap will be saved as well as the sampled image path and points.
        from_file: you need to provide a configuration file generated by 'random_generate' mode.
        ''')
    args, _ = parser.parse_known_args()
    parser.add_argument('--weight', type=str, required=True, help='weight path')
    parser.add_argument('--cfg', type=str, default="./configs/wiki_resnet50.yaml", help='cfg file')
    parser.add_argument('--dataset', type=str, default="wikiscenes", help='dataset')
    if args.mode == 'interactive':
        parser.add_argument('--image1', type=str, required=True, help='image1')
        parser.add_argument('--image2', type=str, required=True, help='image2')
        parser.add_argument('--weight2', type=str, required=False, help='another set of parameters to compare with')
    elif args.mode == 'random_generate':
        parser.add_argument('--image_folder', type=str,required=True, help="image folder. the program will find all images recursively")
        parser.add_argument('--sample_size', type=int, required=True, help="number of pairs to sample")
        parser.add_argument('--id', type=str, default="export", help="the output folder name.")
    else:
        parser.add_argument('--file', type=str, required=True, help="a list to sample")
        parser.add_argument('--id', type=str, default='export', help="the output folder name.")
    args = parser.parse_args()
    cfg_from_file(args.cfg)

    compare = Compare(cfg, args)
    compare.load_model(args.weight)

    if args.mode == 'interactive':
        compare.load_image(args.image1, args.image2)
        if args.weight2 != None:
            assert os.path.exists(args.weight2)

        gui(compare)
    elif args.mode == 'random_generate':
        random_generate(compare)
    else:
        from_file(compare)

