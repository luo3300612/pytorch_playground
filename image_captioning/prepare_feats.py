import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.functional as F
from torchvision import transforms as T
import argparse
import os
import numpy as np
import json
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self, opt):
        self.feat_size = opt.feat_size
        self.cnn = resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, self.feat_size)

    def forward(self, x):
        return self.cnn(x)


def main(args):
    preprocess = T.Compose([T.Normalize([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])])
    net = FeatureExtractor(args)
    net.eval()
    dir_feat = args.output_dir + '_feat'

    dataset_coco = json.load(open(args.input_json, 'r'))
    images = dataset_coco['images']

    with torch.no_grad():
        for img_info in tqdm(images):
            img = np.load(os.path.join(args.image_root, img_info['filepath'], img_info['filename']))
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            feature = net(img).squeeze(0)
            np.save(os.path.join(dir_feat, img_info['coco_id'], '.npy'), feature.cpu().float().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image feature prepare')
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    parser.add_argument('--image_root', help='img root directory')
    parser.add_argument('--feat_size', type=int, default=512, help='img feature size')
    args = parser.parse_args()
    main(args)
