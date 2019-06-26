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
from PIL import Image


class FeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(FeatureExtractor, self).__init__()
        self.feat_size = opt.feat_size
        self.cnn = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.cnn(x)


def main(args):
    preprocess = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = FeatureExtractor(args).cuda()
    net.eval()
    dir_feat = args.output_dir + '_feat'
    if not os.path.exists(dir_feat):
        os.mkdir(dir_feat)
    dataset_coco = json.load(open(args.input_json, 'r'))
    images = dataset_coco['images']

    with torch.no_grad():
        for img_info in tqdm(images):
            if os.path.exists(os.path.join(dir_feat, str(img_info['cocoid']) + '.npy')):
                continue
            img = np.array(Image.open(os.path.join(args.image_root, img_info['filepath'], img_info['filename'])))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = preprocess(img).cuda().unsqueeze(0)
            feature = net(img).squeeze()
            # print(feature.shape)
            np.save(os.path.join(dir_feat, str(img_info['cocoid']) + '.npy'), feature.cpu().float().numpy())
            # break

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image feature prepare')
    parser.add_argument('--input_json', default='dataset_coco', help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    parser.add_argument('--image_root', help='img root directory')
    parser.add_argument('--feat_size', type=int, default=512, help='img feature size')
    args = parser.parse_args()
    main(args)
