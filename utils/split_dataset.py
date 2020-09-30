import os
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from imgaug import BoundingBoxesOnImage, BoundingBox
import imgaug
from imgaug import augmenters as iaa
import argparse
import random
random.seed(42)


def main(args):
    dataset_name = args.dataset
    dataset = os.path.join(
        '/home/ubuntu/workspace_aihub/data/raw/', dataset_name)

    class_wise = glob(os.path.join(dataset, '*'))
    for per_class_dir in tqdm(class_wise):
        datas = glob(os.path.join(per_class_dir, '*'))
        for file in tqdm(datas):
            if file.split('.')[-1] != 'json':
                continue

            base_name = file.strip('.json')
            image_file = None
            if os.path.isfile(base_name+'.jpg'):
                image_file = base_name+'.jpg'

            if os.path.isfile(base_name+'.JPG'):
                image_file = base_name+'.JPG'

            if image_file is None:
                print('no image')
                continue
            conf = OmegaConf.create()
            image = cv2.imread(image_file)

            if len(image.shape) != 3:
                print('this image dose not have 3 chennels')

            with open(file) as json_file:
                json_data = json.load(json_file)
                if len(json_data['regions']) != 1:
                    print('this regin has more than 1 bbx')
                    print(json_data['regions'])
                conf['regions'] = json_data['regions']

                x1 = json_data['regions'][0]['boxcorners'][0]
                y1 = json_data['regions'][0]['boxcorners'][1]
                x2 = json_data['regions'][0]['boxcorners'][2]
                y2 = json_data['regions'][0]['boxcorners'][3]

                bbs = BoundingBoxesOnImage([
                    BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                ], shape=image.shape)

                if image.shape[0] > image.shape[1]:
                    seq = iaa.Sequential([
                        iaa.Resize(
                            {"height": 480, "width": "keep-aspect-ratio"}),
                        iaa.CenterPadToFixedSize(height=480, width=640, )
                    ])
                else:
                    seq = iaa.Sequential([
                        iaa.Resize(
                            {"width": 640, "height": "keep-aspect-ratio"}),
                        iaa.CenterPadToFixedSize(height=480, width=640, )
                    ])

                #image = np.transpose(image, (1,0,2))
                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
                x1 = int(bbs_aug[0].x1)
                y1 = int(bbs_aug[0].y1)
                x2 = int(bbs_aug[0].x2)
                y2 = int(bbs_aug[0].y2)
                conf['bbox'] = {}
                conf['bbox']['x1'] = x1
                conf['bbox']['y1'] = y1
                conf['bbox']['x2'] = x2
                conf['bbox']['y2'] = y2

                rand_number = random.randint(0, 9)
                if rand_number == 0:
                    mode = 'test_dataset'
                elif rand_number == 1:
                    mode = 'validation_dataset'
                else:
                    mode = 'train_dataset'

                dst_path = os.path.join(
                    '/home/ubuntu/workspace_aihub/data/refined', mode)

                OmegaConf.save(conf, os.path.join(
                    dst_path, f'{os.path.basename(base_name)}.yaml'))
                cv2.imwrite(os.path.join(
                    dst_path, f'{os.path.basename(base_name)}.jpg'), image_aug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I want girfriend')
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    main(args)
