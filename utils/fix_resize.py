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
import pandas as pd


def get_transform(image):
    seq = []

    if image.shape[0] >= image.shape[1]:

        seq.append(iaa.Resize({"height": 480, "width": "keep-aspect-ratio"}))

        if image.shape[1] / image.shape[1] * 480. > 640.:

            seq.append(iaa.Resize(
                {"width": 640, "height": "keep-aspect-ratio"}))

    if image.shape[0] < image.shape[1]:

        seq.append(iaa.Resize({"width": 640, "height": "keep-aspect-ratio"}))

        if image.shape[0] / image.shape[1] * 640. > 480.:

            seq.append(iaa.Resize(
                {"height": 480, "width": "keep-aspect-ratio"}))
    seq.append(iaa.CenterPadToFixedSize(height=480, width=640, ))
    return iaa.Sequential(seq)


def main(args):
    df = pd.DataFrame()
    df_count = 0
    fold = args.fold

    dataset_name = args.dataset
    dataset = os.path.join(
        '/home/ubuntu/workspace_aihub/data/refined/', dataset_name)
    class_wise = glob(os.path.join(dataset, '*'))

    classes_per_batch = 65

    st = fold * classes_per_batch
    en = fold * classes_per_batch + classes_per_batch

    if fold != 15:
        class_wise = class_wise[st:en]
    else:
        class_wise = class_wise[st:]

    print('# classes:', len(class_wise))

    dst_root = os.path.join(
        '/home/ubuntu/workspace_aihub/data/refined/v3/', dataset_name)
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)

    dst_root = os.path.join(dst_root, str(fold))
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)

    for per_class_dir in tqdm(class_wise):
        datas = glob(os.path.join(per_class_dir, '*'))
        for file in tqdm(datas):
            if file.split('.')[-1] != 'yaml':
                continue
            base_name = file.strip('.yaml')

            image_file = base_name+'.jpg'
            # print(image_file)
            image = cv2.imread(image_file)
            config = OmegaConf.load(file)
            if image is None:
                print('no image!')
                continue

            x1 = config.bbox.x1
            y1 = config.bbox.y1
            x2 = config.bbox.x2
            y2 = config.bbox.y2

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            ], shape=image.shape)

            transform = get_transform(image)

            image_aug, bbs_aug = transform(image=image, bounding_boxes=bbs)

            x1 = int(bbs_aug[0].x1)
            y1 = int(bbs_aug[0].y1)
            x2 = int(bbs_aug[0].x2)
            y2 = int(bbs_aug[0].y2)
            config['bbox'] = {}
            config['bbox']['x1'] = x1
            config['bbox']['y1'] = y1
            config['bbox']['x2'] = x2
            config['bbox']['y2'] = y2

            label = config['regions'][0]['tags'][1].split(':')[-1]
            dst_path = os.path.join(dst_root, label)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            OmegaConf.save(config, os.path.join(
                dst_path, f'{os.path.basename(base_name)}.yaml'))
            cv2.imwrite(os.path.join(
                dst_path, f'{os.path.basename(base_name)}.jpg'), image_aug)

            df.loc[df_count, 'bbox_x1'] = x1
            df.loc[df_count, 'bbox_y1'] = y1
            df.loc[df_count, 'bbox_x2'] = x2
            df.loc[df_count, 'bbox_y2'] = y2
            df.loc[df_count, 'image_width'] = image_aug.shape[1]
            df.loc[df_count, 'image_height'] = image_aug.shape[0]
            df.loc[df_count, 'class'] = label
            df.loc[df_count, 'yaml_path'] = os.path.join(
                dst_path, f'{os.path.basename(base_name)}.yaml')
            df.loc[df_count, 'image_path'] = os.path.join(
                dst_path, f'{os.path.basename(base_name)}.jpg')

            df_count += 1

    df.to_csv(f'{dataset_name}_dataframe_fold{fold}.csv')
# pandas에 저장할 것들
# 1. Bbox_x1
# 2. Bbox_y1
# 3. Bbox_x2
# 4. Bbox_y2
# 5. Image_width
# 6. Image_height
# 5. Class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I want girfriend')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()

    main(args)
