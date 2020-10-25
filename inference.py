#-*- coding:utf-8 -*-
from absl import app, flags, logging
from absl.flags import FLAGS
from detectron2.structures import BoxMode
import random
import cv2
import json
import os, glob, time
import csv
from natsort import natsorted

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from tqdm.auto import tqdm

import numpy as np
import torch
import torchvision
import detectron2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

#TODO needs to change 0
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
flags.DEFINE_string('i', '', 'path to input image folder')

def get_dataset_dicts(config_dir):

    df = pd.read_csv(config_dir)

    class_mapper = np.load('./dataset/trainval_classes.npy', allow_pickle=True)
    dataset_dicts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
#        if idx == 1000:
#            break
        record = {}
        filename = row.image_path
        height, width = 480, 640

        #Parse data filename
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        x1 = row.bbox_x1
        y1 = row.bbox_y1
        x2 = row.bbox_x2
        y2 = row.bbox_y2

        poly = [373.5, 618.5, 358.5, 608.5, 343.5, 607.5, 326.5, 609.5, 314.5, 622.5, 298.5, 639.5, 286.5, 653.5, 295.5, 673.5, 296.5, 678.5, 433.5, 678.5, 427.5, 667.5, 413.5, 653.5, 391.5, 632.5, 373.5, 618.5]
        obj = {
            "bbox": [x1, y1, x2, y2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": np.where(class_mapper == row['class'])[0][0],
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def main(_argv):
    class2idx = dict()
    with open('./dataset/class_numbering.csv', "rt", encoding='euc-kr', ) as f: 
        reader = csv.reader(f, delimiter=',') 
        for i, line in enumerate(reader):
            if i == 0:
                continue
            class2idx[line[0]] = line[2]

    thing_classes = np.load('./dataset/trainval_classes.npy', allow_pickle=True)

#    a = set(class2idx.keys()) - set(thing_classes)
#    print("miss class:", a)
#    b = set(thing_classes) - set(class2idx.keys())
#    print("Ours:", b)
#    exit()

#    for d in ["test"]:
#        DatasetCatalog.register(
#            "aihub_" + d, lambda d=d: get_dataset_dicts(f"dataset/filtered_{d}_dataframe.csv"))
#        MetadataCatalog.get("aihub_" + d).set(thing_classes=thing_classes)

#aihub_metadata = MetadataCatalog.get("aihub_test")
#dataset_dicts = get_dataset_dicts(f"dataset/test_dataframe.csv")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("aihub_test",)
    cfg.DATASETS.TEST = ("aihub_test")
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 3000000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1031

    # RetinaNet
    cfg.MODEL.RETINANET.NUM_CLASSES = 1031
    cfg.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.RETINANET.NUM_CONVS = 4
    cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
    cfg.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.RETINANET.PRIOR_PROB = 0.01
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
    cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # Loss parameters
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
    # Options are: "smooth_l1", "giou"
    cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"

    # One of BN, SyncBN, FrozenBN, GN
    # Only supports GN until unshared norm is implemented
    cfg.MODEL.RETINANET.NORM = ""

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set a custom testing threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.DEVICE='cuda'

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,'model_1864999.pth')

    # Set model
    predictor = DefaultPredictor(cfg)

    # Input folder
    imgfolder = FLAGS.i

    # Gather images as data
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(imgfolder):
        for file in f:
            if file.endswith(".png") or file.endswith(".jpg") \
                or file.endswith(".jpeg"):
                files.append(os.path.join(r, file))
    files.sort()
    
    # Open output file
    f = open('output.csv', 'w')
    header = "filename, classID, Confidence, bbox_left_top_x, bbox_left_top_y, bbox_right_bottom_x, bbox_right_bottom_y\n"
    f.write(header)

    # Inference
    for item in files:
        im = cv2.imread(item)
        h, w,_ = im.shape
        # TODO Save scaling information
        ratio = 1/max(h/480 , w/640)
        # TODO Resize image
        im = cv2.resize(im, (0,0), ratio, ratio)
        h_prime, w_prime, _ = im.shape 

        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Parse outputs
#print('PRED:', str(thing_classes[outputs["instances"].to("cpu").pred_classes]).encode('utf-8'))
        filename = os.path.basename(item)
        #Set emtpy initial value
        classID = ''
        confidence = ''
        bbox = ['','','','']

        scores = outputs["instances"].to("cpu").scores.tolist()
        print(item, scores)
        if len(scores) > 0:
            idx = scores.index(max(scores))
            candidate =  thing_classes[outputs["instances"].to("cpu").pred_classes]

            if isinstance(candidate, str):
                class_name = candidate
            else:
                class_name = thing_classes[outputs["instances"].to("cpu").pred_classes][idx]

            confidence = scores[idx]
            # Get first bbox item
            bbox = outputs["instances"].to("cpu").pred_boxes[idx].tensor.tolist()[0]
            # TODO restore bbox

            print('PRED:', class_name)
            print('Score: ', confidence)
            print("filename:", filename)

            # Change class to official index
            if class_name in class2idx.keys(): 
                classID = class2idx[class_name]
            print("official_class", classID)
    

        # Make csv (filename, class_ID,, Confidence, bbox_left_top_x, bbox_left_top_y, bbox_right_bottom_x, bbox_right_bottom_y)
        result = [filename, classID, confidence] + bbox
        row = ','.join(str(x) for x in result) + '\n'
        print(row)
        f.write(row)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
       
