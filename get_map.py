#-*- coding:utf-8 -*-
from detectron2.structures import BoxMode
import random
import cv2
import json
import os, glob, time
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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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
import csv
class2idx = dict()
with open('./dataset/class_numbering.csv', "rt", encoding='euc-kr', ) as f: 
    reader = csv.reader(f, delimiter=',') 
    for i, line in enumerate(reader):
        if i == 0:
            continue
        class2idx[line[0]] = line[2]

thing_classes = list(np.load('./dataset/trainval_classes.npy', allow_pickle=True))

#filtered_class = []
#for item in thing_classes:
#    if item in class2idx.keys():
#        filtered_class.append(class2idx[item])
#     
#print(len(thing_classes), '->', len(filtered_class))

#thing_classes = list(np.load('./dataset/thing_classes.npy', allow_pickle=True))
for d in ["test"]:
    DatasetCatalog.register(
        "aihub_" + d, lambda d=d: get_dataset_dicts(f"dataset/filtered_{d}_dataframe.csv"))
    MetadataCatalog.get("aihub_" + d).set(thing_classes=thing_classes)

#aihub_metadata = MetadataCatalog.get("aihub_test")
#dataset_dicts = get_dataset_dicts(f"dataset/test_dataframe.csv")
#thing_classes = np.load('./dataset/thing_classes.npy', allow_pickle=True)

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
#cfg.MODEL.DEVICE='cpu'
cfg.MODEL.DEVICE='cuda'

# Load weight file
weight_list = []
for item in glob.glob("./output/model_*.pth"):
    weight_list.append(os.path.basename(item))

# Sort weight
weight_list = natsorted(weight_list)[:-1]

maxAP = 0.0
maxName = ""
iter_list = []
AP_list = []

for idx, weight in enumerate(weight_list):
    start = time.time()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weight)
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("aihub_test", cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, "aihub_test")

    infer = inference_on_dataset(predictor.model, test_loader, evaluator)
    AP_list.append(infer['bbox']['AP'])
    iter_list.append(str(int(weight.split('.')[0].split('_')[1])))

    if maxAP < infer['bbox']['AP']:
        maxAP = infer['bbox']['AP']
        maxName = weight
        print("max changed:", maxName,"(AP:",maxAP, ")")
    print("time:", time.time() - start)
    print("Name:", weight,"(AP:",infer['bbox']['AP'], ")")

print("Final result!")
print("name:", maxName)
print("AP:", maxAP)

print(iter_list)
print(AP_list)

#Draw graph
plt.plot(iter_list, AP_list)
plt.savefig("AP_result.png")

#AP result
#name: model_1864999.pth
#AP: 49.9687107246589
#['54999', '104999', '204999', '304999', '404999', '504999', '604999', '704999', '804999', '904999', '1004999', '1034999', '1074999', '1104999', '1134999', '1174999', '1204999', '1234999', '1274999', '1304999', '1309999', '1334999', '1374999', '1404999', '1434999', '1474999', '1504999', '1534999', '1574999', '1604999', '1634999', '1674999', '1704999', '1734999', '1774999', '1804999', '1834999', '1859999', '1864999']
#[12.391292278607855, 27.625722543908765, 41.579621197314225, 46.79081346682297, 47.17699301962195, 47.51888353939951, 47.66416550460297, 47.88091506132384, 48.186239335996426, 48.320789193493766, 48.55994433909421, 48.634880874841876, 48.72178725101238, 48.750924337287024, 48.77986402307666, 48.85604310046764, 48.99189491842048, 49.02099016605124, 49.127875404780916, 49.08840000040094, 49.03363385494002, 49.12842051356878, 49.23293809618266, 49.238166407284005, 49.327687941785356, 49.36039265472566, 49.39661689762491, 49.42029318697962, 49.58394139739502, 49.63150943404202, 49.61112164598241, 49.66979711641537, 49.74210549174124, 49.68322488645261, 49.85618847960457, 49.81314778068761, 49.87318105831162, 49.91290565097373, 49.9687107246589]

#from absl import app, flags, logging
#from absl.flags import FLAGS
#flags.DEFINE_string('i', '', 'path to input image folder')

# Input folder
# imgfolder = FLAGS.i

# Gather images as data
#    files = []
#    # r=root, d=directories, f = files
#    for r, d, f in os.walk(imgfolder):
#        for file in f:
#            if file.endswith(".png") or file.endswith(".jpg") \
#                or file.endswith(".jpeg"):
#                files.append(os.path.join(r, file))
#    files.sort()

# Set weights
#    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weight)
#    predictor = DefaultPredictor(cfg)

# Inference

#for item in files:
#    im = cv2.imread(item)
#    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

# Parse outputs

# Make csv (filename, class_ID,, Confidence, bbox_left_top_x, bbox_left_top_y, bbox_right_bottom_x, bbox_right_bottom_y)

#    print('PRED:', str(thing_classes[outputs["instances"].to("cpu").pred_classes]).encode('utf-8'))
#    print('Score: ', outputs["instances"].to("cpu").scores)
#    print('LABEL:', str(thing_classes[d['annotations'][0]['category_id']]).encode('utf-8'))
#

    

