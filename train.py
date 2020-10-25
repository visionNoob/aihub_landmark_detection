from detectron2.structures import BoxMode
import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import pandas as pd
setup_logger()


def get_dataset_dicts(config_dir):

    df = pd.read_csv(config_dir)

    class_mapper = np.load('./dataset/trainval_classes.npy', allow_pickle=True)
    dataset_dicts = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        record = {}
        #config = OmegaConf.load(row.yaml_path)
        filename = row.image_path
        height, width = 480, 640

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        x1 = row.bbox_x1
        y1 = row.bbox_y1
        x2 = row.bbox_x2
        y2 = row.bbox_y2

        obj = {
            "bbox": [x1, y1, x2, y2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": np.where(class_mapper == row['class'])[0][0],
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def main():
    # for d in ["train", "val"]:
    thing_classes = np.load('./dataset/thing_classes.npy', allow_pickle=True)
    for d in ["trainval"]:
        DatasetCatalog.register(
            "aihub_" + d, lambda d=d: get_dataset_dicts(f"dataset/{d}_dataframe.csv"))
        MetadataCatalog.get(
            "aihub_" + d).set(thing_classes=list(thing_classes))

    aihub_metadata = MetadataCatalog.get("aihub_trainval")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("aihub_trainval",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 12
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
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    # One of BN, SyncBN, FrozenBN, GN
    # Only supports GN until unshared norm is implemented
    cfg.MODEL.RETINANET.NORM = ""

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
