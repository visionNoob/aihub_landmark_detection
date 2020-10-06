import sys 
import os 
sys.path.append(os.pardir)
sys.path.insert(0, '..')
import os.path as osp 
import math 

import pandas as pd 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm 
import yaml 
from PIL import Image 

trainDir_list = ['train', 'train2']
    
for trainDir in tqdm(trainDir_list):  
    motherDir = osp.abspath('../..')
    print(motherDir)
    dataDir = osp.join(motherDir, 'data', 'refined', trainDir_list[1])
    dataList = os.listdir(dataDir)



    landmark = [] 
    xmin, ymin, xmax, ymax = [], [], [], []
    fileName = [] 

    for path in tqdm(dataList):
        target = osp.join(dataDir, path)
        yamls = glob(osp.join(target, '*.yaml'))
    
        for item in yamls: 
            item_yaml = item 
        
            with open(item_yaml) as file: 
                ds = yaml.load(file)
                
            fileName.append(item_yaml.split("/")[-1].split(".")[0])
            landmark.append(ds['regions'][0]['class'])
            
            x1, y1, x2, y2 = ds['bbox']['x1'], ds['bbox']['y1'], ds['bbox']['x2'], ds['bbox']['y2']
            xmin.append(x1)
            ymin.append(y1)
            xmax.append(x2)
            ymax.append(y2)
        

df = pd.DataFrame({ 'name'  : fileName,
                    'width' : 640,
                    'height': 480,
                    'class' : landmark,
                    'xmin' : xmin,
                    'ymin' : ymin,
                    'xmax' : xmax,
                    'ymax' : ymax})

df.to_csv('Anchor.csv', sep=',', na_rep='NaN')