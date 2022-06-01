import json
import glob
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import os

def areacal(segmentation):
    segmentation = [(seg['x'],seg['y']) for seg in segmentation]
    polygon = Polygon(segmentation)
    return polygon.area

def bboxcal(segmentation):
    x = min(segmentation[::2])
    y = min(segmentation[1::2])
    x_ = max(segmentation[::2])
    y_ = max(segmentation[1::2])
    return [round(x), round(y), round(x_-x), round(y_-y)]

data_root = './pig_life/'
anno_1040_path = data_root + 'AVAT1040/'
anno_1050_path = data_root + 'AVAT1050/'
image_1040_path = data_root + '1040/'
image_1050_path = data_root + '1050/'

files1040 = glob.glob(anno_1040_path + '*.json')
files1050 = glob.glob(anno_1050_path + '*.json')
imgpaths = glob.glob(image_1040_path + '*.png') + \
           glob.glob(image_1050_path + '*.png')
           
if os.path.isdir(data_root + 'train'):
    shutil.rmtree(data_root + 'train')
os.makedirs(data_root + 'train')
if os.path.isdir(data_root + 'test'):
    shutil.rmtree(data_root + 'test')
os.makedirs(data_root + 'test')

output = {
    'train':{'images':[], 'categories':[{"supercategory": "pig", "id": 1, "name": "pig"}], 'annotations':[]},
    'test':{'images':[], 'categories':[{"supercategory": "pig", "id": 1, "name": "pig"}], 'annotations':[]}
}

img_train, img_test = train_test_split([i.split('/')[-1] for i in imgpaths],test_size=0.2)

splitlist = {v:'train' for v in img_train}
splitlist.update({v:'test' for v in img_test})
img2id = {v:i+1 for i,v in enumerate(img_train)}
img2id.update({v:i+1 for i,v in enumerate(img_test)})

pig_id = {'train': 1, 'test': 1}

for file in files1040+files1050:
    with open(file, 'r') as f:
        data = json.load(f)
    
    horizontal_res = data['vid_metadata']['horizontal_res']
    vertical_res = data['vid_metadata']['vertical_res']
    
    for img in data['annotations']:
        
        for pig in img:
            img_name = pig['fileName']
            img_id = img2id[img_name]
            split = 'train' if img_name in img_train else 'test'
            if len(output[split]['images']) == 0 or img_name not in [i['file_name'] for i in output[split]['images']]:
                img_path0 = data_root + img_name[:4] + '/' + img_name
                img_path1 = data_root + split + '/' + img_name
                output[split]['images'].append({'height': vertical_res, 'width': horizontal_res, 'id': img_id, 'file_name': img_path1})
                shutil.copy(img_path0, img_path1)
            segmentation = pig['points']
            area = areacal(segmentation)
            segmentation = [i for seg in segmentation for i in [seg['x'],seg['y']]]
            bbox = bboxcal(segmentation)
            output[split]['annotations'].append({"iscrowd": 0, "image_id": img_id, "bbox": bbox,
                                          "segmentation": [segmentation], "category_id": 1, "id": pig_id[split], "area": area})
            pig_id[split] += 1
            

with open(data_root + 'train/pig_coco.json', 'w') as f:
    json.dump(output['train'], f)
with open(data_root + 'test/pig_coco.json', 'w') as f:
    json.dump(output['test'], f)
with open(data_root + 'split.json', 'w') as f:
    json.dump(splitlist, f)
    
