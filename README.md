# Pig Segmentation Scripts for AIFARM


## Step 1: Build Dataset

I have re-written the annotation files, and re-organize the images. The original annotation json files have been re-written in a coco-format for the image segmentation task. The original images are randomly splitted into train set and test set.

There are two options to load the data.

### Option 1: Directly download the re-organized data from google drive (totalling 7.0 GB)
  Directly download the ```pig_life/test``` and ```pig_life/train``` folders and ```pig_life/split.json``` from this [shared data folder](https://drive.google.com/drive/folders/1rkWhmL3zh9m0fVEFw9sXZuyU0eSm0g4X?usp=sharing), and place them under the ```pig_life``` folder.
### Option 2: Re-organize the original files using my python script
Move the original images and annotations under the ```pig_life``` folder, and rename them as ```1040```, ```1050```, ```AVAT1040```, ```AVAT1050```, which are exactly the same as the four folders under ```pig_life``` in [shared data folder](https://drive.google.com/drive/folders/1rkWhmL3zh9m0fVEFw9sXZuyU0eSm0g4X?usp=sharing).

Run ```data.py```, which can generate two anno json files under ```pig_life/train``` and ```pig_life/test``` and copy images to these two folders.


## Step 2: Install Detectron2
Follow the official link https://detectron2.readthedocs.io/en/latest/tutorials/install.html to install. 

### My environment
I run 
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
to install the detectron2.

- python=3.8
- pytorch=1.0+cu111
- torchvision=0.11.2

## Step 3: Train the segmentation model
Run ```Detectron2_Pig_Segmentation.ipynb``` step by step.

## Step 4: Inference on any image
Check the **Testing** section in the above code. The last section **Load model and inference on your image** (a copy from **Testing** section) is a template for your to load video frames (only a for loop is need).