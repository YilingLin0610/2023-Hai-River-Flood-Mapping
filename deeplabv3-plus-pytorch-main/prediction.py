"""
Function for batch predicting on test images.
Author: Modified from  https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Mobile.
"""

import time
import cv2
import numpy as np
from PIL import Image
import os
from deeplab import DeeplabV3
from tqdm import *



def prediction(FilePath,OutPath,model_path):
    """
    Function for batch predicting on test images.
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        FilePath: The path that stores the JPG testing images.
        OutPath: The path that stores the prediction PNG result.
        model_path: The file path of the trained checkpoint.
    @return:
        None
    """
    # Initialize the Deeplabv3+ model.
    deeplab = DeeplabV3(model_path=model_path)
    # ----------------------------------------------------------------------------------------------------------#
    #   'predict'           Images batch prediction.
    # ----------------------------------------------------------------------------------------------------------#
    # -------------------------------------------------------------------------#
    #   count               Specifies whether to perform pixel counting (i.e., area) and proportion calculation for the targets.
    #   name_classes        Specifies the classes, which should match those in json_to_dataset, and is used for printing the categories and their counts.
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["RTS", "non_RTS"]
    # name_classes    = ["background","cat","dog"]

    filelist = [x for x in os.listdir(FilePath) if x.endswith(".jpg")]
    for name in tqdm(filelist):
        file_path_name = FilePath + "/" + name  # 坿猟周揃抄
        image = Image.open(file_path_name)
        r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
        out_name = OutPath + name[0:-4] + ".png"
        r_image.save(out_name)

