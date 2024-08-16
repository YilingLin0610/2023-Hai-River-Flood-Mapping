"""
Split the dataset into training, validation, and testing groups.
Author: Modified from https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Mobile.
"""



import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

trainval_percent = 1
train_percent = 0.8
VOCdevkit_path = 'RTS_datas'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = r"/media/dell/disk/Yiling/Beijing_flood/training_data/0712_Beijing_st/label_data_pngs_aug"
    saveBasePath = os.path.join(VOCdevkit_path, 'RTS_datasets/ImageSets/Segmentation')

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("Check the format of the dataset")
    classes_nums = np.zeros([256], np.int_)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not detected. Please check if the file exists in the specified path and ensure the file extension is .png." % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(
                "The shape of the label image %s is %s, which is not a grayscale or 8-bit color image. Please check the dataset format carefully." % (
                name, str(np.shape(png))))
            print(
                "The label image should be a grayscale or 8-bit color image, where the value of each pixel indicates the class to which the pixel belongs." % (
                name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("Print the values and quantities of the pixels.")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Detected that the label contains only pixel values of 0 and 255, indicating incorrect data format.")
        print(
            "For binary classification, the label should be set with pixel values of 0 for background and 1 for the target.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print(
            "Detected that the label contains only background pixels, indicating incorrect data format. Please check the dataset format carefully.")

    print("Images in JPEGImages should be .jpg files, and images in SegmentationClass should be .png files.")
    print("If the format is incorrect, refer to:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")

