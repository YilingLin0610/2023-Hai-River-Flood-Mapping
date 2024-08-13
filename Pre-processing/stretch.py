#coding=gbk
import numpy as np
from osgeo import gdal
from utils import writeTiff,readTif

def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    '''
    A linear stretch to an array.
    ——————————————————————————
    @param：
        image: An array that store the data of the raster TIFF file.
        truncated_value: The threshold to truncate the pixel values to either
                         below or above a certain percentage

    @return:
        image_stretch: An array that store the data the stretched raster TIFF file.
    '''


    def gray_process(gray):
        gray_2=gray.flatten()
        gray_3=gray_2[gray_2!=0]
        truncated_down = np.percentile(gray_3, truncated_value)
        truncated_up = np.percentile(gray_3, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out

        gray[gray > max_out] = max_out
        if (max_out <= 255):
            gray = np.uint8(gray)
        elif (max_out <= 65535):
            gray = np.uint16(gray)
        return gray

    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    else:
        image_stretch = gray_process(image)
    return image_stretch

def all(image_in,image_out,percent):
    '''
    A linear stretch to a raster TIFF file.
    ——————————————————————————
    @param：
        image_in: The file path of the raster TIFF file.
        image_out: The file path of the stretched raster TIFF file.
        percent: The threshold to truncate the pixel values to either
                         below or above a certain percentage

    @return:
        None
    '''
    width, height, bands, data, geotrans, proj = readTif(image_in)
    data_stretch = truncated_linear_stretch(data, percent)
    writeTiff(data_stretch, geotrans, proj, image_out)




if __name__ == "__main__":
    #images_source_name=r"E:\Beijing_flood\mosaic_rgb\0601_tianjin.tif"
    image_in=r"E:\Beijing_flood\mosaic_rgb\0910_matched.tif"
    image_out=r"E:\Beijing_flood\mosaic_rgb\0910_matched_stretched.tif"
    all(image_in,image_out,0.5)


