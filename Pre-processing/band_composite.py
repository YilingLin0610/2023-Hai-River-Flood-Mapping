
from utils import read_raster_as_array,writeTiff

import numpy as np


def band_composite(in_file_VV,in_file_VH,mosaic_path):
    '''
    Composite three bands into one TIFF file: VV VH VH/VV
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        in_file_VV: The filepath of VV file
        in_file_VH: The filepath of VH file
        mosaic_path: The filepath of the composite image
    @return:
        None
    '''
    # Read the VV and VH SAR images into arrays.
    array_VV, proj, geotrans = read_raster_as_array(in_file_VV)
    array_VH, _, _ = read_raster_as_array(in_file_VH)
    # Create an array with the same shape as the VV image to store the new third band (VH/VV).
    array_b = np.zeros_like(array_VV)
    # Calculate the third band.
    array_b[array_VV != 0] = array_VH[array_VV != 0] / array_VV[array_VV != 0]
    array_b[np.isnan(array_b)] = 0

    # Set the extreme values in the third band to 0.
    condition = (array_b > 70) | (array_b < -70)

    array_b[condition]=0

    # Stretch the three bands to 0-255
    array_VV[array_VV != 0] = 255 * (array_VV[array_VV != 0] - np.min(array_VV)) / (np.max(array_VV) - np.min(array_VV))
    array_VH[array_VH != 0] = 255 * (array_VH[array_VH != 0] - np.min(array_VH)) / (np.max(array_VH) - np.min(array_VH))
    array_b[array_VH != 0] = 255 * (array_b[array_VH != 0] - np.min(array_b)) / (np.max(array_b) - np.min(array_b))
    array_VH=array_VH.astype(int)
    array_VV = array_VV.astype(int)


    out_data = np.array([array_VV, array_VH, array_b])
    out_data = out_data[:, :, :, 0]

    writeTiff(out_data, geotrans, proj, mosaic_path)




if __name__=="__main__":

    # For Sentinel-1
    in_file_VV=r"C:\Users\xiehu\Desktop\Beijing_flood\mosaiced_images\0910_VV.tif"
    in_file_VH=r"C:\Users\xiehu\Desktop\Beijing_flood\mosaiced_images\0910_VH.tif"
    array_VV,proj,geotrans=read_raster_as_array(in_file_VV)
    array_VH, _, _ = read_raster_as_array(in_file_VH)

    array_b=np.zeros_like(array_VV)
    array_b[array_VV!=0]=array_VH[array_VV!=0]/array_VV[array_VV!=0]
    print(np.max(array_b),np.min(array_b))
    array_VV[array_VV!=0]=255*(array_VV[array_VV!=0]-np.min(array_VV))/(np.max(array_VV)-np.min(array_VV))
    array_VH[array_VH!=0] = 255*(array_VH[array_VH!=0] - np.min(array_VH)) / (np.max(array_VH) - np.min(array_VH))
    array_b[array_b!=0] = 255*(array_b[array_b!=0] - np.min(array_b)) / (np.max(array_b) - np.min(array_b))
    out_data=np.array([array_VV,array_VH,array_b])
    out_data = out_data[:, :, :, 0]
    path=r"E:\Beijing_flood\mosaic_rgb\0910.tif"
    writeTiff(out_data, geotrans, proj, path)

    # for Gaofen-images
    # array_VV, proj, geotrans = transform_to_db(in_file_VV)
    # array_VH, _, _ = transform_to_db(in_file_VH)
    # array_b = np.zeros_like(array_VH)
    # array_b[array_VV != 0] = array_VH[array_VV != 0] / array_VV[array_VV != 0]
    # array_VV[array_VV != 0] = 255 * (array_VV[array_VV != 0] - np.min(array_VV)) / (np.max(array_VV) - np.min(array_VV))
    # array_VH[array_VH != 0] = 255 * (array_VH[array_VH != 0] - np.min(array_VH)) / (np.max(array_VH) - np.min(array_VH))
    #
    # array_b[array_b != 0] = 255 * (array_b[array_b != 0] - np.min(array_b)) / (np.max(array_b) - np.min(array_b))
    # out_data = np.array([array_VV, array_VH, array_b])
    # out_data=out_data[:,:,:,0]
    # images_matched = np.transpose(out_data, (2, 0, 1))
    # path = r"E:\Beijing_flood\mosaic_rgb\gaofen_0801_3.tif"
    # writeTiff(out_data, geotrans, proj, path)


    #print(np.max(array_VH))