"""
Historgram matching
"""
import numpy as np
from utils import read_raster_as_array,writeTiff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = ['Helvetica']
mpl.rcParams['font.size'] = 40


def cdf_compute(image):
    '''
    Compute the CDF of TIFF image
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        image: An array that stores the data of the TIFF image.
    @return:
        cdf: The normalized cdf
    '''
    hist, bins = np.histogram(image,bins=256,range=[0,256])
    hist[0]=0
    cdf=hist.cumsum()/hist.sum()
    return cdf

def pdf_compute(image):
    '''
    Compute the PDF of TIFF image
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        image: An array that stores the  data of TIFF image.
    @return:
        hist: The PDF
    '''
    hist, bins = np.histogram(image,bins=256,range=[0,256])
    hist[0]=0

    return hist

def histogram_matching(source_image, target_image):
    '''
    Histogram matching
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        source_image: An array that stores the  data of source TIFF image.
        target_image: An array that stores the  data of target TIFF image.
    @return:
        matched_image: An array that stores the matched source TIFF file
    '''

    # calculate the CDF of source image and target image, respectively.
    source_cdf=cdf_compute(source_image)
    target_cdf=cdf_compute(target_image)

    # Create a mapping list.
    mapping = np.zeros((256, 1), dtype=np.uint8)
    for source_intensity in range(256):
        source_cdf_value = source_cdf[source_intensity]
        target_intensity = np.abs(target_cdf - source_cdf_value).argmin()
        mapping[source_intensity] = target_intensity
    # Histogram matching referring to the mapping list.
    source_image = source_image.astype(np.uint8)
    matched_image=mapping[source_image]

    return matched_image

def bands_matching(images_source_name,images_target_name,matched_name):
    '''
    Histogram matching
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        images_source_name: File path of the source TIFF image.
        images_target_name: File path of the target TIFF image.
        matched_name: File path of the matched source TIFF image.
    @return:
        None
    '''

    images_source,pro,trans=read_raster_as_array(images_source_name)
    images_target,_,_=read_raster_as_array(images_target_name)
    # Matching three bands respectively.
    image_0=histogram_matching(images_source[:,:,0],images_target[:,:,0])
    image_1 = histogram_matching(images_source[:, :, 1], images_target[:, :, 1])
    image_2 = histogram_matching(images_source[:, :, 2], images_target[:, :,2])
    images_matched=np.concatenate([image_0,image_1,image_2],axis=2)
    images_matched=np.transpose(images_matched,(2,0,1))
    writeTiff(images_matched,trans,pro,matched_name)


def density(data,data_gt,xlabel,xspan,colors):
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams["font.size"] = 20
    # »æÖÆÖ±·½Í¼
    fig, ax = plt.subplots(figsize=(9, 6))
    ax2 = ax.twinx()
    # Plot the histogram and KDE
    sns.distplot(
        data ,
        #hist=True,
        bins=256,
        kde=False,
        color=colors[0],
        norm_hist=False,
        hist_kws={"edgecolor": colors[1], "alpha": 0.6},
        #kde_kws={"color": colors[2], "linewidth":2},
        ax=ax
        # Set the x-axis limits
    )
    sns.distplot(
        data,

        hist=False,
        bins=256,
        kde=True,
        color=colors[1],
        norm_hist=False,
        #hist_kws={"edgecolor": colors[2], "alpha": 0.6},
        kde_kws={"color": colors[2], "linewidth":2},
        ax=ax2
        # Set the x-axis limits
    )
    sns.distplot(
        data_gt,
        hist=False,
        bins=256,
        kde=True,
        color="#7F7F7F",
        norm_hist=False,
        kde_kws={"color": "#7F7F7F", "linewidth": 2,"linestyle":'--'},
        ax=ax2
        # Set the x-axis limits
    )


    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.xlim(xspan)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax2.set_ylabel("Probability density")
    plt.show()


if __name__ == "__main__":
    images_source_name=r"E:\Beijing_flood\TA_demo\tifs\0801_streched.tif"
    images_target_name=r"E:\Beijing_flood\TA_demo\tifs\0712_streched.tif"
    matched_name=r"E:\Beijing_flood\TA_demo\tifs\0801_stretched_matched.tif"

    images_target, pro, trans = read_raster_as_array(images_target_name)
    colors = [(242 / 255, 207 / 255, 211 / 255, 1),
              (231 / 255, 169 / 255, 177 / 255, 1),
              "#E2979F"]

    images_matched, _, _ = read_raster_as_array(matched_name)
    images_source, _, _ = read_raster_as_array(images_source_name)
    max_images_target=np.max(images_target)
    print(max_images_target)
    max_images_source = np.max(images_source)
    images_target[images_target==max_images_target]=np.nan
    images_source[images_source == max_images_source] = np.nan
    density(images_target,images_source, "source", [0,256], colors)




