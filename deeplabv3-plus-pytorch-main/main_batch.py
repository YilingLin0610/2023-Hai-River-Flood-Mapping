#coding=gbk
from train_function import train
from prediction import prediction
import os
from post_processing import post_processing
from osgeo import gdal
import warnings
from utils.utils import show_configs
import shutil

#os.environ['PROJ_LIB'] = r'~/miniconda3/envs/dann/share/proj'

def process_all(name_plan,num_GT,model_path,logs_path,jpgPath,shapefile2,basename):
    """

    Step 1: Train the model
    Step 2: Prediction
    Step 3: Post-processing
    ——————————————————————————
    @param：
        name_plan: Note the logs file we used here.
        num_GT: The number of ground truth.
        model_path: The file path of pre-trained checkpoint.
        logs_path: The file path of the trained checkpoint.
        jpgPath: The filepath of the testing JPG images.
        shapefile2: The ground truth shapefile.
        basename: The date.

    @return:
        None
    """
    parameters = {
        "name_plan": name_plan,
        "num_GT":num_GT,
        "model_path":model_path,
        "logs_path":logs_path,
        "jpg_path":jpgPath,
        "shapefile2":shapefile2
    }
    show_configs(parameters)
    print("train")
    # Train the model
    train(model_path, logs_path)
    # Predict the testing images.
    #predictionPath = r"result_batch/" + name_plan +basename+ "_0601_tune/prediction/"
    predictionPath = r"result_batch_haihe/" + name_plan +basename+ "/prediction/"
    # Create filepath to store the prediction results.
    try:
        #os.mkdir(r"result_batch/" + name_plan +basename+"_0601_tune")
        os.mkdir(r"result_batch_haihe/" + name_plan +basename)
    except:
        print("Exist")

    model_path=logs_path+"//best_epoch_weights.pth"
    try:
        os.mkdir(predictionPath)
    except:
        print("Exist")
    prediction(jpgPath, predictionPath,model_path=model_path)

    # Post-processing to generate the final shapefiles.
    RS_image_path=r"/media/dell/disk/Yiling/Beijing_flood/testing_batch_haihe/testing_data/"+basename+"/tifs/"
    FilePath=r"result_batch_haihe/" +name_plan+basename
    #FilePath=r"result_batch/" +name_plan+basename+"_0601_tune"
    post_processing(RS_image_path, FilePath,num_GT,shapefile2)


if __name__ == "__main__":

    num_GT = 1601
    basenames=os.listdir(r"/media/dell/disk/Yiling/Beijing_flood/testing_batch_haihe/testing_data")
    #basenames=["0712Beijing"]
    for basename in basenames:
        name_plan = "0712_Beijing_with_negative_st"
        #name_plan="0712_Beijing_with_negative_st_0601_plus_0712_V2_more_negative"
        if(os.path.exists(r"result_batch_haihe/" + name_plan +basename+ "/merge_prediction_shp/prediction_filted.shp")):
            print("exist, skip")
        else:
            model_path = r"/media/dell/disk/Yiling/Beijing_flood/deeplabv3-plus-pytorch-main/deeplabv3-plus-pytorch-main/model_data/deeplab_mobilenetv2.pth"
            #model_path=r"/media/dell/disk/Yiling/Beijing_flood/deeplabv3-plus-pytorch-main/deeplabv3-plus-pytorch-main/logs/0712_Beijing_with_negative_from_0805/last_epoch_weights.pth"
            #model_path = r"/media/dell/disk/Yiling/Beijing_flood/deeplabv3-plus-pytorch-main/deeplabv3-plus-pytorch-main/logs/0712_Beijing_with_negative_st_0601_plus_0712/best_epoch_weights.pth"
            jpgPath = r"/media/dell/disk/Yiling/Beijing_flood/testing_batch_haihe/testing_data/" + basename + "/jpgs/"
            # The ground truth used to evaluate the accuracy.
            shapefile2 = r"/media/dell/disk/Yiling/Beijing_flood/full_cover_data/flood_label/Beijing_0712_pro.shp"
            #name_plan = "0712_Beijing_with_negative_st_0601_plus_0712_V2_more_negative"
            name_plan = "0712_Beijing_with_negative_st"
            logs_path = r'logs/' + name_plan
            process_all(name_plan, num_GT, model_path,
                         logs_path, jpgPath, shapefile2, basename)
            # To free up storage space, we delete the intermediate files and retain only the final shapefile results.
            try:
                shutil.rmtree(r"result_batch/" + name_plan +basename+ "_0601_tune/prediction/")
                shutil.rmtree(r"result_batch_haihe/" + name_plan +basename+ "/prediction/")
                print("Files successful deleted")
            except:
                print("Files deletion fail")
            try:
                shutil.rmtree(r"result_batch/" + name_plan +basename+ "_0601_tune/merge_prediction_tifs/")
                shutil.rmtree(r"result_batch_haihe/" + name_plan +basename+ "/merge_prediction_tifs/")
                print("Files successful deleted")
            except:
                print("Files deletion fail")



