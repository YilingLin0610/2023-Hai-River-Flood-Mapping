
"""
This section  is used train a model
Author: Modified from https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Mobile.
"""
import os
import datetime

import numpy as np
import torch

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch




def train(model_path,logs_path):
    """
    train a model
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        model_path: The file path of the pretrained checkpoint.
        logs_path: The file path of the trained checkpoint.
    @return:
        None
    """

    # ---------------------------------#
    #   Cuda    Weather to use GPU
    #           Set to False if no GPU is available.
    # ---------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   distributed     Specifies whether to use distributed training with multiple GPUs on a single machine.
    #                   Terminal commands are supported only on Ubuntu. Use CUDA_VISIBLE_DEVICES to specify GPUs on Ubuntu.
    #                   On Windows, the default is to use Data Parallel (DP) mode with all available GPUs, and Distributed Data Parallel (DDP) is not supported.
    #   DP mode:
    #       Set            distributed = False
    #       In terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       In terminal, enter    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py

    # ---------------------------------------------------------------------#
    distributed = True
    # ---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn. This is available in DDP mode with multiple GPUs.
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16       Whether to use mixed precision training.
    #    This can reduce memory usage by approximately half and requires PyTorch 1.7.1 or higher.
    # ---------------------------------------------------------------------#
    fp16 = False
    # -----------------------------------------------------#
    #   num_classes     Must be modified when training on your own dataset.
    #                   Set to the number of classes you need plus one, e.g., 2+1.
    # -----------------------------------------------------#
    #   Backbone used
    #   mobilenet
    #   xception
    # ---------------------------------#
    backbone = "mobilenet"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pre-trained weights for the backbone network. The pre-trained weights are loaded during model construction.
    #                   If `model_path` is set, the backbone weights do not need to be loaded, and the `pretrained` setting is irrelevant.
    #                   If `model_path` is not set and `pretrained = True`, only the backbone is loaded and training starts from there.
    #                   If `model_path` is not set, `pretrained = False`, and `Freeze_Train = False`, training starts from scratch without freezing the backbone.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ---------------------------------------------------------#
    #   downsample_factor   The downsampling factor, with options of 8 or 16.
    #                       A downsampling factor of 8 is smaller and theoretically yields better results.
    #                       However, it requires more GPU memory.
    # ---------------------------------------------------------#
    downsample_factor = 16
    # ------------------------------#
    #   Size of input image
    # ------------------------------#
    input_shape = [512, 512]
    num_classes=3
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two stages: the freezing stage and the unfreezing stage. The freezing stage is designed to accommodate users with limited GPU memory.
    #   Freezing training requires less memory; if you have a very weak GPU, you can set `Freeze_Epoch` to be equal to `UnFreeze_Epoch` to perform only freezing training.
    #
    #   Here are some suggested parameter settings. Adjust them according to your needs:
    #   (1) Training from pre-trained weights of the entire model:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Unfreezing)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Unfreezing)
    #       Note: `UnFreeze_Epoch` can be adjusted between 100 and 300.
    #   (2) Training from pre-trained weights of the backbone network:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (Unfreezing)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 120, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Freezing)
    #           Init_Epoch = 0, UnFreeze_Epoch = 120, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 7e-3, weight_decay = 1e-4. (Unfreezing)
    #       Note: Training from pre-trained backbone weights might require more epochs to avoid local optima. `UnFreeze_Epoch` can be adjusted between 120 and 300.
    #             Adam generally converges faster than SGD, so theoretically, `UnFreeze_Epoch` can be smaller, but more epochs are still recommended.
    #   (3) Setting the batch_size:
    #       Larger batch sizes are preferable if GPU memory permits. Insufficient memory is unrelated to dataset size; if you encounter out-of-memory (OOM) errors, reduce `batch_size`.
    #       Due to the influence of BatchNorm layers, `batch_size` should be at least 2, not 1.
    #       Typically, `Freeze_batch_size` is recommended to be 1-2 times the size of `Unfreeze_batch_size`. Avoid setting a large discrepancy between them, as it affects automatic learning rate adjustment.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Freezing Stage Training Parameters:
    #   During this stage, the backbone of the model is frozen, and only fine-tuning of the network occurs.
    #   It requires less GPU memory and adjusts only the network parameters.
    #   Init_Epoch          The starting epoch for the model training. It can be greater than `Freeze_Epoch`. For example:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    #                       This skips the freezing stage and starts training from epoch 60, adjusting the learning rate accordingly.
    #                       (Used for resuming training)
    #   Freeze_Epoch        The number of epochs for freezing training.
    #                       (Ineffective if `Freeze_Train=False`)
    #   Freeze_batch_size   The batch size for freezing training.
    #                       (Ineffective if `Freeze_Train=False`)

    # ------------------------------------------------------------------#
    Init_Epoch =0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 250
    Unfreeze_batch_size = 4
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   Other Training Parameters: Learning Rate, Optimizer, and Learning Rate Scheduling
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate for the model.
    #                   It is recommended to set Init_lr=5e-4 when using the Adam optimizer.
    #                   It is recommended to set Init_lr=7e-3 when using the SGD optimizer.
    #   Min_lr          The minimum learning rate for the model, default is 0.01 times the maximum learning rate.
    # ------------------------------------------------------------------#

    Init_lr = 7e-3
    Min_lr = 7e-3 * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer to use, options are 'adam' or 'sgd'.
    #                   It is recommended to set Init_lr=5e-4 when using the Adam optimizer.
    #                   It is recommended to set Init_lr=7e-3 when using the SGD optimizer.
    #   momentum        The momentum parameter used by the optimizer.
    #   weight_decay    Weight decay to prevent overfitting.
    #                   Adam may lead to errors with weight_decay; it is recommended to set it to 0 when using Adam.
    # ------------------------------------------------------------------#

    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   The type of learning rate decay to use, options are 'step' or 'cos'.
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     The number of epochs after which to save the model weights.
    # ------------------------------------------------------------------#
    save_period = 100
    # ------------------------------------------------------------------#
    #   save_dir        The directory where model weights and log files are saved.
    # ------------------------------------------------------------------#
    save_dir = r''
    # ------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training on the validation set.
    #   eval_period     The number of epochs between evaluations. Frequent evaluations are not recommended
    #                   as they can be time-consuming and slow down the training process.
    #   Note: The mAP obtained here may differ from the mAP obtained using get_map.py for two reasons:
    #   (1) The mAP obtained here is for the validation set.
    #   (2) The evaluation parameters here are set conservatively to speed up the evaluation process.
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5

    # ------------------------------------------------------------------#
    #   VOCdevkit_path  dataset path
    # ------------------------------------------------------------------#
    VOCdevkit_path = 'RTS_datas'
    # ------------------------------------------------------------------#
    #   Recommended Options:
    #   - When the number of classes is small (a few classes), set this to True.
    #   - When the number of classes is large (more than a dozen classes), and the batch_size is relatively large (10 or more), set this to True.
    #   - When the number of classes is large (more than a dozen classes), and the batch_size is relatively small (less than 10), set this to False.
    # ------------------------------------------------------------------#

    dice_loss = False
    # ------------------------------------------------------------------#
    #   Whether to use focal loss to address class imbalance between positive and negative samples.
    # ------------------------------------------------------------------#
    focal_loss = False
    # ------------------------------------------------------------------#
    #   Whether to assign different loss weights to different classes; the default is balanced.
    #   If set, ensure the weights are provided in a numpy array with a length equal to num_classes.
    #   For example:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    # cls_weights = np.array([1, 2, 3], np.float32)
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     Sets the number of threads used for data loading. Setting it to 1 disables multi-threading.
    #                   Enabling multi-threading can speed up data loading but will use more memory.
    #                   In some cases with Keras, multi-threading can actually slow down the process.
    #                   Enable multi-threading only when I/O is the bottleneck, i.e., when GPU computation speed greatly exceeds the image loading speed.
    # ------------------------------------------------------------------#
    num_workers = 4

    # ------------------------------------------------------#
    #   Set the used GPU
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # ----------------------------------------------------#
    #   Download the pretrained checkpoint
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load weights based on the keys from the pre-trained weights and the model's keys.
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        # pretrained_dict = torch.load(model_path, map_location = 'cuda:0')

        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   Display keys that do not match.
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "´´\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "´´\nFail To Load Key num:", len(no_load_key))

    # ----------------------#
    #   Record Loss
    # ----------------------#
    if local_rank == 0:
        # time_str        = datetime.dat1et
        # ime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, logs_path)
        save_dir = log_dir
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   Multi-GPU Synchronized Batch Normalization
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   Multi-GPU Parallel Execution
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   Read the corresponding dataset txt file
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "RTS_datasets/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "RTS_datasets/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    # Display model configuration
    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   Total training epochs refer to the total number of times
        #   the entire dataset is traversed.
        #   Total training steps refer to the total number of gradient descent operations.
        #   Each training epoch consists of several training steps, where each step performs one gradient descent.
        #   This specifies the minimum recommended training epochs; there is no upper limit.
        #   Only the unfreezing part is considered in the calculation.
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small for training. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total number of training steps to %d or more.\033[0m" % (
            optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] The total training data for this run is %d, with an Unfreeze_batch_size of %d, training for %d epochs��The total number of training steps is calculated%d。\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total number of training steps is %d, which is less than the recommended total steps of %d, it is suggested to set the total number of epochs to %d.\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    #   The backbone feature extraction network is generic,
    #   and frozen training can speed up the training process.
    #   It can also help prevent weights from being damaged in the early stages of training.
    #   Init_Epoch specifies the starting epoch.
    #   Interval_Epoch specifies the epochs for frozen training.
    #   Epoch represents the total number of training epochs.
    #   If encountering OOM or insufficient GPU memory, please reduce the Batch_size.
    # ------------------------------------------------------#

    if True:
        # ------------------------------------#
        #   Freeze certain parts of the training
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   If not using frozen training, set the batch_size directly to Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Adjust the learning rate adaptively based on the current batch_size
        # -------------------------------------------------------------------#

        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   Choose the optimizer based on optimizer_type
        # ---------------------------------------#

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   Obtain the learning rate decay formula
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   Determine the length of each epoch
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("# The dataset is too small to continue training. Please expand the dataset.")

        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        # Download the data
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

        # ----------------------#
        #   Record the mAP curve for evaluation
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   Start model training
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   If the model has a frozen training part,
            #   then unfreeze it and set the parameters
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # -------------------------------------------------------------------#
                #   Adjust the learning rate adaptively based on the current batch_size
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   Obtain the learning rate decay formula
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
