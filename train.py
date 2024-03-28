import os
from tqdm import tqdm
from torchinfo import summary

import gc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from lib.datesnet import DatesNet
from lib.datesnet_mlp import DatesNet
from config.datesnet_config import Config as cfg
from data.data_processing import DataProcessing
from data.dataset_loader import FERPlusDatasetLoader
from utils.dnn_utils import get_optimizer, get_lr_scheduler, EvaluationMetric


def train(model, dataset_loader, loss_fn, optimizer, epoch, writer):
    losses = EvaluationMetric()

    model.train()
    for batch_index, (input_data, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        gc.collect()
        torch.cuda.empty_cache()

        if cfg.device == 'cuda':
            target = target.cuda(non_blocking=True)

        log_prob = model(input_data)

        loss = loss_fn(log_prob, target)



        # Store the losses
        losses.update(loss.item(), cfg.TrainConfig.batch_size)

        # Step the optimizer and backprop losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Add loss to Tensorboard writer
    writer.add_scalar("Loss/train", losses.value, epoch)
    print('Training epoch {}: loss {:.4f}'.format(epoch, losses.value))


def validation(model, dataset_loader, loss_fn, epoch, writer):
    losses = EvaluationMetric()

    model.eval()

    with torch.no_grad():
        for batch_index, (input_data, target) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            gc.collect()
            torch.cuda.empty_cache()

            if cfg.device == 'cuda':
                target = target.cuda(non_blocking=True)

            output = model(input_data)

            loss = loss_fn(output, target)

            losses.update(loss.item(), cfg.TrainConfig.batch_size)

    # Add loss and nme to Tensorboard writer
    writer.add_scalar("Loss/validation", losses.average, epoch)

    print('Validation epoch {}: loss {:.4f}'.format(epoch, losses.average))

    return losses.average


def check_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main():
    # Initialization of variables
    best_val_loss = 1e9
    best_model_state_dict = dict()
    check_dirs(cfg.logdir)
    check_dirs(cfg.output_dir)
    check_dirs(cfg.checkpoint_dir)

    # load the model
    model = DatesNet(cfg=cfg)

    batch_size = 1
    summary(model, input_size=(batch_size, 3, cfg.DatasetConfig.image_shape[0], cfg.DatasetConfig.image_shape[1]))
    print('Model parameters: %.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))

    if 'cuda' in cfg.device:
        cudnn.benchmark = cfg.TrainConfig.is_cudnn_benchmark_enabled
        cudnn.deterministic = cfg.TrainConfig.is_cudnn_deterministic
        cudnn.enabled = cfg.TrainConfig.is_cudnn_enabled
        gpus = list(cfg.TrainConfig.gpus)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Loss function
    # loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Optimizer
    optimizer = get_optimizer(model, cfg)

    # Learning rate scheduler
    lr_scheduler = get_lr_scheduler(cfg, optimizer=optimizer, last_epoch=0)

    # Loading Tensorboard writer
    writer = SummaryWriter(log_dir=cfg.logdir)

    # Resume training
    if cfg.TrainConfig.resume_training:
        device = torch.device(cfg.device)

        if len(os.listdir(cfg.checkpoint_dir)) != 0:
            last_epoch = max([int(os.path.splitext(x)[0].split('_')[-1]) for x in os.listdir(cfg.checkpoint_dir) if 'model' in x])
            model_file_name = os.path.join(cfg.checkpoint_dir, 'model_' + str(last_epoch) + '.pth')

            checkpoint = torch.load(model_file_name, map_location=device)

            state_dict = checkpoint.get('state_dict')

            if cfg.device == 'cuda':
                model.module.load_state_dict(state_dict=state_dict)
                model.to(device=device)
            else:
                model.load_state_dict(state_dict=state_dict)

            optimizer.load_state_dict(checkpoint.get('optimizer'))

            print("=> loaded checkpoint (epoch {})".format(last_epoch))
        else:
            print("=> no checkpoint found")

    # loading training dataset
    data_processor = DataProcessing(cfg)
    mean, std = data_processor.compute_db_mean_std()
    print("mean: {}, std: {}".format(mean, std))
    aug_transforms = data_processor.get_online_aug_transform(mean=mean, std=std)
    train_dataset = FERPlusDatasetLoader(cfg=cfg, dataset_name='train', transforms=aug_transforms)
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=cfg.TrainConfig.batch_size,
                                      shuffle=cfg.TrainConfig.shuffle,
                                      num_workers=cfg.TrainConfig.workers,
                                      pin_memory=cfg.TrainConfig.pin_memory)

    # loading validation dataset
    val_aug_transform = data_processor.get_valid_transform(mean=mean, std=std)
    validation_dataset = FERPlusDatasetLoader(cfg=cfg, dataset_name='val', transforms=val_aug_transform)
    validation_dataset_loader = DataLoader(validation_dataset,
                                           batch_size=cfg.ValidConfig.batch_size,
                                           shuffle=cfg.ValidConfig.shuffle,
                                           num_workers=cfg.ValidConfig.workers,
                                           pin_memory=cfg.ValidConfig.pin_memory)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # for batch_idx, (inputs, _) in enumerate(validation_dataset_loader):
    #     fig = plt.figure(figsize=(14, 7))
    #     for i in range(1):
    #         ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    #         plt.imshow((inputs[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    #     break
    # plt.show(block=True)

    # Run training and validation for all the epochs
    for curr_epoch_idx in tqdm(range(cfg.TrainConfig.number_epochs)):
        train(model=model, dataset_loader=train_dataset_loader, optimizer=optimizer, loss_fn=loss_fn, epoch=curr_epoch_idx, writer=writer)
        val_loss = validation(model=model, dataset_loader=validation_dataset_loader, loss_fn=loss_fn, epoch=curr_epoch_idx, writer=writer)

        lr_scheduler.step()

        # Get model state dict to save for inference or resume training
        model_state_dict = model.module.state_dict() if cfg.device == 'cuda' else model.state_dict()
        states = dict(state_dict=model_state_dict,
                      optimizer=optimizer.state_dict(),
                      last_epoch=curr_epoch_idx,
                      best_val_loss=best_val_loss)

        # Name the checkpoint file and save the checkpoint
        checkpoint_model_file_name = os.path.join(cfg.checkpoint_dir, 'datesnet_model_cont.pth')
        if (curr_epoch_idx + 1) % 5 == 0 or (curr_epoch_idx + 1) == cfg.TrainConfig.number_epochs:
            torch.save(states, checkpoint_model_file_name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = states

    if best_model_state_dict:
        final_output_model_file_name = os.path.join(cfg.output_dir, 'datesnet_model' + '.pth')
        torch.save(best_model_state_dict, final_output_model_file_name)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    # import cv2
    # import numpy as np
    # import pandas as pd
    # from tqdm import tqdm
    #
    # fer = pd.read_csv(r'D:\dataset\face\FER+\fer2013.csv')
    # fer_plus = pd.read_csv(r'D:\dataset\face\FER+\fer2013new.csv')
    #
    # images = list()
    # labels = list()
    # classes = fer_plus.columns[2:-1].tolist()
    # data = fer_plus[fer_plus["Usage"] == "Training"]
    # for idx, (image_name, image_data) in tqdm(enumerate(zip(data['Image name'], fer['pixels'])), total=len(data)):
    #     if pd.isna(image_name):
    #         continue
    #
    #     image_data = np.fromstring(image_data, dtype=np.uint8, sep=' ').reshape(48, 48)
    #     cv2.namedWindow('test_image', cv2.WINDOW_NORMAL)
    #     cv2.imshow('test_image', image_data)
    #     cv2.waitKey(0)
    #     images.append(image_data)
    #     labels.append([fer_plus.loc[idx, each_class] for each_class in classes])

    # target = np.array([fer_plus.loc[idx, each_class] / 10.0 for each_class in classes])

    # idx = np.random.choice(len(target), p=target)
    # new_target_old = np.zeros_like(target)
    # new_target_old[idx] = 1.0
    #
    # new_target = np.array(target)
    # new_target[new_target > 0] = 1.0
    # epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
    # alpha = 0.001
    # res = (1 - alpha) * target + alpha * np.ones_like(target)
    #
    # images_ar = np.array(images)
    # print('checkpoint')

    main()
