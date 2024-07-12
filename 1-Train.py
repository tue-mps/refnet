import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
from datetime import datetime

######Standalone operation#######
from model.standalone.only_camera import cam_only
from model.standalone.FFTRadNet import FFTRadNet

######REFNET: Our Model#######
from model.fusion.bev.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder_bev

######Perspective Fusion: out of scope#######
from model.fusion.perspective.earlyfusion import earlyfusion

####Ablation#####
from model.fusion.bev.camrad_ad_fullresnet50 import cr_ad_res50full
from model.fusion.bev.efficientnetb2.effb2_model import cr_adfusion_effnetb2
from model.fusion.bev.unetformer import cr_adfusion_unetformer

from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
from loss import pixor_loss
from utils.evaluation import run_evaluation
from utils.metrics_bev import count_params

def main(config):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)
    st = time.time()

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config_allmodality.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device Used:', device)
    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # load dataset and create model
    if config['model']['view_perspective'] == 'True':
        dataset = RADIal(config=config,
                         encoder=None,
                         difficult=True)
        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['perspective']['only_camera'] == 'True':
            net = cam_only(channels_bev=config['model']['channels_bev'],
                           blocks=config['model']['backbone_block'],
                           detection_head=config['model']['DetectionHead'],
                           segmentation_head=config['model']['SegmentationHead'],
                           config=config)
            print("***************************************")
            print("CameraOnly in front view has been chosen")
            print("***************************************")

        if config['architecture']['perspective']['early_fusion'] == 'True':
            net = earlyfusion(channels_bev=config['model']['channels_bev'],
                              blocks=config['model']['backbone_block'],
                              detection_head=config['model']['DetectionHead'],
                              segmentation_head = config['model']['SegmentationHead'],
                              config=config)
            print("***************************************")
            print("Early fusion in front view has been chosen")
            print("***************************************")

    if config['model']['view_birdseye'] == 'True':
        enc = ra_encoder(geometry=config['dataset']['geometry'],
                         statistics=config['dataset']['statistics'],
                         regression_layer=2)

        dataset = RADIal(config=config,
                         encoder=enc.encode,
                         difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['bev']['only_radar'] == 'True':
            net = FFTRadNet(blocks=config['model']['backbone_block'],
                            mimo_layer=config['model']['MIMO_output'],
                            channels=config['model']['channels'],
                            detection_head=config['model']['DetectionHead'],
                            segmentation_head=config['model']['SegmentationHead'],
                            config=config,
                            regression_layer=2)
            print("***************************************")
            print("Only Radar (FFTRadNet) has been chosen")
            print("***************************************")

        if config['architecture']['bev']['after_decoder_fusion'] == 'True':
            net = cameraradar_fusion_Afterdecoder_bev(mimo_layer=config['model']['MIMO_output'],
                                                      channels=config['model']['channels'],
                                                      channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head=config['model']['SegmentationHead'],
                                                      config=config,
                                                      regression_layer=2)

            print("*************************************************")
            print("REFNET: CameraRadar AD (after-decoder) fusion in BEV has been chosen")
            print("*************************************************")

        if config['architecture']['bev']['ad_fusion_res50full_ablation'] == 'True':
            net = cr_ad_res50full(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  blocks=config['model']['backbone_block'],
                                  detection_head=config['model']['DetectionHead'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  config=config,
                                  regression_layer=2)
            print("*************************************************")
            print("(Resnet50-Ablation)CameraRadar AD fusion in BEV has been chosen")
            print("*************************************************")

        if config['architecture']['bev']['ad_fusion_effB2_ablation'] == 'True':
            net = cr_adfusion_effnetb2(mimo_layer=config['model']['MIMO_output'],
                                      channels=config['model']['channels'],
                                      blocks=config['model']['backbone_block'],
                                      detection_head=config['model']['DetectionHead'],
                                      segmentation_head=config['model']['SegmentationHead'],
                                      config=config,
                                      regression_layer=2)
            print("*************************************************")
            print("(EfficientNetB2-Ablation)CameraRadar AD fusion in BEV has been chosen")
            print("*************************************************")

        if config['architecture']['bev']['ad_fusion_unetformer_ablation'] == 'True':
            net = cr_adfusion_unetformer(mimo_layer=config['model']['MIMO_output'],
                                          channels=config['model']['channels'],
                                          blocks=config['model']['backbone_block'],
                                          detection_head=config['model']['DetectionHead'],
                                          segmentation_head=config['model']['SegmentationHead'],
                                          config=config,
                                          regression_layer=2)
            print("*************************************************")
            print("(UNetFormer-Ablation)CameraRadar AD fusion in BEV has been chosen")
            print("*************************************************")

    print('Number of trainable parameters in the model: %s' % str(count_params(net) / 1e6))
    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'mIoU': []}
    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')

    classif_loss = torch.tensor(0, dtype=torch.float64)
    reg_loss = torch.tensor(0, dtype=torch.float64)
    loss_seg = torch.tensor(0, dtype=torch.float64)

    for epoch in range(startEpoch,num_epochs):
        
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            if (config['architecture']['perspective']['only_camera'] == 'True' or config['architecture']['perspective']['early_fusion'] == 'True'
                    or config['architecture']['bev']['only_radar'] == 'True'):
                inputs1 = data[0].to(device).float() # fused inputs
                seg_map_label = data[1].to(device).double()
                det_label = data[2].to(device).float()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = net(inputs1)

            if (config['architecture']['bev']['after_decoder_fusion'] == 'True'
                or config['architecture']['bev']['ad_fusion_res50full_ablation'] == 'True'
                or config['architecture']['bev']['ad_fusion_effB2_ablation'] == 'True'
                or config['architecture']['bev']['ad_fusion_unetformer_ablation'] == 'True'
                ):
                inputs1 = data[0].to(device).float()
                inputs2 = data[1].to(device).float()
                seg_map_label = data[2].to(device).double()
                det_label = data[3].to(device).float()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = net(inputs2, inputs1)

            if config['model']['DetectionHead']=='True':
                classif_loss, reg_loss = pixor_loss(outputs['Detection'], det_label, config['losses'],config['model'])
                classif_loss *= config['losses']['weight'][0]
                reg_loss *= config['losses']['weight'][1]
            if config['model']['SegmentationHead'] == 'True':
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = freespace_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                loss_seg *= config['losses']['weight'][2]

            loss = classif_loss + reg_loss + loss_seg
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

            if config['model']['view_birdseye'] == 'True':
                writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
                # backprop
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs1.size(0)
                if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                    kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),
                                           ("freeSpace", loss_seg.item())])
                if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                    kbar.update(i, values=[("freeSpace", loss_seg.item())])
                if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                    kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item())])
            else:
                # backprop
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs1.size(0)
                if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                    kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("freeSpace", loss_seg.item())])
                if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                    kbar.update(i, values=[("freeSpace", loss_seg.item())])
                if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                    kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item())])
                # kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()),
                #                        ("freeSpace", loss_seg.item())])

            global_step += 1

        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        ######################
        ## validation phase ##
        ######################

        if config['model']['view_birdseye'] == 'True':
            eval = run_evaluation(net=net,loader=val_loader,
                                  device=device,config=config,encoder=enc,
                                  detection_loss=pixor_loss,
                                  segmentation_loss=freespace_loss,
                                  losses_params=config['losses'],
                                  mode_params=config['model'])
        else:
            eval = run_evaluation(net=net,loader=val_loader,
                                  device=device,config=config,encoder=None,
                                  detection_loss=pixor_loss,
                                  segmentation_loss=freespace_loss,
                                  losses_params=config['losses'],
                                  mode_params=config['model'])

        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])
            history['mIoU'].append(eval['mIoU'])

            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])

        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            history['val_loss'].append(eval['loss'])
            history['mIoU'].append(eval['mIoU'])

            kbar.add(1, values=[("val_loss", eval['loss']),
                                ("mIoU", eval['mIoU'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name'] + '_epoch{:02d}_loss_{:.4f}_IOU_{:.4f}.pth'.format(
                epoch, eval['loss'], eval['mIoU'])

        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])

            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'])

        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        et = time.time()
        elapsed_time_seconds = et - st
        elapsed_time_minutes = elapsed_time_seconds / 60
        print('Total time consumed so far in minutes:', elapsed_time_minutes, 'minutes')
        elapsed_time_hours = (et - st) / (60 * 60)
        print('Total time consumed so far in hours:', elapsed_time_hours, 'hours')

        print('')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='REFNET Training')
    parser.add_argument('-c', '--config', default='config/config_allmodality.json',type=str,
                        help='Path to the config file (default: config_allmodality.json)')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config)
