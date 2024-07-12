import json
import argparse
import torch
import random
import numpy as np

######Standalone operation#######
from model.standalone.only_camera import cam_only
from model.standalone.FFTRadNet import FFTRadNet

######REFNET: Our Model#######
from model.fusion.bev.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder_bev

######Perspective Fusion: out of scope#######
from model.fusion.perspective.earlyfusion import earlyfusion

from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
from utils.evaluation import run_FullEvaluation

def main(config, checkpoint):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            print("Evaluating the model (refnet)...")
            print("*************************************************")

    net.to(device)

    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')

    if config['model']['view_birdseye'] == 'True':
        run_FullEvaluation(net=net, loader=test_loader,
                           device=device, config=config,
                           encoder=enc)
    else:
        run_FullEvaluation(net=net, loader=test_loader,
                           device=device, config=config,
                           encoder=None)

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('-c', '--config',
                        default='/home/kach271771/Desktop/config/config_allmodality.json',
                        type=str,
                        help='Path to the config file (default: config_allmodality.json)')

    parser.add_argument('-r', '--checkpoint',
                        default="/home/kach271771/Desktop/resources/pretrained_model/OnlyDetection_CameraRadarAD_epoch99_loss_97041.6179_AP_0.9624_AR_0.9216.pth",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')

    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint)
