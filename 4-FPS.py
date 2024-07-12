import json
import argparse
import torch
import numpy as np


from model.fusion.bev.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder_bev
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
import time
from dataset.dataloader_fusion import CreateDataLoaders

def calculate_fps(model, inputs1):
    start_time = time.time()
    for i in range(100):
        model(inputs1)
    end_time = time.time()
    fps = 100 / (end_time - start_time)
    return fps

def calculate_fps_fusion(model, inputs1, inputs2):
    start_time = time.time()
    for i in range(100):
        model(inputs2, inputs1)
    end_time = time.time()
    fps = 100 / (end_time - start_time)
    return fps

def main(config, checkpoint_filename):
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset and create model
    if config['model']['view_birdseye'] == 'True':
        enc = ra_encoder(geometry=config['dataset']['geometry'],
                         statistics=config['dataset']['statistics'],
                         regression_layer=2)

        dataset = RADIal(config=config,
                         encoder=enc.encode,
                         difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['bev']['after_decoder_fusion'] == 'True':
            net = cameraradar_fusion_Afterdecoder_bev(mimo_layer=config['model']['MIMO_output'],
                                                      channels=config['model']['channels'],
                                                      channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head=config['model']['SegmentationHead'],
                                                      config=config,
                                                      regression_layer=2)

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    fps_list = []
    for idx, data in enumerate(test_loader):

        if (config['architecture']['perspective']['only_camera'] == 'True' or
            config['architecture']['perspective']['early_fusion'] == 'True' or
            config['architecture']['bev']['only_radar'] == 'True'):
            inputs1 = data[0].to(device).float()
            fps = calculate_fps(net, inputs1)
            fps_list.append(fps)
            print(f"FPS for image {idx + 1}: {fps:.2f}")

        if (config['architecture']['bev']['after_decoder_fusion'] == 'True' or
            config['architecture']['bev']['ad_fusion_res50full_ablation'] == 'True' or
            config['architecture']['bev']['ad_fusion_effB2_ablation'] == 'True' or
            config['architecture']['bev']['ad_fusion_unetformer_ablation'] == 'True' or
            config['architecture']['bev']['x4_fusion'] == 'True'):
            inputs1 = data[0].to(device).float()
            inputs2 = data[1].to(device).float()

            fps = calculate_fps_fusion(net, inputs1, inputs2)
            fps_list.append(fps)
            print(f"FPS for image {idx + 1}: {fps:.2f}")

    average_fps = np.mean(fps_list)
    print("**********************************************")
    print(f"Average FPS for all images: {average_fps:.2f}")

    # Calculate and print the standard deviation of FPS
    std_dev_fps = np.std(fps_list)
    print(f"Standard Deviation of FPS for all images: {std_dev_fps:.2f}")
    print("**********************************************")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FPS Computation')

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