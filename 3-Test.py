import json
import argparse
import torch
import numpy as np
import re

from model.fusion.bev.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder_bev
from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
import cv2
from utils.util import DisplayHMI_BEV

gpu_id = 0

def main(config, checkpoint_filename):

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

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

    # Set up the VideoWriter
    save_images = False
    save_video = False
    video = cv2.VideoWriter(
        f'./result_CRLEarlyRaw.mp4',
        cv2.VideoWriter_fourcc(*'DIVX'), 10, (1700, 540)) #1616, 512

    for ii, data in enumerate(dataset):
        inputs1 = torch.tensor(data[0]).permute(2, 0, 1).to(device).float().unsqueeze(0)
        inputs2 = torch.tensor(data[1]).permute(2, 0, 1).to(device).float().unsqueeze(0)
        seg_map_label = torch.tensor(data[2]).to(device).double()
        det_label = torch.tensor(data[3]).to(device).float().unsqueeze(0)
        box_labels = data[4]
        sample_id = re.search(r'_([0-9]+)\.jpg$', data[5])
        sample_id = sample_id.group(1)
        sample_id = int(sample_id)
        with torch.set_grad_enabled(False):
            outputs = net(inputs2, inputs1)
        hmi = DisplayHMI_BEV(data[5],data[0], outputs, box_labels, enc,sample_id, datapath=config['dataset']['root_dir'])

        if save_video == True:
            video.write(hmi)
            cv2.imshow('Multi-Tasking', hmi)

        elif save_images == True:
            cv2.imwrite('/media/BEV_camera/samples/' + data[6][-16:], hmi[:, 257:1217, :]) #hmi*255
            cv2.imwrite('/media/BEV_camera/samples/' + 'bev_' +data[6][-16:], hmi[:, 1217:1473, :])  # hmi*255

        else:
            cv2.imshow('refnet', hmi)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # out.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualization')

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