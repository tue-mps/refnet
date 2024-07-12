import os
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union
import polarTransform

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                          [0.00000000e+00, 1.78869210e+03, 6.07342667e+02],
                          [0., 0., 1]])
dist_coeffs = np.array([2.51771602e-01, -1.32561698e+01, 4.33607564e-03, -6.94637533e-03, 5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624, -0.04003127])
tvecs = np.array([0.09138029, 1.38369885, 1.43674736])
ImageWidth = 1920
ImageHeight = 1080

AoA_mat = np.load('./resources/CalibrationTable.npy',allow_pickle=True).item()

numSamplePerChirp = 512
numRxPerChip = 4
numChirps = 256
numRxAnt = 16
numTxAnt = 12
numReducedDoppler = 16
numChirpsPerLoop = 16
dividend_constant_arr = np.arange(0, numReducedDoppler*numChirpsPerLoop ,numReducedDoppler)
window = np.array(AoA_mat['H'][0])
CalibMat=AoA_mat['Signal'][...,5]

def worldToImage(x, y, z):
    world_points = np.array([[x, y, z]], dtype='float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0, imgpts[0][0][0]), ImageWidth - 1))
    v = int(min(max(0, imgpts[0][0][1]), ImageHeight - 1))

    return u, v


def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W / 2, y, x + W / 2, y, x + W / 2, y + L, x - W / 2, y + L, data[i][0], data[i][1]])

    return boxes


def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):
    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions


def bbox_iou(box1, boxes):
    # currently inspected box
    box1 = box1.reshape((4, 2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4, 2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):
    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions

    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:, -1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))

    return final_point_cloud_predictions


def DisplayHMI_BEV(camimage_path, input, model_outputs, box_labels, encoder, sample_id, datapath):
    RA_VisualPic_ = True

    camimage = cv2.imread(camimage_path)
    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]

    # Decode the output detection map
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if (len(pred_obj) > 0):
        pred_obj = process_predictions_FFT(pred_obj, confidence_threshold=0.2)

    ## FFT
    FFT = np.abs(input[..., :16] + input[..., 16:] * 1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum - PowerSpectrum.min()) / (PowerSpectrum.max() - PowerSpectrum.min()) * 255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'), cv2.COLOR_GRAY2BGR)

    ## Image
    Object_pred_ra = []
    Object_pred_pic = []
    m = 0
    for box in pred_obj:
        if box[0] > 0.6:
            box = box[1:]
            u1, v1 = worldToImage(-box[2], box[1], 0)
            u2, v2 = worldToImage(-box[0], box[1], 1.6)

            u1 = int(u1 / 2)
            v1 = int(v1 / 2)
            u2 = int(u2 / 2)
            v2 = int(v2 / 2)

            finall_pred_pic = [(u1, v1), (u2, v2)]
            Object_pred_pic.append(finall_pred_pic)

            pred_obj_RA = pred_obj[:, -2:]
            R = pred_obj_RA[m][0]
            A = pred_obj_RA[m][1]
            # RA_box_pred = RA_to_cartesian_box(np.asarray([R, A])[np.newaxis, :])
            # final_Object_predictions = np.asarray(RA_box_pred)

            final_pred_ra = np.asarray([R, A])[np.newaxis, :]
            Object_pred_ra.append(final_pred_ra)

            camimage = cv2.rectangle(camimage, (u1, v1), (u2, v2), (255, 0, 0), 2)
            m += 1

    for box_label in box_labels: #ground truth bbox
        x1, y1, x2, y2 = box_label[6], box_label[7], box_label[8], box_label[9]
        # Convert to integer and ensure coordinates are within image dimensions
        x1, y1, x2, y2 = int(x1 / 2), int(y1 / 2), int(x2 / 2), int(y2 / 2)
        camimage = cv2.rectangle(camimage, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if RA_VisualPic_ == True:
        image, RA = RA_VisualPic_save(sample_id, Object_pred_pic, Object_pred_ra, box_labels, data_path=datapath)
        RA = RA.transpose(1,0,2)
        RA = RA[:512]
        return np.hstack((PowerSpectrum, camimage[:512], RA))

def RA_VisualPic_save(id, pre, pre_ra, box_labels, data_path):

    img_name = os.path.join(data_path, 'camera', "image_{:06d}.jpg".format(id))
    image = np.asarray(Image.open(img_name))
    image_cv = cv2.imread(img_name)

    radar_name = os.path.join(data_path, 'radar_FFT', "fft_{:06d}.npy".format(id))
    input = np.load(radar_name, allow_pickle=True)
    radar_FFT = input.real + 1j * input.imag

    ## radar_signal_processing
    doppler_indexes = []
    for doppler_bin in range(numChirps):
        DopplerBinSeq = np.remainder(doppler_bin + dividend_constant_arr, numChirps)
        DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]], DopplerBinSeq[5:]])
        doppler_indexes.append(DopplerBinSeq)

    MIMO_Spectrum = np.array(radar_FFT[:, doppler_indexes, :].reshape(radar_FFT.shape[0] * radar_FFT.shape[1], -1))
    MIMO_Spectrum = np.multiply(MIMO_Spectrum, window).transpose()
    Azimuth_spec = np.abs(np.dot(CalibMat, MIMO_Spectrum))
    Azimuth_spec = Azimuth_spec.reshape(AoA_mat['Signal'].shape[0], radar_FFT.shape[0], radar_FFT.shape[1])
    RA_map = np.log10(np.sum(np.abs(Azimuth_spec), axis=2))
    RA_map = RA_map / np.max(RA_map) * 255

    # visualize
    plt.figure(id)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(RA_map)
    currentAxis = fig.gca()
    for i in range(len(pre)):
        image = cv2.rectangle(image_cv, pre[i][0], pre[i][1], (255, 0, 0), 2)
        r = int(pre_ra[i][:, 0] * 512/103)
        a = int(-pre_ra[i][:, 1] / 0.2 + 375)
        rect = patches.Rectangle((r, a), 35, 20, linewidth=1, edgecolor='b', facecolor='none')
        currentAxis.add_patch(rect)

        rgt = int(box_labels[i][0] * 512 / 103)
        agt = int(-box_labels[i][1] / 0.2 + 375)
        rectgt = patches.Rectangle((rgt, agt), 35, 20, linewidth=1, edgecolor='g', facecolor='none')
        currentAxis.add_patch(rectgt)

    plt.axis('off')
    # plt.show()
    fig.canvas.draw()
    rgb_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    ra_ = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    ra = cv2.resize(ra_, (image.shape[1], image.shape[0]))
    ra = cv2.flip(ra, -1)
    ra = cv2.rotate(ra, cv2.ROTATE_90_CLOCKWISE)

    return image, ra
