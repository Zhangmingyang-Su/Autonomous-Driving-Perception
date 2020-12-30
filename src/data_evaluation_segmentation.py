from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import cv2 as cv
import tensorflow as tf
import model_loading import loading_model
from tabulate import tabulate
from data_load import fetch

def evaluate_single(seg_map, ground_truth):
    """Evaluate a single frame with the MODEL loaded."""
    # merge label due to different annotation scheme
    seg_map[np.logical_or(seg_map==14,seg_map==15)] = 13
    seg_map[np.logical_or(seg_map==3,seg_map==4)] = 2
    seg_map[seg_map==12] = 11

    # calculate accuracy on valid area
    acc = np.sum(seg_map[ground_truth!=19]==ground_truth[ground_truth!=19])/np.sum(ground_truth!=19)

    # select valid labels for evaluation
    cm = confusion_matrix(ground_truth[ground_truth!=19], seg_map[ground_truth!=19],
                          labels=np.array([0,1,2,5,6,7,8,9,11,13]))
    intersection = np.diag(cm)
    union = np.sum(cm, 0) + np.sum(cm, 1) - np.diag(cm)
    return acc, intersection, union


def semantic_segmentation_iou(SAMPLE_VIDEO):
    # Evaluate on the sample video
    MODEL = loading_model(DeepLabModel)
    video = cv.VideoCapture(SAMPLE_VIDEO)
    num_frames = 598
    acc = []
    intersection = []
    union = []

    for i in tqdm(range(num_frames)):
        _, frame = video.read()
        original_im = Image.fromarray(frame[..., ::-1])
        seg_map = MODEL.run(original_im)
        gt = dataset.fetch(i)
        _acc, _intersection, _union = evaluate_single(seg_map, gt)
        intersection.append(_intersection)
        union.append(_union)
        acc.append(_acc)

    class_iou = np.round(np.sum(intersection, 0) / np.sum(union, 0), 4)
    print('pixel accuracy: %.4f'%np.mean(acc))
    print('mean class IoU: %.4f'%np.mean(class_iou))
    print('class IoU:')
    print(tabulate([class_iou], headers=LABEL_NAMES[[0,1,2,5,6,7,8,9,11,13]]))

def semantic_segmentation_iou_temporal(SAMPLE_VIDEO):
    MODEL = loading_model(DeepLabModel)
    video = cv.VideoCapture(SAMPLE_VIDEO)
    num_frames = 598
    acc = []
    intersection = []
    union = []
    prev_seg_map_logits = 0

    for i in tqdm(range(num_frames)):
        _, frame = video.read()
        original_im = Image.fromarray(frame[..., ::-1])

        # Get the logits instead of label prediction
        seg_map_logits = MODEL.run(original_im, OUTPUT_TENSOR_NAME='ResizeBilinear_3:0')

        # Add previous frame's logits and get the results
        seg_map = np.argmax(seg_map_logits + prev_seg_map_logits, -1)
        prev_seg_map_logits = seg_map_logits

        gt = dataset.fetch(i)
        _acc, _intersection, _union = evaluate_single(seg_map, gt)
        intersection.append(_intersection)
        union.append(_union)
        acc.append(_acc)

    class_iou = np.round(np.sum(intersection, 0) / np.sum(union, 0), 4)
    print('pixel accuracy: %.4f'%np.mean(acc))
    print('mean class IoU: %.4f'%np.mean(class_iou))
    print('class IoU:')
    print(tabulate([class_iou], headers=LABEL_NAMES[[0,1,2,5,6,7,8,9,11,13]]))

if __name__ == "__main__":
    # original sample Video
    SAMPLE_VIDEO = 'mit_driveseg_sample.mp4'
    sample_video_result = semantic_segmentation_iou(SAMPLE_VIDEO)
    sample_video_result_temporal = semantic_segmentation_iou_temporal(SAMPLE_VIDEO)
