# Example usage
# python yolov8_pose_webcam.py --model_path /home/markwinap/models/yolov8n-pose.rknn
import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2,math
import signal
from math import ceil

# Import RKNN API (try main library first, fallback to lite version if unavailable)
try:
    from rknn.api import RKNN
except ImportError:
    print('import rknn failed, try to import rknnlite')
    from rknnlite.api import RKNNLite as RKNN

CLASSES = ['person'] # List of object classes to detect

# NMS (Non-Maximum Suppression) threshold for filtering overlapping detections
nmsThresh = 0.4
# Confidence threshold for object detection
objectThresh = 0.5

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    """
    Calculate the Intersection over Union (IOU) of two bounding boxes.
    
    :param xmin1: minimum x-coordinate of first bounding box
    :param ymin1: minimum y-coordinate of first bounding box
    :param xmax1: maximum x-coordinate of first bounding box
    :param ymax1: maximum y-coordinate of first bounding box
    :param xmin2: minimum x-coordinate of second bounding box
    :param ymin2: minimum y-coordinate of second bounding box
    :param xmax2: maximum x-coordinate of second bounding box
    :param ymax2: maximum y-coordinate of second bounding box
    :return: IOU value (intersection area / union area)
    """
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    """
    Perform Non-Maximum Suppression (NMS) on detection results.
    
    Removes duplicate detections by suppressing boxes with high Intersection over Union (IOU)
    for the same class. Boxes are processed in order of descending confidence score.
    
    :param detectResult: list of detection objects with attributes (xmin, ymin, xmax, ymax, classId, score)
    :return: list of detection objects after NMS filtering
    """
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    """
    Apply sigmoid activation function.
    
    :param x: input array or scalar
    :return: sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    """
    Apply softmax activation function.
    
    Normalizes input to probability distribution. Subtracts max for numerical stability.
    
    :param x: input array
    :param axis: axis along which to apply softmax (default: -1)
    :return: softmax(x) normalized along specified axis
    """
    # Subtracting the maximum value from the input vector improves numerical stability.
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process(out, keypoints, index, model_w, model_h, stride, scale_w=1, scale_h=1):
    """
    Process YOLOv8 pose detection output and decode bounding boxes with keypoints.
    
    Decodes model predictions into detection boxes with pose keypoints. Applies spatial
    transformations to convert model coordinates to image coordinates.
    
    :param out: raw model output predictions (xywh and confidence scores)
    :param keypoints: decoded keypoint coordinates from model
    :param index: starting index for keypoint extraction
    :param model_w: model output width
    :param model_h: model output height
    :param stride: stride of feature map relative to input
    :param scale_w: width scale factor for coordinate conversion (default: 1)
    :param scale_h: height scale factor for coordinate conversion (default: 1)
    :return: list of DetectBox objects with bounding boxes and keypoints
    """
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0,c,(h*model_w)+w]>objectThresh:
                    xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                    xywh_=xywh_.reshape(1,4,16,1)
                    data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                    xywh_=softmax(xywh_,2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp=xywh_.copy()
                    xywh_temp[0]=(w+0.5)-xywh_[0]
                    xywh_temp[1]=(h+0.5)-xywh_[1]
                    xywh_temp[2]=(w+0.5)+xywh_[2]
                    xywh_temp[3]=(h+0.5)+xywh_[3]

                    xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                    xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                    xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                    xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                    xywh_=xywh_*stride

                    xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint=keypoints[...,(h*model_w)+w+index] 
                    keypoint[...,0:2]=keypoint[...,0:2]//1
                    box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                    out.append(box)

    return out

# RGB color palette for pose visualization (20 colors)
pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)
# Keypoint colors indexed from pose_palette
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# Bone connections between keypoints
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# Limb/bone colors for skeleton visualization
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yolov8 Pose Python Demo', add_help=True)
    # basic params
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path, could be .rknn file')
    parser.add_argument('--target', type=str,
                        default=None, help='target RKNPU platform')
    parser.add_argument('--device_id', type=str,
                        default=None, help='device id')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Load RKNN model
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print('Load RKNN model \"{}\" failed!'.format(args.model_path))
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Initialize webcam capture and set resolution and format
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print('Failed to open webcam /dev/video0')
        rknn.release()
        exit(1)

    running = [True]

    def _handle_stop(signum, frame):
        running[0] = False
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print('--> Running model from webcam. Press Ctrl+C to stop')
    frame_count = 0 # Count of processed frames
    last_log_ts = time.time() # Timestamp of last log message
    try:
        while running[0]:
            # Read a frame from the webcam
            ret, img = cap.read()
            if not ret or img is None:
                continue

            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, (640, 640), 56) # Resize and letterbox the image to 640x640 with a gray background (value 56)
            infer_img = letterbox_img[..., ::-1]  # Convert BGR to RGB
            infer_img = np.expand_dims(infer_img, 0) # Add batch dimension for inference

            results = rknn.inference(inputs=[infer_img]) # Run inference on the preprocessed image and get the results

            outputs = [] # Initialize an empty list to store processed detection boxes
            keypoints = results[3] # Get the keypoints output from the model results
            
            for x in results[:3]: # Process the first three outputs of the model (feature maps at different scales)
                index, stride = 0, 0
                if x.shape[2] == 20: # If the feature map has a height of 20, it corresponds to the largest stride (32) and the last index for keypoints
                    stride = 32
                    index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2 # Calculate the starting index for keypoints based on the feature map size
                if x.shape[2] == 40: # If the feature map has a height of 40, it corresponds to a stride of 16 and the second index for keypoints
                    stride = 16
                    index = 20 * 4 * 20 * 4
                if x.shape[2] == 80: # If the feature map has a height of 80, it corresponds to the smallest stride (8) and the first index for keypoints
                    stride = 8
                    index = 0
                feature = x.reshape(1, 65, -1) # Reshape the feature map to have 65 channels (4 for bounding box, 1 for confidence, and 60 for keypoints) and flatten the spatial dimensions
                output = process(feature, keypoints, index, x.shape[3], x.shape[2], stride) # Process the feature map to decode bounding boxes and keypoints
                outputs = outputs + output
            predbox = NMS(outputs) # Apply Non-Maximum Suppression to filter overlapping detections

            for i in range(len(predbox)): # Loop through each detected bounding box and draw it on the original image
                xmin = int((predbox[i].xmin - offset_x) / aspect_ratio)
                ymin = int((predbox[i].ymin - offset_y) / aspect_ratio)
                xmax = int((predbox[i].xmax - offset_x) / aspect_ratio)
                ymax = int((predbox[i].ymax - offset_y) / aspect_ratio)
                classId = predbox[i].classId
                score = predbox[i].score
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Draw the bounding box on the image
                ptext = (xmin, ymin)
                title = CLASSES[classId] + "%.2f" % score

                cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Draw the class label and confidence score on the image
                keypoints = predbox[i].keypoint.reshape(-1, 3)  # keypoint [x, y, conf]
                keypoints[..., 0] = (keypoints[..., 0] - offset_x) / aspect_ratio
                keypoints[..., 1] = (keypoints[..., 1] - offset_y) / aspect_ratio

                for k, keypoint in enumerate(keypoints): # Loop through each keypoint and draw it on the image
                    x, y, conf = keypoint
                    color_k = [int(x) for x in kpt_color[k]]
                    if x != 0 and y != 0:
                        cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA) # Draw the keypoint as a filled circle on the image

                for k, sk in enumerate(skeleton): # Loop through each skeleton connection and draw it on the image
                    pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
                    pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))

                    conf1 = keypoints[(sk[0] - 1), 2]
                    conf2 = keypoints[(sk[1] - 1), 2]
                    if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0: # If either keypoint is not detected (coordinates are zero), skip drawing the skeleton connection
                        continue
                    cv2.line(img, pos1, pos2, [int(x) for x in limb_color[k]], thickness=2, lineType=cv2.LINE_AA) # Draw the skeleton connection as a line on the image

            cv2.imwrite('./result.jpg', img) # Save the processed image with bounding boxes and keypoints to a file named 'result.jpg'
            frame_count += 1

            now = time.time()
            if now - last_log_ts >= 2.0:
                print('saved ./result.jpg (frames: {})'.format(frame_count))
                last_log_ts = now
    finally:
        cap.release()
        rknn.release()
        print('Stopped. Last output saved to ./result.jpg')


