import argparse
import json
import platform
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# Import RKNN API (try main library first, fallback to lite version if unavailable)
try:
    from rknn.api import RKNN
except ImportError:
    print('import rknn failed, try to import rknnlite')
    from rknnlite.api import RKNNLite as RKNN


CLASSES = ['person'] # List of object classes to detect

nmsThresh = 0.4 # NMS (Non-Maximum Suppression) threshold for filtering overlapping detections
objectThresh = 0.5 # Confidence threshold for object detection


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest_jpeg = None
        self.frame_id = 0
        self.frame_count = 0
        self.client_count = 0
        self.started_ts = time.time()


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

    # Calculate the adjusted image size.
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling.
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it.
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
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    out = []
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0, c, (h * model_w) + w] > objectThresh:
                    xywh_ = xywh[0, :, (h * model_w) + w]  # [1,64,1]
                    xywh_ = xywh_.reshape(1, 4, 16, 1)
                    data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
                    xywh_ = softmax(xywh_, 2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp = xywh_.copy()
                    xywh_temp[0] = (w + 0.5) - xywh_[0]
                    xywh_temp[1] = (h + 0.5) - xywh_[1]
                    xywh_temp[2] = (w + 0.5) + xywh_[2]
                    xywh_temp[3] = (h + 0.5) + xywh_[3]

                    xywh_[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
                    xywh_[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
                    xywh_[2] = (xywh_temp[2] - xywh_temp[0])
                    xywh_[3] = (xywh_temp[3] - xywh_temp[1])
                    xywh_ = xywh_ * stride

                    xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint = keypoints[..., (h * model_w) + w + index]
                    keypoint[..., 0:2] = keypoint[..., 0:2] // 1
                    box = DetectBox(c, conf[0, c, (h * model_w) + w], xmin, ymin, xmax, ymax, keypoint)
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

class StreamHandler(BaseHTTPRequestHandler):
    shared_state = None
    stop_event = None

    def _write_bytes(self, status, body, content_type='text/plain; charset=utf-8'):
        """Send HTTP response with headers and body.
        
        Args:
            status: HTTP status code (e.g., HTTPStatus.OK).
            body: Response body as bytes.
            content_type: MIME type of the response body. Defaults to 'text/plain; charset=utf-8'.
        """
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Handle HTTP GET requests for various endpoints.
        
        Supports the following endpoints:
        - '/': Returns an HTML page with embedded MJPEG stream viewer
        - '/healthz': Returns JSON health status including client count, frame count, and uptime
        - '/snapshot.jpg': Returns the latest JPEG frame as an image
        - '/stream.mjpg': Streams live MJPEG video feed to the client
        """
        if self.path == '/':
            page = (
                '<!doctype html>'
                '<html><head><meta charset="utf-8"><title>YOLOv8 Pose Stream</title></head>'
                '<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;">'
                '<div style="padding:12px">'
                '<h3 style="margin:0 0 8px 0">YOLOv8 Pose MJPEG Stream</h3>'
                '<p style="margin:0 0 10px 0">Open in VLC: http://HOST:PORT/stream.mjpg</p>'
                '<img src="/stream.mjpg" style="max-width:100%;height:auto;border:1px solid #333"/>'
                '</div></body></html>'
            ).encode('utf-8')
            self._write_bytes(HTTPStatus.OK, page, content_type='text/html; charset=utf-8')
            return

        if self.path == '/healthz':
            with self.shared_state.lock:
                payload = {
                    'status': 'ok',
                    'clients': self.shared_state.client_count,
                    'frames': self.shared_state.frame_count,
                    'uptime_sec': round(time.time() - self.shared_state.started_ts, 3),
                    'has_frame': self.shared_state.latest_jpeg is not None,
                }
            self._write_bytes(HTTPStatus.OK, json.dumps(payload).encode('utf-8'), content_type='application/json')
            return

        if self.path == '/snapshot.jpg':
            with self.shared_state.lock:
                jpeg = self.shared_state.latest_jpeg
            if jpeg is None:
                self._write_bytes(HTTPStatus.SERVICE_UNAVAILABLE, b'no frame available')
                return
            self._write_bytes(HTTPStatus.OK, jpeg, content_type='image/jpeg')
            return

        if self.path == '/stream.mjpg':
            self.send_response(HTTPStatus.OK)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            with self.shared_state.lock:
                self.shared_state.client_count += 1

            last_seen_id = -1
            try:
                while not self.stop_event.is_set():
                    with self.shared_state.cond:
                        self.shared_state.cond.wait_for(
                            lambda: self.stop_event.is_set() or self.shared_state.frame_id != last_seen_id,
                            timeout=1.0,
                        )
                        if self.stop_event.is_set():
                            break
                        jpeg = self.shared_state.latest_jpeg
                        frame_id = self.shared_state.frame_id

                    if jpeg is None:
                        continue

                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(('Content-Length: %d\r\n\r\n' % len(jpeg)).encode('ascii'))
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    last_seen_id = frame_id
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                with self.shared_state.lock:
                    self.shared_state.client_count -= 1
            return

        self._write_bytes(HTTPStatus.NOT_FOUND, b'not found')

    def log_message(self, fmt, *args):
        # Keep output compact; main loop already prints periodic status.
        return


def create_http_server(host, port, shared_state, stop_event):
    """
    Create and configure an HTTP server for streaming video frames.
    
    Sets up a multi-threaded HTTP server that streams JPEG frames over HTTP.
    The server uses a shared state object to coordinate frame updates between
    the main processing thread and client connection handlers.
    
    Args:
        host: Hostname or IP address to bind the server to.
        port: Port number for the HTTP server.
        shared_state: Shared state object containing frame data and synchronization primitives.
        stop_event: Threading event to signal server shutdown.
    
    Returns:
        ThreadingHTTPServer: Configured HTTP server instance ready to serve requests.
    """
    StreamHandler.shared_state = shared_state
    StreamHandler.stop_event = stop_event
    server = ThreadingHTTPServer((host, port), StreamHandler)
    server.daemon_threads = True
    return server


def draw_pose(img, predbox, aspect_ratio, offset_x, offset_y):
    """
    Draw pose detection results on an image.
    
    Draws bounding boxes, keypoints, and skeleton connections for detected poses.
    
    Args:
        img: Input image to draw on (modified in-place).
        predbox: List of prediction objects containing bounding box and keypoint data.
        aspect_ratio: Scaling factor for coordinate adjustment.
        offset_x: X-axis offset for coordinate transformation.
        offset_y: Y-axis offset for coordinate transformation.
    
    Returns:
        None. The input image is modified in-place.
    """
    for i in range(len(predbox)): # Loop through each detected bounding box and draw it on the image
        # Transform bounding box coordinates from model space to image space
        xmin = int((predbox[i].xmin - offset_x) / aspect_ratio)
        ymin = int((predbox[i].ymin - offset_y) / aspect_ratio)
        xmax = int((predbox[i].xmax - offset_x) / aspect_ratio)
        ymax = int((predbox[i].ymax - offset_y) / aspect_ratio)
        
        classId = predbox[i].classId
        score = predbox[i].score
        ptext = (xmin, ymin)
        title = CLASSES[classId] + '%.2f' % score
        
        # Transform keypoint coordinates from model space to image space
        keypoints = predbox[i].keypoint.reshape(-1, 3)
        keypoints[..., 0] = (keypoints[..., 0] - offset_x) / aspect_ratio
        keypoints[..., 1] = (keypoints[..., 1] - offset_y) / aspect_ratio
        
        # Draw the bounding box on the image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Draw the class label and confidence score on the image
        cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        for k, keypoint in enumerate(keypoints): # Loop through each keypoint and draw it on the image
            x, y, conf = keypoint
            color_k = [int(x) for x in kpt_color[k]]
            if x != 0 and y != 0:
                # Draw a circle at the keypoint location on the image
                cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

        for k, sk in enumerate(skeleton): # Loop through each skeleton connection and draw it on the image
            pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
            pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))

            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0: # If either keypoint is not detected (coordinates are zero), skip drawing the skeleton connection
                continue
            # Draw a line between the two keypoints to represent the limb/bone on the image
            cv2.line(img, pos1, pos2, [int(x) for x in limb_color[k]], thickness=2, lineType=cv2.LINE_AA)


def resolve_runtime_target(target):
    """
    Resolve the runtime target, accounting for native Linux ARM64 systems.
    
    On native Linux ARM64 (aarch64 or arm64) systems, this function ignores
    the provided target parameter and returns None to use the local RKNN Lite runtime
    instead of a remote target.
    
    Args:
        target: The specified runtime target, or None.
    
    Returns:
        None if running on native Linux ARM64, otherwise returns the original target.
    """
    if target and platform.system() == 'Linux' and platform.machine().lower() in ('aarch64', 'arm64'):
        print('Native Linux Arm64 detected; ignoring --target and using the local RKNN Lite runtime.')
        return None
    return target


def infer_and_stream(args):
    """
    Load an RKNN pose detection model and perform inference on webcam frames with HTTP streaming.
    
    This function captures frames from a webcam, runs YOLOv8 pose detection inference using RKNN,
    draws pose keypoints and skeletal connections on the frames, and streams the results via HTTP
    in MJPEG format. Handles graceful shutdown via Ctrl+C.
    
    Args:
        args: Argument namespace containing:
            - model_path (str): Path to the RKNN model file (.rknn)
            - target (str): Target RKNPU platform (None for local runtime)
            - device_id (str): Device ID for RKNN runtime
            - host (str): HTTP server bind address (default: '0.0.0.0')
            - port (int): HTTP server bind port (default: 8080)
            - jpeg_quality (int): JPEG compression quality 1-100 (default: 80)
            - max_fps (float): Maximum output FPS; 0 for unlimited (default: 0)
            - camera_index (int): OpenCV camera device index (default: 0)
            - camera_width (int): Camera capture width (default: 640)
            - camera_height (int): Camera capture height (default: 480)
    
    Returns:
        int: 0 on successful completion, non-zero error codes on failure.
    """
    stop_event = threading.Event()
    shared_state = SharedState()

    def _handle_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # Create RKNN object.
    rknn = RKNN(verbose=False)

    # Load RKNN model.
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print('Load RKNN model "{}" failed!'.format(args.model_path))
        return ret
    print('done')

    # Init runtime environment.
    print('--> Init runtime environment')
    runtime_target = resolve_runtime_target(args.target)
    ret = rknn.init_runtime(target=runtime_target, device_id=args.device_id)
    if ret != 0:
        print('Init runtime environment failed!')
        rknn.release()
        return ret
    print('done')

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print('Failed to open webcam /dev/video{}'.format(args.camera_index))
        rknn.release()
        return 1

    http_server = create_http_server(args.host, args.port, shared_state, stop_event)
    server_thread = threading.Thread(target=http_server.serve_forever, kwargs={'poll_interval': 0.5})
    server_thread.daemon = True
    server_thread.start()

    print('--> Running model from webcam with HTTP streaming. Press Ctrl+C to stop')
    print('--> Live page: http://{}:{}/'.format(args.host, args.port))
    print('--> MJPEG URL: http://{}:{}/stream.mjpg'.format(args.host, args.port))

    frame_count_window = 0
    last_log_ts = time.time()
    last_frame_ts = 0.0

    try:
        while not stop_event.is_set():
            ret, img = cap.read()
            if not ret or img is None:
                continue

            # Optional throttle can reduce network and CPU load.
            if args.max_fps > 0:
                now = time.time()
                min_interval = 1.0 / args.max_fps
                elapsed = now - last_frame_ts
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_frame_ts = time.time()

            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, (640, 640), 56)
            infer_img = letterbox_img[..., ::-1]  # BGR2RGB
            infer_img = np.expand_dims(infer_img, 0)  # add batch dim

            results = rknn.inference(inputs=[infer_img])

            outputs = []
            keypoints = results[3]
            for x in results[:3]:
                index, stride = 0, 0
                if x.shape[2] == 20:
                    stride = 32
                    index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
                if x.shape[2] == 40:
                    stride = 16
                    index = 20 * 4 * 20 * 4
                if x.shape[2] == 80:
                    stride = 8
                    index = 0
                feature = x.reshape(1, 65, -1)
                output = process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
                outputs = outputs + output
            predbox = NMS(outputs)

            draw_pose(img, predbox, aspect_ratio, offset_x, offset_y)

            ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                continue
            jpeg = enc.tobytes()

            with shared_state.cond:
                shared_state.latest_jpeg = jpeg
                shared_state.frame_id += 1
                shared_state.frame_count += 1
                shared_state.cond.notify_all()

            frame_count_window += 1
            now = time.time()
            if now - last_log_ts >= 2.0:
                with shared_state.lock:
                    clients = shared_state.client_count
                    total = shared_state.frame_count
                fps = frame_count_window / (now - last_log_ts)
                # print('streaming fps={:.2f}, clients={}, total_frames={}'.format(fps, clients, total))
                frame_count_window = 0
                last_log_ts = now
    finally:
        stop_event.set()
        http_server.shutdown()
        http_server.server_close()
        server_thread.join(timeout=2.0)

        cap.release()
        rknn.release()
        print('Stopped. HTTP stream terminated cleanly.')

    return 0


def parse_args():
    """
    Parse and return command-line arguments for the YOLOv8 Pose HTTP streaming application.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - model_path (str): Path to the RKNN model file (required)
            - target (str): Target RKNPU platform (optional)
            - device_id (str): Device ID for the RKNPU (optional)
            - host (str): HTTP server bind host (default: '0.0.0.0')
            - port (int): HTTP server bind port (default: 8080)
            - jpeg_quality (int): JPEG quality for stream output 1-100 (default: 80)
            - max_fps (float): Maximum FPS limit; 0 means uncapped (default: 0.0)
            - camera_index (int): Camera device index for OpenCV (default: 0)
            - camera_width (int): Camera capture width in pixels (default: 640)
            - camera_height (int): Camera capture height in pixels (default: 480)
    """
    parser = argparse.ArgumentParser(description='Yolov8 Pose Webcam HTTP Streaming Demo', add_help=True)
    parser.add_argument('--model_path', type=str, required=True, help='model path, should be .rknn file')
    parser.add_argument('--target', type=str, default=None, help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')

    parser.add_argument('--host', type=str, default='0.0.0.0', help='HTTP bind host')
    parser.add_argument('--port', type=int, default=8080, help='HTTP bind port')
    parser.add_argument('--jpeg_quality', type=int, default=80, help='JPEG quality for stream (1-100)')
    parser.add_argument('--max_fps', type=float, default=0.0, help='limit output FPS; 0 means uncapped')

    parser.add_argument('--camera_index', type=int, default=0, help='camera device index for OpenCV')
    parser.add_argument('--camera_width', type=int, default=640, help='camera capture width')
    parser.add_argument('--camera_height', type=int, default=480, help='camera capture height')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.port <= 0 or args.port > 65535:
        raise ValueError('port must be in range 1..65535')
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError('jpeg_quality must be in range 1..100')

    exit_code = infer_and_stream(args)
    raise SystemExit(exit_code)
