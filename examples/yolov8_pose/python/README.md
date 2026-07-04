# YOLOv8 Pose Webcam HTTP Server

## Description

This demo runs YOLOv8 pose inference from a webcam using an RKNN model and serves live results over HTTP.

Instead of continuously writing `result.jpg`, it keeps the latest processed frame in memory and exposes:

- A browser viewer page
- A live MJPEG stream (works with VLC)
- A single-frame snapshot endpoint
- A health endpoint

Script file:

- `yolov8_pose_webcam_http_server.py`

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy
- RKNN runtime (`rknn.api` or `rknnlite.api`)
- A working webcam (default index: `0`)

## How To Run

From this folder:

```bash
cd rknn_model_zoo/examples/yolov8_pose/python
python yolov8_pose_webcam_http_server.py \
  --model_path ../model/yolov8n-pose.rknn \
  --target rk3588 \
  --host 0.0.0.0 \
  --port 8080
```

Replace values as needed:

- `--model_path`: path to your `.rknn` model
- `--target`: your Rockchip platform (for example `rk3588`)

## View The Result

Open in browser:

- `http://<device-ip>:8080/`

Open in VLC:

- `http://<device-ip>:8080/stream.mjpg`

Other endpoints:

- Latest frame: `http://<device-ip>:8080/snapshot.jpg`
- Health check: `http://<device-ip>:8080/healthz`

## Useful Options

```bash
python yolov8_pose_webcam_http_server.py --help
```

Common options:

- `--camera_index` (default: `0`)
- `--camera_width` (default: `640`)
- `--camera_height` (default: `480`)
- `--jpeg_quality` (default: `80`, range `1..100`)
- `--max_fps` (default: `0`, uncapped)

## Stop

Press `Ctrl+C` to stop. The script shuts down the HTTP server, releases the camera, and releases RKNN resources cleanly.
