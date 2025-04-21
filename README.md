# RF-DETR with BLIP-2 for Object Detection and Description

This project combines RF-DETR (Roboflow's DETR implementation) for object detection with BLIP-2 for generating natural language descriptions of the detected objects in videos.

## Features

- Object detection using RF-DETR models
- Natural language description of detected objects using BLIP-2
- Support for video processing with customizable parameters
- Optimized for Apple Silicon (MPS acceleration)

## System Requirements

- Python 3.9+ (Python 3.12 is supported)
- macOS (optimized for Apple Silicon, but will work on Intel Macs too)
- 16GB+ RAM recommended for BLIP-2 model

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Note: Some dependencies that cause issues on macOS with Apple Silicon are commented out in the requirements.txt file. The project works without them, but you might see warnings related to unsupported features.

### 3. Verify installation

Run the verification script to ensure all required dependencies are installed:

```bash
python verify_setup.py
```

## Usage

### Basic usage

```bash
python main.py --input your_video.mp4 --output output.mp4
```

### Enable BLIP-2 descriptions

```bash
python main.py --input your_video.mp4 --output output.mp4 --blip2
```

### Full options

```bash
python main.py \
  --input your_video.mp4 \
  --output output.mp4 \
  --blip2 \
  --prompt "Describe this object:" \
  --detection-threshold 0.5 \
  --fps 10 \
  --seconds 60 \
  --start 30 \
  --log-level INFO
```

## Parameters

- `--input`: Input video path (default: ./input/shinjuku.mp4)
- `--output`: Output video path (default: ./output/video.mp4)
- `--blip2`: Enable BLIP-2 descriptions for detected objects
- `--prompt`: Prompt for BLIP-2 (default: "Describe this object:")
- `--detection-threshold`: Detection confidence threshold (default: 0.5)
- `--fps`: Target FPS to process (default: original video FPS)
- `--seconds`: How many seconds of video to process (default: all)
- `--start`: Start time in seconds (default: 0)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Troubleshooting

### Installation issues on macOS Apple Silicon

If you encounter issues with some dependencies like `onnxsim` or `onnx_graphsurgeon`, they can be safely skipped for most use cases.

### Memory issues with BLIP-2

The BLIP-2 model requires a significant amount of memory. If you experience out-of-memory errors:

1. Reduce the input video resolution 
2. Process fewer frames by setting a lower `--fps` value
3. Consider disabling BLIP-2 with high-resolution videos

### GPU Acceleration

- On Apple Silicon Macs, the code automatically uses MPS (Metal Performance Shaders) for acceleration
- On systems with NVIDIA GPUs, CUDA will be used automatically if available

## License

This project uses components from:
- RF-DETR: Apache License 2.0
- BLIP-2: BSD 3-Clause License
- Supervision: Apache License 2.0 