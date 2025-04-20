# RF-DETR Video Object Detection Pipeline

This project provides a pipeline for running real-time object detection on videos using the [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow Detection Transformer) model. It leverages the `rfdetr` and `supervision` Python packages to annotate videos with detected objects and save the results.

## Features
- Processes video files and annotates detected objects frame-by-frame
- Supports GPU acceleration (CUDA, MPS) and CPU fallback
- Configurable input/output paths, FPS, and video segment selection
- Uses COCO class labels for object annotation

## Installation

1. **Clone this repository** (if applicable):
   ```bash
   git clone https://github.com/athrael-soju/rf-detr
   cd rf-detr
   ```

2. **Install dependencies:**
   Ensure you have Python 3.9 or newer. Install required packages:
   ```bash
   pip install -r requirements.txt
   # Or install manually:
   pip install rfdetr supervision
   ```
   The `rfdetr` package is available on PyPI. For the latest features, you can also install from source:
   ```bash
   pip install git+https://github.com/roboflow/rf-detr.git
   ```

3. **Download a pre-trained checkpoint:**
   Place the `rf-detr-base.pth` checkpoint in the project root if you want to use a custom checkpoint. By default, the model will use the built-in COCO checkpoint.

## Directory Structure

- `main.py` — Main script for video processing (imports detection logic from `rf_detr_runner.py`)
- `check_gpu.py` — Utility to check GPU availability
- `input/` — Place your input video files here (e.g., `shinjuku.mp4`, `tokyo_15min.mp4`)
- `output/` — Processed/annotated videos will be saved here (e.g., `video.mp4`)
- `logs/` — Log files (if any)
- `requirements.txt` — Python dependencies
- `rf_detr_runner.py` — Contains the RF-DETR model instantiation and the callback function for frame annotation

## Usage

Run the main script from the command line:

```bash
python main.py --input ./input/shinjuku.mp4 --output ./output/video.mp4
```

The detection and annotation logic is now modularized in `rf_detr_runner.py`. If you want to use the detection callback or model in other scripts, simply import from `rf_detr_runner.py`:

```python
from rf_detr_runner import rf_detr_callback
# or import the model directly if needed
```

### Optional Arguments
- `--fps` — Target FPS to process (default: original video FPS)
- `--seconds` — How many seconds of video to process (default: all)
- `--start` — Start time in seconds (default: 0)
- `--input` — Input video path (default: `./input/shinjuku.mp4`)
- `--output` — Output video path (default: `./output/video.mp4`)

Example:
```bash
python main.py --input ./input/tokyo_15min.mp4 --output ./output/annotated_tokyo.mp4 --fps 10 --seconds 60 --start 30
```

## Model & References
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [Roboflow Blog: RF-DETR](https://blog.roboflow.com/rf-detr/)
- [supervision Python package](https://github.com/roboflow/supervision)

## Notes
- The script will automatically use GPU (CUDA or Apple MPS) if available, otherwise it will run on CPU.
- The COCO class labels are used for annotation.
- For best performance, use a machine with a compatible GPU.

## License
This project is for research and educational purposes. The RF-DETR model and weights are released under the Apache 2.0 license by Roboflow. 