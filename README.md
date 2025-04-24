# RF-DETR Video Object Detection Pipeline

This project provides a pipeline for running real-time object detection on videos using the [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow Detection Transformer) model. It leverages the `rfdetr` and `supervision` Python packages to annotate videos with detected objects and save the results.

## Features
- Processes video files and annotates detected objects frame-by-frame
- Supports GPU acceleration (CUDA, MPS) and CPU fallback
- Configurable input/output paths, FPS, and video segment selection
- Uses COCO class labels for object annotation
- Optional BLIP-2 AI-powered entity descriptions (instead of COCO labels)
- Detailed logging with timing and performance metrics
- **NEW: Dataset generation with entity tracking across frames**
- **NEW: Advanced dataset visualizations and analytics**

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
   
   For BLIP-2 support, additional dependencies are required:
   ```bash
   pip install transformers torch Pillow
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
- `blip2_describer.py` — Module for generating AI descriptions for detected entities using BLIP-2
- `dataset_generator.py` — NEW: Module for tracking entities across frames and generating structured datasets
- `visualize_dataset.py` — NEW: Tool for generating visualizations from the entity tracking dataset

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
- `--blip2` — Use BLIP-2 to generate descriptions for each detected entity
- `--detection-threshold` — Detection confidence threshold (default: 0.5)
- `--prompt` — Custom prompt for BLIP-2 description (default: "Describe this object:")
- `--log-level` — Set logging level [DEBUG, INFO, WARNING, ERROR] (default: INFO)
- `--generate-dataset` — NEW: Generate a structured dataset with entity tracking
- `--dataset-output` — NEW: Path to save the generated dataset (default: auto-generated path in output directory)
- `--environment` — NEW: Environment label for the dataset (default: "unknown")
- `--iou-threshold` — NEW: IoU threshold for entity tracking between frames (default: 0.5)

Example:
```bash
python main.py --input ./input/tokyo_15min.mp4 --output ./output/annotated_tokyo.mp4 --fps 10 --seconds 60 --start 30
```

To use AI-powered descriptions instead of COCO class labels:
```bash
python main.py --input ./input/shinjuku.mp4 --output ./output/video_with_descriptions.mp4 --blip2
```

## Dataset Generation

The new dataset generation capability creates structured JSON datasets from processed videos, with entity tracking between frames. This is ideal for:

- Creating training data for computer vision models
- Analyzing object movement and relationships in videos
- Building knowledge graphs of visual scenes
- Creating annotated time series data

### Dataset Structure

The generated dataset is a JSON array containing frame-by-frame data:

```json
[
  {
    "frame_id": 1,
    "timestamp": "2024-08-24T09:00:00Z",
    "environment": "street",
    "entities": [
      {
        "entity_id": "person_0",
        "type": "person",
        "bbox": [100, 200, 300, 400],
        "confidence": 0.97,
        "description": "Woman in a red coat walking"
      },
      ...
    ],
    "relationships": [
      { "subject": "person_0", "predicate": "next_to", "object": "car_2" },
      ...
    ],
    "delta": {
      "new_entities": ["person_0", "car_2"],
      "updated_entities": [],
      "removed_entities": []
    },
    "previous_frame_id": null
  },
  ...
]
```

### Key Features

- **Entity Tracking**: Each entity gets a consistent ID across frames
- **Delta Information**: Track which entities are new, updated, or removed between frames
- **Spatial Relationships**: Automatically infers relationships between entities (next_to, contains, etc.)
- **Rich Descriptions**: Optional AI-generated descriptions using BLIP-2

### Dataset Generation

To generate a dataset with specific parameters:

```bash
python main.py --input ./input/street_scene.mp4 --output ./output/annotated.mp4 --generate-dataset --dataset-output ./output/street_data.json --environment street --blip2 --fps 5 --seconds 30
```

## Dataset Visualization

The project includes tools to visualize and analyze the entity tracking datasets. These visualizations help understand how entities move and interact throughout a video.

### Visualization Types

- **Entity Timeline**: Shows which entities are present in each frame
- **Entity Changes**: Visualizes new, updated, and removed entities between frames
- **Relationship Graphs**: Network graphs showing entity relationships for selected frames
- **Movement Animation**: Animated visualization of entity movement across frames
- **Entity Statistics**: Charts and statistics about entity types, lifespans, and relationships

### Generating Visualizations

Use the visualization tool with a generated dataset:

```bash
python visualize_dataset.py --dataset ./output/street_data.json --output-dir ./output/visualizations --all
```

You can also generate specific visualizations:

```bash
python visualize_dataset.py --dataset ./output/street_data.json --timeline --changes --statistics
```

## Model & References
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [Roboflow Blog: RF-DETR](https://blog.roboflow.com/rf-detr/)
- [supervision Python package](https://github.com/roboflow/supervision)
- [BLIP-2 on Hugging Face](https://huggingface.co/Salesforce/blip2-opt-2.7b)

## Notes
- The script will automatically use GPU (CUDA or Apple MPS) if available, otherwise it will run on CPU.
- The COCO class labels are used for annotation.
- For best performance, use a machine with a compatible GPU.
- BLIP-2 processing can be slow, especially on CPU. Use `--fps` to reduce the number of frames to process.
- Detailed logs are provided for timing and performance metrics during processing.
- The entity tracking uses IoU (Intersection over Union) to match entities between frames.
- Visualizations require matplotlib, networkx, and pandas packages.

## License
This project is for research and educational purposes. The RF-DETR model and weights are released under the Apache 2.0 license by Roboflow. 