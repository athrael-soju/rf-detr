import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import argparse
import logging
import torch

model = RFDETRBase()

def callback(frame, index):
    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)
        
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

def main():
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info('CUDA is available. Using GPU.')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logging.info('MPS is available. Using Apple Silicon GPU.')
    else:
        device = 'cpu'
        logging.info('No GPU found. Using CPU.')

    parser = argparse.ArgumentParser(description="Process a video with detection and annotation.")
    parser.add_argument('--fps', type=float, default=None, help='Target FPS to process (default: original video FPS)')
    parser.add_argument('--seconds', type=float, default=None, help='How many seconds of video to process (default: all)')
    parser.add_argument('--start', type=float, default=0, help='Start time in seconds (default: 0)')
    parser.add_argument('--input', type=str, default='./input/shinjuku.mp4', help='Input video path')
    parser.add_argument('--output', type=str, default='./output/video.mp4', help='Output video path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    video_info = sv.VideoInfo.from_video_path(args.input)
    logging.info(f"Loaded video: {args.input} ({video_info.width}x{video_info.height}, {video_info.fps} FPS, {video_info.total_frames} frames)")

    # Calculate stride for FPS
    stride = 1
    if args.fps is not None and args.fps > 0:
        stride = max(1, round(video_info.fps / args.fps))
        logging.info(f"Processing every {stride} frame(s) to achieve ~{args.fps} FPS")
    else:
        logging.info(f"Processing at original FPS: {video_info.fps}")

    # Calculate start and end frames
    start_frame = int(args.start * video_info.fps)
    if args.seconds is not None and args.seconds > 0:
        max_frames = int(args.seconds * video_info.fps // stride)
        end_frame = start_frame + max_frames * stride
        logging.info(f"Processing {args.seconds} seconds from {args.start}s (frames {start_frame} to {end_frame})")
    else:
        end_frame = video_info.total_frames
        max_frames = None
        logging.info(f"Processing from {args.start}s (frame {start_frame}) to end of video")

    # Prepare output video info
    if args.fps is not None and args.fps > 0:
        output_fps = args.fps
    else:
        output_fps = video_info.fps
    output_video_info = sv.VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=output_fps,
        total_frames=None
    )

    frames_generator = sv.get_video_frames_generator(
        source_path=args.input,
        stride=stride,
        start=start_frame,
        end=end_frame
    )

    with sv.VideoSink(target_path=args.output, video_info=output_video_info) as sink:
        for idx, frame in enumerate(frames_generator):
            if max_frames is not None and idx >= max_frames:
                break
            if idx % 10 == 0:
                logging.info(f"Processing frame {idx}")
            annotated = callback(frame, idx)
            sink.write_frame(annotated)
        logging.info(f"Processing complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()