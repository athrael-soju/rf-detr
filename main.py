import supervision as sv
from rfdetr.util.coco_classes import COCO_CLASSES
import argparse
import logging
import torch
import time
from rf_detr_runner import rf_detr_callback
import blip2_describer  # Import entire module instead of individual components
from dataset_generator import EntityTracker
import os
import datetime

def main():
    start_time = time.time()
    
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
    parser.add_argument('--input', type=str, default='./input/nyc.mp4', help='Input video path')
    parser.add_argument('--output', type=str, default='./output/video.mp4', help='Output video path')
    parser.add_argument('--blip2', action='store_true', help='Use BLIP-2 to generate descriptions for each detected entity')
    parser.add_argument('--detection-threshold', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--prompt', type=str, default="Describe this object:", help='Prompt for BLIP-2 description')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set logging level')
    parser.add_argument('--environment', type=str, default="unknown", help='Environment label for the dataset (e.g., home, office, street)')
    parser.add_argument('--dataset-output', type=str, default=None, help='Output path for the JSON dataset')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for tracking (default: 0.5)')
    parser.add_argument('--generate-dataset', action='store_true', help='Generate a dataset from the processed video')
    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s: %(message)s')

    # Log current settings
    logging.info(f"Starting video processing with the following settings:")
    logging.info(f"  Input: {args.input}")
    logging.info(f"  Output: {args.output}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Detection Threshold: {args.detection_threshold}")
    if args.fps is not None:
        logging.info(f"  Target FPS: {args.fps}")
    if args.seconds is not None:
        logging.info(f"  Processing Duration: {args.seconds}s")
    logging.info(f"  Start Time: {args.start}s")
    if args.blip2:
        logging.info(f"  Using BLIP-2 for entity description")
        logging.info(f"  BLIP-2 Prompt: '{args.prompt}'")
    if args.generate_dataset:
        logging.info(f"  Generating dataset with environment: '{args.environment}'")
    
    # Preload BLIP-2 model if needed
    if args.blip2:
        try:
            # Load the BLIP-2 model
            processor, model = blip2_describer.load_blip2_model(device=device)
            logging.info("BLIP-2 model loaded successfully")
            blip2_available = True
        except Exception as e:
            logging.error(f"Failed to load BLIP-2 model: {str(e)}")
            logging.warning("Falling back to COCO class labels")
            blip2_available = False
            args.blip2 = False
    else:
        blip2_available = False
    
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

    # Initialize entity tracker if dataset generation is enabled
    if args.generate_dataset:
        entity_tracker = EntityTracker(
            iou_threshold=args.iou_threshold,
            environment=args.environment
        )
        
        # Setup default dataset output path if not provided
        if args.dataset_output is None:
            input_filename = os.path.splitext(os.path.basename(args.input))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.dataset_output = f"./output/{input_filename}_{timestamp}_dataset.json"
            logging.info(f"Using default dataset output path: {args.dataset_output}")

    frames_generator = sv.get_video_frames_generator(
        source_path=args.input,
        stride=stride,
        start=start_frame,
        end=end_frame
    )

    total_frames_processed = 0
    total_entities_detected = 0
    detection_time = 0
    description_time = 0
    annotation_time = 0
    dataset_generation_time = 0
    
    with sv.VideoSink(target_path=args.output, video_info=output_video_info) as sink:
        for idx, frame in enumerate(frames_generator):
            if max_frames is not None and idx >= max_frames:
                break
                
            frame_start_time = time.time()
            frame_id = start_frame + (idx * stride)
            
            # Generate timestamp for this frame
            seconds_from_start = idx * stride / video_info.fps
            timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=seconds_from_start)).isoformat()
            
            # Run detection
            detection_start = time.time()
            from rf_detr_runner import model as rf_model
            detections = rf_model.predict(frame[:, :, ::-1].copy(), threshold=args.detection_threshold)
            detection_time += time.time() - detection_start
            total_entities_detected += len(detections.xyxy)
            
            # Log number of detections for this frame
            logging.info(f"Frame {idx}: Detected {len(detections.xyxy)} entities")
            
            # Process descriptions if BLIP-2 is enabled
            if args.blip2 and blip2_available:
                try:
                    description_start = time.time()
                    labels = blip2_describer.describe_entities_with_blip2(frame, detections, prompt_template=args.prompt)
                    description_time += time.time() - description_start
                    
                    # Debug log all the descriptions
                    logging.info(f"BLIP-2 generated {len(labels)} descriptions:")
                    for i, desc in enumerate(labels):
                        logging.info(f"  Description {i+1}: {desc}")
                except Exception as e:
                    logging.error(f"Error generating BLIP-2 descriptions: {str(e)}")
                    logging.warning("Falling back to COCO class labels for this frame")
                    # Use COCO class labels as fallback
                    labels = [
                        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
            else:
                # Use COCO class labels
                labels = [
                    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
                
            # Generate dataset if enabled
            if args.generate_dataset:
                dataset_start_time = time.time()
                
                # Get clean descriptions without confidence values if using BLIP2
                if args.blip2 and blip2_available:
                    clean_descriptions = labels
                else:
                    # Strip confidence values from COCO labels
                    clean_descriptions = [
                        COCO_CLASSES[class_id]
                        for class_id in detections.class_id
                    ]
                
                # Process frame for dataset
                frame_data = entity_tracker.process_frame(
                    frame_id=frame_id,
                    detections=detections,
                    descriptions=clean_descriptions,
                    timestamp=timestamp
                )
                
                dataset_generation_time += time.time() - dataset_start_time
                
                if idx % 10 == 0:
                    logging.info(f"Generated dataset entry for frame {idx} with {len(frame_data['entities'])} entities")
                    logging.info(f"  New: {len(frame_data['delta']['new_entities'])}, " + 
                               f"Updated: {len(frame_data['delta']['updated_entities'])}, " +
                               f"Removed: {len(frame_data['delta']['removed_entities'])}")
            
            # Annotate frame and write to output
            annotation_start = time.time()
            annotated_frame = frame.copy()
            
            # Use different color for BLIP-2 descriptions to make them stand out
            box_color = sv.Color.RED if args.blip2 and blip2_available else sv.Color.GREEN
            
            annotated_frame = sv.BoxAnnotator(thickness=2, color=box_color).annotate(annotated_frame, detections)
            
            # Set up enhanced label annotator for better visibility
            text_scale = 0.6
            text_thickness = 2
            text_padding = 10
            text_color = sv.Color.WHITE

            if args.blip2 and blip2_available:
                # For BLIP-2 descriptions, use more visible styling
                text_scale = 0.8
                text_thickness = 2
                text_padding = 15
                box_color = sv.Color.BLUE  # Different color for BLIP-2 text boxes
            
            label_annotator = sv.LabelAnnotator(
                text_scale=text_scale,
                text_thickness=text_thickness, 
                text_padding=text_padding,
                color=box_color,
                text_color=text_color
            )
            
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # Debug: confirm annotations were added
            if idx % 10 == 0:
                if len(labels) > 0:
                    logging.info(f"Added {len(labels)} label annotations to frame {idx}")
                else:
                    logging.warning(f"No labels were added to frame {idx}")
            
            sink.write_frame(annotated_frame)
            annotation_time += time.time() - annotation_start
            
            frame_time = time.time() - frame_start_time
            total_frames_processed += 1
            
            # Log progress
            if idx % 10 == 0:
                logging.info(f"Processed frame {idx} ({total_frames_processed} total) - " +
                           f"Found {len(detections.xyxy)} entities - " +
                           f"Frame time: {frame_time:.2f}s")
    
    # Save dataset if enabled
    if args.generate_dataset:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.dataset_output), exist_ok=True)
        entity_tracker.save_dataset(args.dataset_output)
        logging.info(f"Dataset saved to {args.dataset_output}")
    
    # Log summary statistics
    total_time = time.time() - start_time
    avg_time_per_frame = total_time / total_frames_processed if total_frames_processed > 0 else 0
    avg_entities_per_frame = total_entities_detected / total_frames_processed if total_frames_processed > 0 else 0
    
    logging.info(f"Processing complete. Output saved to {args.output}")
    logging.info(f"Total processing time: {total_time:.2f}s")
    logging.info(f"Frames processed: {total_frames_processed}")
    logging.info(f"Total entities detected: {total_entities_detected}")
    logging.info(f"Average entities per frame: {avg_entities_per_frame:.2f}")
    logging.info(f"Average time per frame: {avg_time_per_frame:.2f}s")
    time_breakdown = f"Time breakdown - Detection: {detection_time:.2f}s, " + \
                    f"Description: {description_time:.2f}s, " + \
                    f"Annotation: {annotation_time:.2f}s"
    
    if args.generate_dataset:
        time_breakdown += f", Dataset Generation: {dataset_generation_time:.2f}s"
    
    logging.info(time_breakdown)

if __name__ == "__main__":
    main()