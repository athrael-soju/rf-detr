import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import logging
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_blip2_model(device=None):
    """
    Load the BLIP-2 model.
    
    Args:
        device (str): Device to load the model on
    
    Returns:
        processor, model: The BLIP-2 processor and model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Loading BLIP-2 processor...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    logging.info(f"BLIP-2 processor loaded successfully")
    
    logging.info(f"Loading BLIP-2 model on {device}...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map="auto" if device == 'cuda' else None
    )
    logging.info("BLIP-2 model loaded successfully")
    
    return processor, model

# Initialize processor and model with default settings
logging.info("Initializing BLIP-2 model and processor with default settings...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor, model = load_blip2_model(device=device)

def describe_entities_with_blip2(frame: np.ndarray, detections, prompt_template: str = "Describe this object:"):
    """
    Given a frame (numpy array, BGR) and a Detections object (from supervision),
    crop each bounding box and generate a description using BLIP-2.
    Returns a list of descriptions (one per detection).
    
    Args:
        frame (np.ndarray): The video frame (BGR format)
        detections: Supervision Detections object containing bounding boxes
        prompt_template (str): The prompt to use for BLIP-2
        
    Returns:
        list: Descriptions for each detected entity
    """
    descriptions = []
    start_time = time.time()
    num_entities = len(detections.xyxy)
    
    if num_entities == 0:
        logging.info(f"No entities detected in frame, processing time: {time.time() - start_time:.2f}s")
        return descriptions
    
    logging.info(f"Starting to process {num_entities} entities with BLIP-2...")
    
    for i, box in enumerate(detections.xyxy):
        box_start_time = time.time()
        logging.info(f"Processing entity {i+1}/{num_entities}...")
        
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Add boundary checks
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logging.warning(f"Invalid box dimensions: ({x1}, {y1}, {x2}, {y2}). Skipping entity.")
                descriptions.append("Invalid bounding box")
                continue
                
            crop = frame[y1:y2, x1:x2, ::-1]  # Convert BGR to RGB
            logging.debug(f"Cropped box size: {crop.shape}")
            
            pil_image = Image.fromarray(crop)
            
            logging.debug(f"Preparing input for BLIP-2 using prompt: '{prompt_template}'")
            inputs = processor(pil_image, prompt_template, return_tensors="pt").to(device=model.device, dtype=next(model.parameters()).dtype)
            
            logging.debug(f"Generating description...")
            out = model.generate(**inputs, max_length=100)  # Limit length for better display
            
            # Get raw description
            raw_description = processor.decode(out[0], skip_special_tokens=True).strip()
            
            # Remove prompt from description if it appears at the beginning
            description = raw_description
            if prompt_template and description.startswith(prompt_template):
                description = description[len(prompt_template):].strip()
            
            # Format description for better display on video
            if len(description) > 100:
                # Truncate very long descriptions for better visibility on video
                description = description[:100] + "..."
                
            # Make sure we actually have a description
            if not description or description.isspace():
                description = "Unidentified object"
                
            descriptions.append(description)
            
            logging.info(f"Entity {i+1}/{num_entities} described in {time.time() - box_start_time:.2f}s: {description}")
            
        except Exception as e:
            logging.error(f"Error processing entity {i+1}: {str(e)}")
            descriptions.append(f"Error: {str(e)}")
    
    processing_time = time.time() - start_time
    avg_time_per_entity = processing_time / num_entities if num_entities > 0 else 0
    logging.info(f"Processed {num_entities} entities in {processing_time:.2f}s (avg: {avg_time_per_entity:.2f}s per entity)")
    
    return descriptions 