import json
import numpy as np
import os
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
import supervision as sv
from rfdetr.util.coco_classes import COCO_CLASSES
import torch
from collections import defaultdict
import uuid

class EntityTracker:
    """Track entities across video frames with unique IDs and compute deltas between frames"""
    
    def __init__(self, iou_threshold: float = 0.5, environment: str = "unknown"):
        """
        Initialize the entity tracker
        
        Args:
            iou_threshold: Threshold for considering entity matches between frames
            environment: Default environment tag for all frames
        """
        self.frame_data = []
        self.last_frame_entities = {}  # entity_id -> entity_data mapping from previous frame
        self.iou_threshold = iou_threshold
        self.environment = environment
        self.next_entity_id = defaultdict(int)  # Separate counters for each entity type
        
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1: First box in format [x1, y1, x2, y2]
            box2: Second box in format [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        if area1 <= 0 or area2 <= 0:
            return 0.0
            
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def generate_entity_id(self, entity_type: str) -> str:
        """
        Generate a unique ID for a new entity
        
        Args:
            entity_type: The type of entity (e.g., 'person', 'car')
            
        Returns:
            A unique entity ID string
        """
        entity_id = f"{entity_type}_{self.next_entity_id[entity_type]}"
        self.next_entity_id[entity_type] += 1
        return entity_id
    
    def match_entities(self, 
                      current_detections: sv.Detections, 
                      current_descriptions: List[str]) -> Tuple[Dict[str, Dict], Set[str], Set[str], Set[str]]:
        """
        Match entities from current frame with previous frame
        
        Args:
            current_detections: Detections from current frame
            current_descriptions: Descriptions for each detection
            
        Returns:
            Tuple containing:
                - Dict of current entities
                - Set of new entity IDs
                - Set of updated entity IDs
                - Set of removed entity IDs
        """
        current_entities = {}
        new_entity_ids = set()
        updated_entity_ids = set()
        removed_entity_ids = set()
        
        # Track which previous entities have been matched
        matched_prev_entities = set()
        
        # Process each detection in current frame
        for i, (bbox, class_id, confidence) in enumerate(zip(
            current_detections.xyxy, 
            current_detections.class_id, 
            current_detections.confidence
        )):
            # Get entity type from COCO classes
            entity_type = COCO_CLASSES[class_id].lower()
            description = current_descriptions[i] if i < len(current_descriptions) else ""
            
            # Find best match in previous frame
            best_match_id = None
            best_match_iou = 0
            
            for prev_id, prev_entity in self.last_frame_entities.items():
                # Only match with same type
                if prev_entity["type"] == entity_type:
                    iou = self.calculate_iou(bbox, prev_entity["bbox"])
                    if iou > self.iou_threshold and iou > best_match_iou:
                        best_match_id = prev_id
                        best_match_iou = iou
            
            # If match found, use existing ID, otherwise generate new ID
            if best_match_id:
                entity_id = best_match_id
                matched_prev_entities.add(best_match_id)
                updated_entity_ids.add(entity_id)
            else:
                entity_id = self.generate_entity_id(entity_type)
                new_entity_ids.add(entity_id)
            
            # Store entity data
            current_entities[entity_id] = {
                "entity_id": entity_id,
                "type": entity_type,
                "bbox": bbox.tolist(),  # Convert numpy array to list
                "confidence": float(confidence),  # Convert numpy float to Python float
                "description": description
            }
        
        # Find removed entities (in previous frame but not current)
        for prev_id in self.last_frame_entities:
            if prev_id not in matched_prev_entities:
                removed_entity_ids.add(prev_id)
        
        return current_entities, new_entity_ids, updated_entity_ids, removed_entity_ids
    
    def infer_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """
        Infer spatial relationships between entities
        
        Args:
            entities: Dictionary of entity_id -> entity_data
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # List of entity IDs for iteration
        entity_ids = list(entities.keys())
        
        # Compare each pair of entities
        for i, subject_id in enumerate(entity_ids):
            subject = entities[subject_id]
            subject_box = np.array(subject["bbox"])
            
            for j in range(i+1, len(entity_ids)):
                object_id = entity_ids[j]
                obj = entities[object_id]
                object_box = np.array(obj["bbox"])
                
                # Calculate center points
                subject_center = [(subject_box[0] + subject_box[2])/2, (subject_box[1] + subject_box[3])/2]
                object_center = [(object_box[0] + object_box[2])/2, (object_box[1] + object_box[3])/2]
                
                # Calculate distance between centers
                distance = np.sqrt((subject_center[0] - object_center[0])**2 + 
                                  (subject_center[1] - object_center[1])**2)
                
                # Check IoU for overlap/containment
                iou = self.calculate_iou(subject_box, object_box)
                
                # Determine relationship type based on spatial arrangement
                # These are simplified heuristics - could be more sophisticated
                if iou > 0.7:  # High overlap
                    predicate = "contains" if subject_box[2] - subject_box[0] > object_box[2] - object_box[0] else "inside"
                elif iou > 0.1:  # Some overlap
                    predicate = "overlaps"
                elif distance < 200:  # Close proximity (could be tuned based on frame size)
                    predicate = "next_to"
                else:
                    predicate = "near"
                
                # Add both directions of the relationship
                relationships.append({
                    "subject": subject_id,
                    "predicate": predicate, 
                    "object": object_id
                })
        
        return relationships
    
    def process_frame(self, 
                     frame_id: int,
                     detections: sv.Detections,
                     descriptions: List[str],
                     timestamp: Optional[str] = None) -> Dict:
        """
        Process a video frame and update the dataset
        
        Args:
            frame_id: Unique identifier for this frame
            detections: Supervision detections for this frame
            descriptions: List of descriptions for each detection
            timestamp: Optional timestamp string (ISO format)
            
        Returns:
            The frame data dictionary
        """
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        # Match current entities with previous frame
        current_entities, new_entity_ids, updated_entity_ids, removed_entity_ids = self.match_entities(
            detections, descriptions
        )
        
        # Convert entities dictionary to list for JSON format
        entities_list = list(current_entities.values())
        
        # Generate relationships between entities
        relationships = self.infer_relationships(current_entities)
        
        # Create frame data
        frame_data = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "environment": self.environment,
            "entities": entities_list,
            "relationships": relationships,
            "delta": {
                "new_entities": list(new_entity_ids),
                "updated_entities": list(updated_entity_ids),
                "removed_entities": list(removed_entity_ids)
            },
            "previous_frame_id": None if frame_id == 1 else frame_id - 1
        }
        
        # Store frame data and update last frame entities
        self.frame_data.append(frame_data)
        self.last_frame_entities = current_entities
        
        return frame_data
    
    def save_dataset(self, output_path: str) -> None:
        """
        Save the accumulated dataset to a JSON file
        
        Args:
            output_path: Path to save the output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.frame_data, f, indent=2)
        
        logging.info(f"Dataset saved to {output_path} with {len(self.frame_data)} frames")
        
    def get_dataset(self) -> List[Dict]:
        """
        Get the current dataset
        
        Returns:
            List of frame data dictionaries
        """
        return self.frame_data 