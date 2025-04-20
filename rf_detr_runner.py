import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import torch

model = RFDETRBase()

def rf_detr_callback(frame, index, threshold=0.5):
    detections = model.predict(frame[:, :, ::-1].copy(), threshold=threshold)
    return detections 