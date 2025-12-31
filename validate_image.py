"""
Validate and visualize YOLO predictions on a specific image
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

from src.yolo_trainer import YOLOTrainer


def validate_image(image_path: str, model_path: str = "yolo_project/weights/best.pt"):
    """Run validation on a single image and save visualization."""
    
    # Check if image exists
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return
    
    print(f"=== YOLO Image Validation ===")
    print(f"Image: {img_path}")
    print(f"Model: {model_path}\n")
    
    # Load model
    yolo = YOLOTrainer()
    print(f"Loading model from {model_path}...")
    yolo.load_model(model_path)
    
    # Load class names from data.yaml
    data_yaml = Path("yolo_training/data.yaml")
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            yolo.setup_classes(data.get('names', []))
    
    # Load image
    print(f"Loading image...")
    screenshot = Image.open(img_path)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Run prediction
    print(f"Running prediction with confidence threshold 0.25...\n")
    detections = yolo.predict(screenshot, conf=0.25)
    
    # Group detections by label
    label_counts = {}
    for det in detections:
        label = det['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Found {len(detections)} detections:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()
    
    # Draw detections
    colors = {}
    np.random.seed(42)
    
    for det in detections:
        label = det['label']
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        
        # Assign consistent color per label
        if label not in colors:
            colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
        color = colors[label]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        print(f"  {label} ({conf:.3f}): Box=({x1},{y1},{x2},{y2})")
    
    # Save visualization
    output_dir = Path("debug/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{img_path.stem}_validation.jpg"
    cv2.imwrite(str(output_path), frame)
    
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Also save ground truth comparison if labels exist
    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    if label_path.exists():
        print(f"\n=== Ground Truth Labels ===")
        with open(label_path, 'r') as f:
            lines = f.readlines()
        print(f"Found {len(lines)} ground truth annotations")
        
        # Load class names
        class_names = yolo.class_names if yolo.class_names else []
        
        # Draw ground truth on separate image
        frame_gt = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        img_height, img_width = frame_gt.shape[:2]
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert from YOLO format to pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                label_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                # Draw in green for ground truth
                cv2.rectangle(frame_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_gt,
                    label_name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                print(f"  {label_name}: Box=({x1},{y1},{x2},{y2})")
        
        # Save ground truth visualization
        output_gt_path = output_dir / f"{img_path.stem}_ground_truth.jpg"
        cv2.imwrite(str(output_gt_path), frame_gt)
        print(f"\n✓ Ground truth visualization saved to: {output_gt_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate YOLO model on a single image")
    parser.add_argument('image', type=str,
                       help='Path to image file')
    parser.add_argument('--model', type=str, default='yolo_project/weights/best.pt',
                       help='Path to model weights (default: yolo_project/weights/best.pt)')
    
    args = parser.parse_args()
    
    validate_image(args.image, args.model)
