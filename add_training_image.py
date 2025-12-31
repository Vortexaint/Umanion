"""
Add new image to training dataset with annotations
"""
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def add_to_training(image_path: str, label_name: str = "race_day"):
    """
    Add a new image to the training dataset.
    This creates a copy in the training folder and helps you create the label file.
    """
    
    src_path = Path(image_path)
    if not src_path.exists():
        print(f"Error: Image not found at {src_path}")
        return
    
    print(f"=== Add Image to Training Dataset ===")
    print(f"Source: {src_path}")
    print(f"Label to add: {label_name}\n")
    
    # Load class names
    data_yaml = Path("yolo_training/data.yaml")
    class_names = []
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
    
    # Find class ID
    if label_name not in class_names:
        print(f"Error: '{label_name}' not found in classes")
        print(f"Available classes: {', '.join(class_names)}")
        return
    
    class_id = class_names.index(label_name)
    print(f"Class ID for '{label_name}': {class_id}")
    
    # Create destination paths
    train_images_dir = Path("yolo_training/train/images")
    train_labels_dir = Path("yolo_training/train/labels")
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    base_name = src_path.stem
    dest_image = train_images_dir / f"{base_name}_new.jpg"
    dest_label = train_labels_dir / f"{base_name}_new.txt"
    
    # Copy image
    shutil.copy(src_path, dest_image)
    print(f"\n✓ Image copied to: {dest_image}")
    
    # Load image for annotation helper
    img = cv2.imread(str(src_path))
    if img is None:
        print("Error: Could not load image")
        return
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Open image in default viewer
    print(f"\n{'='*60}")
    print("MANUAL ANNOTATION")
    print(f"{'='*60}")
    print(f"Opening image in default viewer...")
    
    import os
    os.startfile(str(src_path))
    
    print(f"\nImage dimensions: {width} x {height}")
    print(f"\nTo annotate '{label_name}' in the image:")
    print("1. Look at the image that opened")
    print("2. Identify the bounding box coordinates")
    print("3. Enter the coordinates below\n")
    
    try:
        print("Enter bounding box coordinates (in pixels):")
        x1 = int(input("  Top-left X: "))
        y1 = int(input("  Top-left Y: "))
        x2 = int(input("  Bottom-right X: "))
        y2 = int(input("  Bottom-right Y: "))
        
        # Validate coordinates
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            print("\nError: Invalid coordinates")
            return
        
        # Convert to YOLO format (x_center, y_center, width, height) normalized to 0-1
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        
        # Save label file
        with open(dest_label, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        print(f"\n✓ Label saved to: {dest_label}")
        print(f"  Class: {label_name} (ID: {class_id})")
        print(f"  Box: x_center={x_center:.3f}, y_center={y_center:.3f}, width={box_width:.3f}, height={box_height:.3f}")
        
        # Draw visualization
        vis_img = img.copy()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis_img, label_name, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save visualization
        vis_path = Path("debug/annotations") / f"{dest_image.stem}_annotated.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_path), vis_img)
        print(f"✓ Visualization saved to: {vis_path}")
        
        print(f"\n✓ Image added to training dataset!")
        print(f"\nTo retrain the model, run:")
        print(f"  python train_yolo.py")
        
    except ValueError:
        print("\nError: Please enter valid integer coordinates")
    except KeyboardInterrupt:
        print("\n\nCancelled - no label file created")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add image to training dataset with annotation")
    parser.add_argument('image', type=str,
                       help='Path to image file')
    parser.add_argument('--label', type=str, default='race_day',
                       help='Label name (default: race_day)')
    
    args = parser.parse_args()
    
    add_to_training(args.image, args.label)
