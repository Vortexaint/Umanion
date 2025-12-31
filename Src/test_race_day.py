"""
Test script to check race_day detection
Tests YOLO detection and saves output images
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time
from datetime import datetime

from screen_capture import ScreenCapture
from yolo_trainer import YOLOTrainer


def test_race_day_detection(monitor: int = 2, save_images: bool = True, max_captures: int = 10):
    """Test race_day detection on screen."""
    print("=== Race Day Detection Test ===")
    print("This will capture the screen and check for race_day detection")
    print(f"Will capture {max_captures} screenshots\n")
    
    # Create output directory
    output_dir = Path("debug/race_day_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    screen_capture = ScreenCapture()
    yolo = YOLOTrainer()
    
    # Load YOLO model
    model_path = "yolo_project/weights/best.pt"
    print(f"Loading YOLO model from {model_path}...")
    yolo.load_model(model_path)
    
    # Load class names
    data_yaml_path = Path("yolo_training/data.yaml")
    if data_yaml_path.exists():
        import yaml
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            yolo.setup_classes(data.get('names', []))
    
    print(f"Capturing from Monitor {monitor}")
    print("Looking for 'race_day' label...")
    print(f"Saving images to {output_dir}\n")
    
    try:
        for i in range(max_captures):
            # Capture screen
            screenshot = screen_capture.capture_monitor(monitor)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Detect race_day
            detections = yolo.predict(screenshot, conf=0.25)
            
            race_day_found = False
            detection_info = []
            
            for det in detections:
                if det['label'] == 'race_day':
                    race_day_found = True
                    x1, y1, x2, y2 = det['box']
                    conf = det['conf']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 3)
                    
                    # Draw label
                    label_text = f"race_day {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
                    
                    # Draw center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
                    
                    detection_info.append({
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'conf': conf
                    })
                    
                    print(f"[{i+1}/{max_captures}] RACE_DAY DETECTED: Box=({x1},{y1},{x2},{y2}), Center=({center_x},{center_y}), Conf={conf:.3f}")
            
            if not race_day_found:
                # Display "Not Detected" message
                cv2.putText(frame, "race_day: NOT DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                print(f"[{i+1}/{max_captures}] race_day: NOT DETECTED")
            else:
                # Display "Detected" message
                cv2.putText(frame, f"race_day: DETECTED ({len(detection_info)})", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Save the frame
            if save_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                status = "detected" if race_day_found else "not_detected"
                filename = f"race_day_{status}_{timestamp}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"    Saved: {filepath}")
            
            time.sleep(1)
        
        print(f"\n✓ Test completed - captured {max_captures} screenshots")
        print(f"✓ Images saved to: {output_dir}")
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test race_day detection")
    parser.add_argument('--monitor', type=int, default=2,
                       help='Monitor number to capture from (default: 2)')
    parser.add_argument('--captures', type=int, default=10,
                       help='Number of screenshots to capture (default: 10)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save images to disk')
    
    args = parser.parse_args()
    
    test_race_day_detection(
        monitor=args.monitor,
        save_images=not args.no_save,
        max_captures=args.captures
    )
