"""
Test Script for Game Automation Components
Allows testing individual components before running full automation.
"""
import argparse
import time
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from screen_capture import ScreenCapture
from template_matcher import TemplateMatcher
from yolo_trainer import YOLOTrainer
from ocr import extract_text, extract_number
from get_stats_region import capture_and_read


def test_screen_capture(monitor: int = 1):
    """Test screen capture functionality."""
    print("\n" + "="*60)
    print("Testing Screen Capture")
    print("="*60)
    
    sc = ScreenCapture()
    sc.list_all_monitors()
    
    print(f"\nCapturing from monitor {monitor}...")
    screenshot = sc.capture_monitor(monitor)
    
    output_path = Path("debug/test_screenshot.png")
    output_path.parent.mkdir(exist_ok=True)
    screenshot.save(output_path)
    
    print(f"✓ Screenshot saved to {output_path}")
    print(f"  Size: {screenshot.size}")
    return screenshot


def test_template_matching(screenshot: Image.Image, template_name: str):
    """Test template matching."""
    print("\n" + "="*60)
    print(f"Testing Template Matching: {template_name}")
    print("="*60)
    
    tm = TemplateMatcher("assets")
    print(f"Loaded {len(tm.templates)} templates")
    
    screen_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    template_stem = Path(template_name).stem
    result = tm.find_template(screen_cv, template_stem, threshold=0.7)
    
    if result:
        print(f"✓ Found template '{template_stem}':")
        for match in result:
            print(f"  Position: {match['center']}")
            print(f"  Confidence: {match['confidence']:.3f}")
            
        # Visualize
        for match in result:
            x, y = match['center']
            cv2.circle(screen_cv, (x, y), 20, (0, 255, 0), 3)
            cv2.putText(screen_cv, f"{match['confidence']:.2f}", (x-20, y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_path = Path(f"debug/test_template_{template_stem}.png")
        cv2.imwrite(str(output_path), screen_cv)
        print(f"  Visualization saved to {output_path}")
    else:
        print(f"✗ Template '{template_stem}' not found")
    
    return result


def test_yolo_detection(screenshot: Image.Image, model_path: str, labels: list = None):
    """Test YOLO detection."""
    print("\n" + "="*60)
    print("Testing YOLO Detection")
    print("="*60)
    
    yolo = YOLOTrainer()
    
    print(f"Loading model from {model_path}...")
    yolo.load_model(model_path)
    
    # Load class names
    import yaml
    data_yaml = Path("yolo_training/data.yaml")
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            yolo.setup_classes(data.get('names', []))
        print(f"Loaded {len(yolo.class_names)} classes")
    
    print("\nRunning detection...")
    detections = yolo.predict(screenshot, conf=0.25)
    
    # Filter by labels if specified
    if labels:
        detections = [d for d in detections if d['label'] in labels]
        print(f"Filtered to labels: {labels}")
    
    print(f"\n✓ Found {len(detections)} detections:")
    
    # Group by label
    label_counts = {}
    for det in detections:
        label = det['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Visualize
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['conf']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = Path("debug/test_yolo_detection.png")
    cv2.imwrite(str(output_path), frame)
    print(f"\nVisualization saved to {output_path}")
    
    return detections


def test_ocr(screenshot: Image.Image, region: tuple = None):
    """Test OCR functionality."""
    print("\n" + "="*60)
    print("Testing OCR")
    print("="*60)
    
    if region:
        x1, y1, x2, y2 = region
        crop = screenshot.crop((x1, y1, x2, y2))
        print(f"Using region: ({x1}, {y1}) to ({x2}, {y2})")
    else:
        crop = screenshot
        print("Using full screenshot")
    
    # Save crop for debugging
    crop_path = Path("debug/test_ocr_crop.png")
    crop.save(crop_path)
    print(f"Crop saved to {crop_path}")
    
    print("\nExtracting text...")
    text = extract_text(crop)
    print(f"✓ Extracted text:")
    print(f"  '{text}'")
    
    print("\nExtracting number...")
    number = extract_number(crop)
    print(f"✓ Extracted number: {number}")
    
    return text, number


def test_stat_reading(monitor: int = 1):
    """Test stat reading from coordinates."""
    print("\n" + "="*60)
    print("Testing Stat Reading")
    print("="*60)
    
    print("Reading stats from default region (307, 722, 745, 745)...")
    stats = capture_and_read(307, 722, 745, 745, monitor=monitor)
    
    print("✓ Stats read:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value}")
    
    print("\nDebug images saved to debug/stat_*.png")
    
    return stats


def test_all(monitor: int = 1, model_path: str = "yolo_project/weights/best.pt"):
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*25 + "AUTOMATION COMPONENT TESTS")
    print("="*80)
    
    # Test 1: Screen Capture
    screenshot = test_screen_capture(monitor)
    time.sleep(1)
    
    # Test 2: Template Matching
    test_templates = [
        "assets/Buttons/FullStats.png",
        "assets/Buttons/Recreation.png",
        "assets/Buttons/Go1.png",
        "assets/Buttons/Race.png",
    ]
    
    for template in test_templates:
        if Path(template).exists():
            test_template_matching(screenshot, template)
        else:
            print(f"\n⚠ Template not found: {template}")
        time.sleep(0.5)
    
    # Test 3: YOLO Detection
    if Path(model_path).exists():
        test_labels = ['good', 'Great', 'goal', 'race_day', 'go', 'training']
        test_yolo_detection(screenshot, model_path, labels=test_labels)
    else:
        print(f"\n⚠ YOLO model not found: {model_path}")
    time.sleep(1)
    
    # Test 4: OCR
    # Test on a sample region (adjust based on your screen)
    test_ocr(screenshot, region=(300, 200, 600, 250))
    time.sleep(1)
    
    # Test 5: Stat Reading
    test_stat_reading(monitor)
    
    print("\n" + "="*80)
    print(" "*30 + "TESTS COMPLETE")
    print("="*80)
    print("\nCheck the 'debug/' folder for output images.")


def main():
    parser = argparse.ArgumentParser(description="Test Game Automation Components")
    parser.add_argument('--test', choices=['all', 'capture', 'template', 'yolo', 'ocr', 'stats'],
                       default='all', help='Which test to run')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor number to capture from')
    parser.add_argument('--model', type=str, default='yolo_project/weights/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--template', type=str,
                       help='Template path for template matching test')
    parser.add_argument('--region', type=str,
                       help='OCR region as "x1,y1,x2,y2"')
    
    args = parser.parse_args()
    
    # Create debug directory
    Path("debug").mkdir(exist_ok=True)
    
    if args.test == 'all':
        test_all(args.monitor, args.model)
    
    elif args.test == 'capture':
        test_screen_capture(args.monitor)
    
    elif args.test == 'template':
        screenshot = test_screen_capture(args.monitor)
        if args.template:
            test_template_matching(screenshot, args.template)
        else:
            print("Error: --template required for template test")
    
    elif args.test == 'yolo':
        screenshot = test_screen_capture(args.monitor)
        test_yolo_detection(screenshot, args.model)
    
    elif args.test == 'ocr':
        screenshot = test_screen_capture(args.monitor)
        region = None
        if args.region:
            region = tuple(map(int, args.region.split(',')))
        test_ocr(screenshot, region)
    
    elif args.test == 'stats':
        test_stat_reading(args.monitor)


if __name__ == "__main__":
    main()
