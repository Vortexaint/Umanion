"""Test script to detect training stats and read failure rate + stat additions from image."""
import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr import extract_text, extract_number, extract_stat_addition, extract_percentage
from get_stats_region import capture_and_read
from read_stat_additions import read_stat_additions
from screen_capture import ScreenCapture

def test_training_image(image_path: str):
    """Test training detection on a static image."""
    print(f"\n{'='*60}")
    print(f"Testing Training Detection: {image_path}")
    print(f"{'='*60}\n")
    
    # Load image
    img = Image.open(image_path)
    print(f"Image size: {img.size}")
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Test 1: Check for "fail" text in expected region
    # Typically failure rate appears in upper area during hover
    print("\n--- Test 1: Searching for failure rate ---")
    
    # Common failure rate regions (adjust based on game UI)
    fail_regions = [
        (200, 100, 800, 300),  # Top area
        (300, 200, 700, 400),  # Mid-top area
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(fail_regions):
        region = img.crop((x1, y1, x2, y2))
        region.save(f"debug/fail_region_{i}.png")
        text = extract_text(region).lower()
        print(f"Region {i} ({x1},{y1},{x2},{y2}): '{text}'")
        
        if "fail" in text or "%" in text:
            number = extract_number(region)
            print(f"  -> Found failure rate: {number}%")
    
    # Test 2: Read stat additions from region 270, 647 to 745, 695
    print("\n--- Test 2: Reading stat additions from (270,647) to (745,695) ---")
    
    stat_additions = read_stat_additions(img, 270, 647, 745, 695)
    
    print("\nDetected stat additions:")
    for stat, value in stat_additions.items():
        display = f"+{value}" if value > 0 else "none"
        print(f"  {stat:8s}: {display}")
    
    print(f"\nFinal stat additions: {stat_additions}")
    
    # Test 2b: Read failure rate from correct region (273, 767 to 841, 815)
    print("\n--- Test 2b: Reading failure rate from (273,767) to (841,815) ---")
    
    fail_region = img.crop((273, 767, 841, 815))
    fail_region.save("debug/fail_rate_region.png")
    print(f"Saved fail rate region to debug/fail_rate_region.png")
    
    fail_text = extract_text(fail_region)
    fail_number_old = extract_number(fail_region)
    fail_percentage = extract_percentage(fail_region)
    print(f"Fail region text: '{fail_text}'")
    print(f"Fail rate (old extract_number): {fail_number_old}%")
    print(f"Fail rate (new extract_percentage): {fail_percentage}%")
    
    # Test 3: Check entire image for any visible text
    print("\n--- Test 3: Full image OCR ---")
    full_text = extract_text(img)
    print(f"All detected text: {full_text[:200]}...")
    
    # Test 4: Visual display with regions marked
    print("\n--- Test 4: Creating annotated image ---")
    display = img_cv.copy()
    
    # Mark stat additions region
    cv2.rectangle(display, (270, 647), (745, 695), (0, 255, 0), 2)
    cv2.putText(display, "Stat Additions", (270, 640), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mark failure rate region
    cv2.rectangle(display, (273, 767), (841, 815), (0, 0, 255), 2)
    cv2.putText(display, "Failure Rate", (273, 760), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Mark individual stat segments
    stats = ["speed", "stamina", "power", "guts", "wit"]
    width = 745 - 270
    seg_w = width // 5
    
    for i, stat_name in enumerate(stats):
        seg_left = 270 + i * seg_w
        seg_right = 270 + ((i + 1) * seg_w if i < 4 else width)
        
        cv2.rectangle(display, (seg_left, 647), (seg_right, 695), (255, 0, 0), 1)
        cv2.putText(display, stat_name[:3].upper(), (seg_left + 5, 670),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Save annotated image
    cv2.imwrite("debug/training_annotated.png", display)
    print("Saved annotated image to debug/training_annotated.png")
    
    print(f"\n{'='*60}")
    print("Testing complete! Check debug/ folder for output images.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    # Create debug directory
    Path("debug").mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Test training stat detection")
    parser.add_argument("--image", type=str, default="test/image.png",
                       help="Path to test image")
    args = parser.parse_args()
    
    test_training_image(args.image)
