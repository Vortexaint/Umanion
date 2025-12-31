"""Read stat additions from training hover UI."""
from pathlib import Path
from PIL import Image
from ocr import extract_stat_addition
import argparse


def read_stat_additions(img: Image.Image, left: int = 270, top: int = 647, right: int = 745, bottom: int = 695):
    """
    Read stat additions from the training hover region.
    
    Returns dict: {'speed': +X, 'stamina': +Y, 'power': +Z, 'guts': +A, 'wit': +B}
    where values are 0 if no addition detected.
    """
    # Crop the stat additions region
    stat_region = img.crop((left, top, right, bottom))
    
    # Split into 5 equal segments for 5 stats
    width = right - left
    height = bottom - top
    seg_w = width // 5
    
    stats = ["speed", "stamina", "power", "guts", "wit"]
    results = {}
    
    for i, stat_name in enumerate(stats):
        seg_left = i * seg_w
        seg_right = (i + 1) * seg_w if i < 4 else width
        
        # Crop individual stat segment
        crop = stat_region.crop((seg_left, 0, seg_right, height))
        
        # Save debug image
        Path("debug").mkdir(exist_ok=True)
        crop.save(f"debug/stat_add_{i+1}_{stat_name}.png")
        
        # Extract stat addition (looks for +X format, 0-30 range)
        value = extract_stat_addition(crop)
        results[stat_name] = value
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Read stat additions from training hover screen")
    parser.add_argument("--image", required=True, help="Path to screenshot image")
    parser.add_argument("--left", type=int, default=270)
    parser.add_argument("--top", type=int, default=647)
    parser.add_argument("--right", type=int, default=745)
    parser.add_argument("--bottom", type=int, default=695)
    args = parser.parse_args()
    
    img = Image.open(args.image)
    results = read_stat_additions(img, args.left, args.top, args.right, args.bottom)
    
    print("Stat additions detected:")
    for stat, value in results.items():
        display = f"+{value}" if value > 0 else "none"
        print(f"  {stat:8s}: {display}")
    
    print(f"\nRaw dict: {results}")


if __name__ == "__main__":
    main()
