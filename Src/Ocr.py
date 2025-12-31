import easyocr
from PIL import Image
import numpy as np
import re

reader = easyocr.Reader(["en"], gpu=False)

def extract_text(pil_img: Image.Image) -> str:
  img_np = np.array(pil_img)
  result = reader.readtext(img_np)
  texts = [text[1] for text in result]
  return " ".join(texts)

def extract_number(pil_img: Image.Image) -> int:
  img_np = np.array(pil_img)
  result = reader.readtext(img_np, allowlist="0123456789")
  texts = [text[1] for text in result]
  joined_text = "".join(texts)

  digits = re.sub(r"[^\d]", "", joined_text)

  if digits:
    return int(digits)
  
  return -1

def extract_stat_addition(pil_img: Image.Image) -> int:
  """Extract stat addition value (looks for +X format, returns small values 0-30 range)."""
  img_np = np.array(pil_img)
  result = reader.readtext(img_np)
  
  # Look for patterns like "+5", "+13", etc.
  for detection in result:
    text = detection[1].strip()
    
    # Check if text contains + followed by digits
    match = re.search(r'\+\s*(\d+)', text)
    if match:
      value = int(match.group(1))
      # Stat additions are typically 1-30
      if 1 <= value <= 30:
        return value
    
    # Also try just digits if small enough (in case + is missed)
    digits = re.sub(r'[^\d]', '', text)
    if digits and len(digits) <= 2:  # 1-2 digit numbers only
      value = int(digits)
      if 1 <= value <= 30:
        return value
  
  return 0  # No addition found

def extract_percentage(pil_img: Image.Image) -> int:
  """Extract percentage value (looks for X% format, returns 0-100 range)."""
  img_np = np.array(pil_img)
  result = reader.readtext(img_np)
  
  # Look for patterns like "15%", "0%", "Failure 10%", etc.
  for detection in result:
    text = detection[1].strip()
    
    # Match number followed by %
    match = re.search(r'(\d+)\s*%', text)
    if match:
      value = int(match.group(1))
      # Failure rates are 0-100
      if 0 <= value <= 100:
        return value
  
  return -1  # No percentage found
