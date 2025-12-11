import os
import re
import numpy as np
import pytesseract


class OCR:
    def __init__(self):
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text(self, img: np.ndarray) -> str:
        """Extract plain text from an image."""
        config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, lang="eng", config=config)
        return text.strip()

    def extract_number(self, img: np.ndarray) -> int:
        """Extract digits from an image, return -1 if none."""
        digit_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(img, lang="eng", config=digit_config)
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else -1

    def extract_data(self, img: np.ndarray) -> dict:
        """Return pytesseract image_to_data output as a dict."""
        config = r"--oem 3 --psm 6"
        return pytesseract.image_to_data(img, lang="eng", config=config, output_type=pytesseract.Output.DICT)

    def extract_words_with_boxes(self, img: np.ndarray) -> list:
        """Return non-empty words with bounding boxes and confidence."""
        data = self.extract_data(img)
        words = []
        n = len(data.get('text', []))
        for i in range(n):
            txt = str(data.get('text', [''])[i]).strip()
            if not txt:
                continue
            try:
                conf = int(float(data.get('conf', ['-1'])[i]))
            except Exception:
                try:
                    conf = int(data.get('conf', ['-1'])[i])
                except Exception:
                    conf = -1
            words.append({
                'text': txt,
                'left': int(data.get('left', [0])[i]),
                'top': int(data.get('top', [0])[i]),
                'width': int(data.get('width', [0])[i]),
                'height': int(data.get('height', [0])[i]),
                'conf': conf,
            })
        return words
