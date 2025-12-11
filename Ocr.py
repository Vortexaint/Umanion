import os
import re
import numpy as np
import pytesseract


class OCR:
    def __init__(self):
        # Path to tesseract.exe
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text(self, img: np.ndarray) -> str:
        """Extract text"""
        config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, lang="eng", config=config)
        return text.strip()

    def extract_number(self, img: np.ndarray) -> int:
        """Extract number (digit only)"""

        digit_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(img, lang="eng", config=digit_config)

        # Cleanup for safety
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else -1

    def extract_data(self, img: np.ndarray) -> dict:
        """Return detailed OCR data (boxes, confidences, text) using pytesseract's image_to_data.

        The returned dict follows pytesseract.Output.DICT format with keys like
        'left','top','width','height','text','conf'.
        """
        config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(img, lang="eng", config=config, output_type=pytesseract.Output.DICT)
        return data

    def extract_words_with_boxes(self, img: np.ndarray) -> list:
        """Return a list of words with bounding boxes and confidence.

        Each item is a dict: {'text', 'left', 'top', 'width', 'height', 'conf'}
        Only non-empty text entries are returned.
        """
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
