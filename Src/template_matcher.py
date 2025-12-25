"""Template matching system for finding images on screen."""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import os
from pathlib import Path


class TemplateMatcher:
    """Template matching for finding UI elements on screen."""
    
    def __init__(self, template_dir: str = "assets"):
        """
        Initialize template matcher.
        
        Args:
            template_dir: Directory containing template images
        """
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.load_templates()
        
    def load_templates(self):
        """Load all templates from the template directory."""
        if not self.template_dir.exists():
            print(f"Warning: Template directory {self.template_dir} does not exist")
            return
        
        for root, dirs, files in os.walk(self.template_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    template_path = Path(root) / file
                    template_name = file.rsplit('.', 1)[0]
                    
                    # Load template
                    template_img = cv2.imread(str(template_path))
                    if template_img is not None:
                        self.templates[template_name] = template_img
        
        print(f"Loaded {len(self.templates)} templates")
    
    def add_template(self, name: str, template_img: np.ndarray):
        """Add a template programmatically."""
        self.templates[name] = template_img
    
    def find_template(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Find a template in the screen image.
        
        Args:
            screen: Screen image as numpy array
            template_name: Name of the template to find
            threshold: Matching threshold (0-1)
            method: OpenCV matching method
            
        Returns:
            (x, y, width, height, confidence) or None if not found
        """
        if template_name not in self.templates:
            print(f"Template '{template_name}' not found")
            return None
        
        template = self.templates[template_name]
        
        # Convert to grayscale for better matching
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if len(screen.shape) == 3 else screen
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # Perform template matching
        result = cv2.matchTemplate(screen_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get the best match location
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            confidence = 1 - min_val
            match_loc = min_loc
        else:
            confidence = max_val
            match_loc = max_loc
        
        if confidence >= threshold:
            h, w = template_gray.shape
            x, y = match_loc
            return (x, y, w, h, confidence)
        
        return None
    
    def find_all_templates(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Find all instances of a template in the screen.
        
        Returns:
            List of (x, y, width, height, confidence) tuples
        """
        if template_name not in self.templates:
            print(f"Template '{template_name}' not found")
            return []
        
        template = self.templates[template_name]
        
        # Convert to grayscale
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if len(screen.shape) == 3 else screen
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # Perform template matching
        result = cv2.matchTemplate(screen_gray, template_gray, method)
        
        # Find all locations above threshold
        locations = np.where(result >= threshold)
        matches = []
        
        h, w = template_gray.shape
        for pt in zip(*locations[::-1]):
            x, y = pt
            confidence = result[y, x]
            matches.append((x, y, w, h, confidence))
        
        # Remove overlapping matches (non-maximum suppression)
        matches = self._non_max_suppression(matches)
        
        return matches
    
    def _non_max_suppression(
        self,
        matches: List[Tuple[int, int, int, int, float]],
        overlap_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int, float]]:
        """Remove overlapping matches using non-maximum suppression."""
        if len(matches) == 0:
            return []
        
        # Sort by confidence
        matches = sorted(matches, key=lambda x: x[4], reverse=True)
        
        selected = []
        while matches:
            best = matches.pop(0)
            selected.append(best)
            
            # Remove overlapping matches
            matches = [
                m for m in matches
                if self._calculate_iou(best, m) < overlap_threshold
            ]
        
        return selected
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int, float],
        box2: Tuple[int, int, int, int, float]
    ) -> float:
        """Calculate Intersection over Union of two boxes."""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_with_pil(
        self,
        screen: Image.Image,
        template_name: str,
        threshold: float = 0.8
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Find template using PIL Image.
        
        Args:
            screen: PIL Image of screen
            template_name: Name of template
            threshold: Matching threshold
            
        Returns:
            (x, y, width, height, confidence) or None
        """
        # Convert PIL to numpy
        screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        return self.find_template(screen_np, template_name, threshold)
    
    def visualize_match(
        self,
        screen: np.ndarray,
        match: Tuple[int, int, int, int, float],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw rectangle around matched region.
        
        Returns:
            Screen image with match highlighted
        """
        x, y, w, h, confidence = match
        screen_copy = screen.copy()
        cv2.rectangle(screen_copy, (x, y), (x + w, y + h), color, 2)
        
        # Add confidence text
        text = f"{confidence:.2f}"
        cv2.putText(screen_copy, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return screen_copy
