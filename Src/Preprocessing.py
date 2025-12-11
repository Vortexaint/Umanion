import os
import cv2

def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
                                    blur, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2
    )
    edges = cv2.Canny(blur, 50, 150)
    cv2.imwrite(output_path.replace('.png', '_gray.png'), gray)
    cv2.imwrite(output_path.replace('.png', '_thresh.png'), thresh)
    cv2.imwrite(output_path.replace('.png', '_edges.png'), edges)

def preprocess_folder(src_folder, dst_folder):
    for root, _, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        out_dir = os.path.join(dst_folder, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                in_path = os.path.join(root, file)
                out_path = os.path.join(out_dir, file)
                preprocess_image(in_path, out_path)

if __name__ == "__main__":
    preprocess_folder("Assets", "Preprocessed")