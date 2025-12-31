import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the classes
classes_file = "D:/Kampus/Comvis/Uma/yolo_project/classes.txt"
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random per-class precision, recall, and F1-score for demonstration
# Replace this with actual per-class metrics if available
np.random.seed(42)  # For consistent results
precision = np.random.uniform(0.5, 0.9, len(classes))
recall = np.random.uniform(0.3, 0.8, len(classes))
f1_score = 2 * (precision * recall) / (precision + recall)

# Create bar width and positions
bar_width = 0.25
r1 = np.arange(len(classes))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot bar chart
plt.figure(figsize=(15, 8))
plt.bar(r1, precision, color='blue', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r2, recall, color='green', width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r3, f1_score, color='red', width=bar_width, edgecolor='grey', label='F1-Score')

# Add labels, legend, and title
plt.xlabel('Classes', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('Per-Class Model Performance: Precision, Recall, and F1-Score', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(classes))], classes, rotation=90)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = "D:/Kampus/Comvis/Uma/yolo_project/train/per_class_model_score_bar_chart.png"
plt.savefig(output_path)
print(f"Per-class bar chart visualization saved to {output_path}")

# Show the plot
plt.show()