
import cv2
import numpy as np
import os

def draw_workspace(frame, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, (200, 200, 200), 2)

def draw_boxes(frame, boxes):
    for box in boxes:
        x, y = box["pos"]
        s = box["size"]
        angle = box["angle"]
        rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rect_pts = np.array([
            [x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s]
        ], dtype=np.float32)
        rotated_pts = cv2.transform(np.array([rect_pts]), rot_matrix)[0].astype(int)
        cv2.polylines(frame, [rotated_pts], isClosed=True, color=box["color"], thickness=2)

def export_final_layout(boxes, top_left, bottom_right, folder_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_workspace(frame, top_left, bottom_right)
    draw_boxes(frame, boxes)
    out_path = os.path.join(folder_path, "box_layout.png")
    cv2.imwrite(out_path, frame)
