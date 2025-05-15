import cv2
import numpy as np
import mediapipe as mp
import socket
import os
import csv
import time

# === CONFIG ===
UDP_IP = "127.0.0.1"
UDP_PORT_GH = 4600 #UPDATE THIS TO YOUR UDP PORT

# === SETUP SOCKET === 
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# INITIALIZE MEDIAPIPE HAND MODULE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# DEFINE WORKSPACE RECTANGLE
WORKSPACE_TOP_LEFT = (450, 100)
WORKSPACE_BOTTOM_RIGHT = (1500, 1000)
BOX_X = 280
BOX_Y = 100
BOX_SPACE = 100

# INITIALIZE BOXES
boxes = [
    {"pos": [BOX_X, 100], "size": 20, "color": (0, 255, 0), "selected": False, "angle": 0},
    {"pos": [BOX_X, 200], "size": 20, "color": (255, 0, 0), "selected": False, "angle": 0},
    {"pos": [BOX_X, 300], "size": 20, "color": (0, 0, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X, 400], "size": 20, "color": (255, 255, 0), "selected": False, "angle": 0},
    {"pos": [BOX_X, 500], "size": 20, "color": (255, 0, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X, 600], "size": 20, "color": (0, 255, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X, 700], "size": 20, "color": (128, 128, 128), "selected": False, "angle": 0},
    {"pos": [BOX_X, 800], "size": 20, "color": (128, 0, 128), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 100], "size": 20, "color": (128, 128, 0), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 200], "size": 20, "color": (255, 128, 128), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 300], "size": 20, "color": (128, 255, 128), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 400], "size": 20, "color": (128, 128, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 500], "size": 20, "color": (255, 255, 128), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 600], "size": 20, "color": (255, 128, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 700], "size": 20, "color": (128, 255, 255), "selected": False, "angle": 0},
    {"pos": [BOX_X + BOX_SPACE, 800], "size": 20, "color": (192, 192, 192), "selected": False, "angle": 0},
]

selected_box_index = None
fsm_state = "IDLE"

# Helper functions
def count_fingers(landmarks):
    TIP_IDS = [4, 8, 12, 16, 20]
    count = 0
    if landmarks[4].x < landmarks[3].x:
        count += 1
    for tip in TIP_IDS[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

def is_inside_box(point, box):
    x, y = point
    bx, by = box["pos"]
    s = box["size"]
    return (bx - s < x < bx + s) and (by - s < y < by + s)

def clamp_to_workspace(x, y, box_size):
    x = max(WORKSPACE_TOP_LEFT[0] + box_size, min(WORKSPACE_BOTTOM_RIGHT[0] - box_size, x))
    y = max(WORKSPACE_TOP_LEFT[1] + box_size, min(WORKSPACE_BOTTOM_RIGHT[1] - box_size, y))
    return x, y

def is_overlapping(i, x, y):
    for j, other in enumerate(boxes):
        if i != j:
            ox, oy = other["pos"]
            dist = np.linalg.norm([x - ox, y - oy])
            if dist < boxes[i]["size"] * 2:
                return True
    return False

# Start capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_pos = None

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        cx = int(landmarks[9].x * w)
        cy = int(landmarks[9].y * h)
        hand_pos = (cx, cy)
        fingers = count_fingers(landmarks)

        if fingers == 0:
            fsm_state = "PICK"
            for i, box in enumerate(boxes):
                if is_inside_box(hand_pos, box):
                    box["selected"] = True
                    selected_box_index = i
                    break
        elif fingers == 5:
            fsm_state = "PLACE"
            if selected_box_index is not None:
                x, y = boxes[selected_box_index]["pos"]
                x, y = clamp_to_workspace(x, y, boxes[selected_box_index]["size"])
                if not is_overlapping(selected_box_index, x, y):
                    boxes[selected_box_index]["pos"] = [x, y]
                    boxes[selected_box_index]["selected"] = False
                    selected_box_index = None
        elif fingers == 2:
            fsm_state = "ROTATE"
            if selected_box_index is not None:
                boxes[selected_box_index]["angle"] += 5
        else:
            fsm_state = "IDLE"

        if selected_box_index is not None and fsm_state == "PICK":
            boxes[selected_box_index]["pos"] = [cx, cy]

        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, (200, 200, 200), 2)

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

    workspace_width = WORKSPACE_BOTTOM_RIGHT[0] - WORKSPACE_TOP_LEFT[0]
    workspace_height = WORKSPACE_BOTTOM_RIGHT[1] - WORKSPACE_TOP_LEFT[1]
    box_positions_vectorized = np.array([
        [(box["pos"][0] - WORKSPACE_TOP_LEFT[0]) / workspace_width,
         (1-(box["pos"][1] - WORKSPACE_TOP_LEFT[1]) / workspace_height)]
        for box in boxes
    ])


    box_data = {
        "positions": box_positions_vectorized.tolist(),
        "angles": [box["angle"] for box in boxes],
        "selected": selected_box_index
    }

    sock_send.sendto(str(box_data).encode("utf-8"), (UDP_IP, UDP_PORT_GH))

    cv2.putText(frame, f'STATE: {fsm_state}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("2D HAND INTERACTION", frame)

    time.sleep(0.05)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()

# Export
export_dir = r"C:\\Users\\billm\\Desktop\\CODE\\coding\\HARDWARE_3\\MRAC-FSM-tutorial-main\\fsm_tutorial"
os.makedirs(export_dir, exist_ok=True)

csv_path = os.path.join(export_dir, "box_data.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x_norm", "y_norm", "angle"])
    for pos, angle in zip(box_data["positions"], box_data["angles"]):
        writer.writerow([pos[0], pos[1], angle])

# Create black background and draw boxes
black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(black_frame, WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, (200, 200, 200), 2)

png_path = os.path.join(export_dir, "box_layout.png")
cv2.imwrite(png_path, black_frame)
