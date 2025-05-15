
import time
import numpy as np
import mediapipe as mp
import cv2

class FSMHandler:
    def __init__(self, workspace_tl, workspace_br):
        self.WORKSPACE_TOP_LEFT = workspace_tl
        self.WORKSPACE_BOTTOM_RIGHT = workspace_br
        self.boxes = self.initialize_boxes()
        self.selected_box_index = None
        self.fsm_state = "IDLE"
        self.last_rotation_time = 0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

    def initialize_boxes(self):
        boxes = [{"pos": [50 + (i % 2) * 50, 100 + (i // 2) * 50], "size": 20,
                  "color": tuple(np.random.randint(0, 255, 3).tolist()), "selected": False, "angle": 0}
                 for i in range(16)]
        for box in boxes:
            box["original_pos"] = box["pos"][:]
        return boxes

    def count_fingers(self, landmarks):
        TIP_IDS = [4, 8, 12, 16, 20]
        count = 0
        if landmarks[4].x < landmarks[3].x:
            count += 1
        for tip in TIP_IDS[1:]:
            if landmarks[tip].y < landmarks[tip - 2].y:
                count += 1
        return count

    def is_inside_box(self, point, box):
        x, y = point
        bx, by = box["pos"]
        s = box["size"]
        return (bx - s < x < bx + s) and (by - s < y < by + s)

    def clamp_to_workspace(self, x, y, box_size):
        x = max(self.WORKSPACE_TOP_LEFT[0] + box_size, min(self.WORKSPACE_BOTTOM_RIGHT[0] - box_size, x))
        y = max(self.WORKSPACE_TOP_LEFT[1] + box_size, min(self.WORKSPACE_BOTTOM_RIGHT[1] - box_size, y))
        return x, y

    def is_overlapping(self, i, test_x, test_y):
        s1 = self.boxes[i]["size"]
        l1, r1 = test_x - s1, test_x + s1
        t1, b1 = test_y - s1, test_y + s1
        for j, box in enumerate(self.boxes):
            if i == j or not box.get("pos"):
                continue
            x2, y2 = box["pos"]
            s2 = box["size"]
            l2, r2 = x2 - s2, x2 + s2
            t2, b2 = y2 - s2, y2 + s2
            if not (r1 <= l2 or l1 >= r2 or b1 <= t2 or t1 >= b2):
                return True
        return False

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            cx, cy = int(landmarks[9].x * w), int(landmarks[9].y * h)
            fingers = self.count_fingers(landmarks)

            if self.selected_box_index is not None and self.fsm_state == "PICK":
                if not self.is_overlapping(self.selected_box_index, cx, cy):
                    self.boxes[self.selected_box_index]["pos"] = [cx, cy]

            if fingers == 0:
                self.fsm_state = "PICK"
                for i, box in enumerate(self.boxes):
                    if self.is_inside_box((cx, cy), box):
                        box["original_pos"] = box["pos"][:]
                        box["selected"] = True
                        self.selected_box_index = i
                        break

            elif fingers == 5:
                self.fsm_state = "PLACE"
                if self.selected_box_index is not None:
                    x, y = self.boxes[self.selected_box_index]["pos"]
                    x, y = self.clamp_to_workspace(x, y, self.boxes[self.selected_box_index]["size"])
                    if self.is_overlapping(self.selected_box_index, x, y):
                        self.boxes[self.selected_box_index]["pos"] = self.boxes[self.selected_box_index]["original_pos"]
                        cv2.putText(frame, "ERROR: OVERLAPPING BOX", (50, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        self.boxes[self.selected_box_index]["pos"] = [x, y]
                    self.boxes[self.selected_box_index]["selected"] = False
                    self.selected_box_index = None

            elif fingers == 2:
                self.fsm_state = "ROTATE"
                if self.selected_box_index is not None:
                    current_time = time.time()
                    if current_time - self.last_rotation_time >= 0.5:
                        self.boxes[self.selected_box_index]["angle"] = (
                            self.boxes[self.selected_box_index]["angle"] + 45) % 360
                        self.last_rotation_time = current_time
            else:
                self.fsm_state = "IDLE"

            self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

        return frame

    def export_box_data(self):
        width = self.WORKSPACE_BOTTOM_RIGHT[0] - self.WORKSPACE_TOP_LEFT[0]
        height = self.WORKSPACE_BOTTOM_RIGHT[1] - self.WORKSPACE_TOP_LEFT[1]
        positions = [
            [(box["pos"][0] - self.WORKSPACE_TOP_LEFT[0]) / width,
             (box["pos"][1] - self.WORKSPACE_TOP_LEFT[1]) / height]
            for box in self.boxes
        ]
        return {
            "positions": positions,
            "angles": [box["angle"] for box in self.boxes],
            "selected": self.selected_box_index
        }

    def export_to_csv(self, folder_path):
        import csv, os
        data = self.export_box_data()
        csv_path = os.path.join(folder_path, "box_data.csv")
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x_norm", "y_norm", "angle"])
            for pos, angle in zip(data["positions"], data["angles"]):
                writer.writerow([pos[0], pos[1], angle])
