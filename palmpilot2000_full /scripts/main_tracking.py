
from FSM_SOURCE.SCRIPTS.lauren.fsm_logic import FSMHandler
from visual_tools import draw_workspace, draw_boxes, export_final_layout
import cv2
import socket
import os
import numpy as np

# CONFIG
UDP_IP = "127.0.0.1"
UDP_PORT_GH = 5000

# WORKSPACE DEFINITION
WORKSPACE_TOP_LEFT = (150, 100)
WORKSPACE_BOTTOM_RIGHT = (600, 400)

# SETUP
fsm = FSMHandler(WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = fsm.process_frame(frame)

    # Draw overlays
    draw_workspace(frame, WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT)
    draw_boxes(frame, fsm.boxes)

    # Export current box state
    box_data = fsm.export_box_data()
    sock_send.sendto(str(box_data).encode("utf-8"), (UDP_IP, UDP_PORT_GH))

    # Display state
    cv2.putText(frame, f'STATE: {fsm.fsm_state}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("2D HAND INTERACTION", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
sock_send.close()

# Export final layout
export_dir = r"C:\Users\laure\Desktop\MRAC-FSM-tutorial\fsm_tutorial\exported_layouts"
os.makedirs(export_dir, exist_ok=True)
fsm.export_to_csv(export_dir)
export_final_layout(fsm.boxes, WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, export_dir)
