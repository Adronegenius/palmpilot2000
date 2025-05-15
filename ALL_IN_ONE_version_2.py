################################################################################################################
# MRAC HARDWARE 3 ( CREATE A PROJECTED LAYOUT OF FURNITURE WITH HAND GESTURES / FINITE STATE MACHINE_SYSTEM )
################################################################################################################


########################################################################################
##### TABLE OF CONTENTS #####
# 1. STATES (SETUP)
# 2. EVENTS (INPUT)
# 3. TRANSITIONS (CONDITIONS)


# !!! EVERY MODULE CONTAINS:
# 📥 VARIABLES INPUT:
# ✏️ VARIABLES DEFINED:
# 📡 VARIABLES OUTPUT:


################################################################################################################
# 1. STATES (SETUP)
################################################################################################################


##########################################
# 1.1 LIBRARIES
##########################################


# TOOLS WITH PREDETERMINED SETTINGS 🛠️


import cv2 # IMPORT CV2 FOR VIDEO PROCESSING
import numpy as np # IMPORT NUMPY FOR GEOMETRIC OPERATIONS
import mediapipe as mp # IMPORT MEDIAPIPE FOR HAND TRACKING
import copy  # IMPORT COPY OF INITIAL LAYOUT FOR RESET FUNCTIONALITY
import socket # FOR UDP COMMUNICATION WITH GRASSHOPPER
import os # INTERACT WITH OPERATING SYSTEM
import csv # READ & WRITE CSV FORMAT
import time # TIME FUNCTIONS
import json # CHANGE DATA FORMAT

##########################################
# 1.2 HARDWARE
##########################################


#####################
# 1.2.1 PROJECTOR 📽️
#####################


# CONNECT PROJECTOR -- GRASSHOPPER ( RENDERED DISPLAY FROM GRASSHOPPER 🦗 )


# 📥 VARIABLES INPUT: None
# ✏️ VARIABLES DEFINED: UDP_IP , UDP_PORT_GH , SEND_MESSAGE
# 📡 VARIABLES OUTPUT: SEND_MESSAGE , AF_INET , SOCK_DGRAM


# UDP = USER DATAGRAM PROTOCOL ( COMMUNICATION PROTOCOL )


UDP_IP = "127.0.0.1" # HOST ( INSIDE GRASSHOPPER FILE : lauren_v04_calibrate )
UDP_PORT_GH = 5000 # UPDATE THIS TO YOUR UDP PORT
SEND_MESSAGE = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # SEND MESSAGES TO PORT


###################
# 1.2.2 PC  💻
###################


# DISPLAY WORKSPACE WITH COMPUTER VISION


# 📥 VARIABLES INPUT: WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, BOX,
# ✏️ VARIABLES DEFINED: WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, BOX, INITIAL_BOXES
# 📡 VARIABLES OUTPUT: INITIAL_BOXES


# WORKSPACE BOUNDARIES
WORKSPACE_TOP_LEFT = (150, 100)      # WORKSPACE BOUNDING BOX TOP-LEFT CORNER
WORKSPACE_BOTTOM_RIGHT = (600, 400)  # WORKSPACE BOUNDING BOX BOTTOM-RIGHT CORNER


# CREATE BOXES OUTSIDE THE WORKSPACE ( A LIST OF DICTIONARIES )
BOX = [
    {"POS": [50, 100], "SIZE": 15, "COLOR": (0, 255, 0),   "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 150], "SIZE": 15, "COLOR": (255, 0, 0),   "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 200], "SIZE": 15, "COLOR": (0, 0, 255),   "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 250], "SIZE": 15, "COLOR": (255, 255, 0), "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 300], "SIZE": 15, "COLOR": (255, 0, 255), "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 350], "SIZE": 15, "COLOR": (0, 255, 255), "SELECTED": False, "ANGLE": 0},
    {"POS": [50, 400], "SIZE": 15, "COLOR": (128, 128, 128),"SELECTED": False, "ANGLE": 0},
    {"POS": [50, 450], "SIZE": 15, "COLOR": (128, 0, 128), "SELECTED": False, "ANGLE": 0},
    {"POS": [100,100], "SIZE": 15, "COLOR": (128, 128, 0), "SELECTED": False, "ANGLE": 0},
    {"POS": [100,150], "SIZE": 15, "COLOR": (255, 128,128),"SELECTED": False, "ANGLE": 0},
    {"POS": [100,200], "SIZE": 15, "COLOR": (128, 255,128),"SELECTED": False, "ANGLE": 0},
    {"POS": [100,250], "SIZE": 15, "COLOR": (128,128,255), "SELECTED": False, "ANGLE": 0},
    {"POS": [100,300], "SIZE": 15, "COLOR": (255,255,128),"SELECTED": False, "ANGLE": 0},
    {"POS": [100,350], "SIZE": 15, "COLOR": (255,128,255),"SELECTED": False, "ANGLE": 0},
    {"POS": [100,400], "SIZE": 15, "COLOR": (128,255,255),"SELECTED": False, "ANGLE": 0},
    {"POS": [100,400], "SIZE": 15, "COLOR": (192,192,192),"SELECTED": False, "ANGLE": 0},
]


# STORE A DEEP COPY OF THE ORIGINAL LAYOUT FOR RESET
INITIAL_BOXES = copy.deepcopy(BOX)    # STORE ORIGINAL LAYOUT FOR RESET



##########################################
# 2.3 GRASSHOPPER COMMUNICATION FUNCTIONS
##########################################

# 📥 VARIABLES INPUT: BOX (list of dictionaries containing box data), SEND_MESSAGE (socket object for UDP communication), UDP_IP (target IP address for Grasshopper), UDP_PORT_GH (target UDP port for Grasshopper)
# ✏️ VARIABLES DEFINED: format_box_data() (formats box data into JSON), send_box_data() (sends formatted data to Grasshopper), reset_and_notify() (resets layout and notifies Grasshopper), notify_event() (sends custom event notifications)
# 📡 VARIABLES OUTPUT: UDP packets containing JSON-formatted box data or event messages to Grasshopper

# REAL TIME CONTROL SYSTEM

# DATA FORMAT 

# FUNCTION: SEND BOX DATA TO GRASSHOPPER
def send_box_data():
    """
    Sends the current BOX layout to Grasshopper.
    """
    data = format_box_data(BOX)
    print(f"Sending to Grasshopper: {data}")  
    SEND_MESSAGE.sendto(data.encode('utf-8'), (UDP_IP, UDP_PORT_GH))


# FUNCTION: FORMAT BOX DATA FOR UDP
def format_box_data(box_list):
    """
    Converts the box list into a JSON string for Grasshopper.
    """
    data = {
        "POS": [box["POS"] for box in box_list],
        "ANGLE": [box["ANGLE"] for box in box_list]
    }
    json_string = json.dumps(data)
    print(f"Formatted JSON: {json_string}") 
    return json_string

# FUNCTION: RESET LAYOUT AND NOTIFY GRASSHOPPER
def reset_and_notify():
    """
    Resets the BOX layout and sends a 'reset' command to Grasshopper.
    """
    global BOX
    BOX = copy.deepcopy(INITIAL_BOXES)
    SEND_MESSAGE.sendto(b"RESET", (UDP_IP, UDP_PORT_GH))

# FUNCTION: NOTIFY A SPECIFIC EVENT TO GRASSHOPPER
def notify_event(event_name):
    """
    Sends a custom event string to Grasshopper.
    Useful for states like ROTATE, MOVE, etc.
    """
    message = f"EVENT:{event_name}"
    SEND_MESSAGE.sendto(message.encode(), (UDP_IP, UDP_PORT_GH))


###################
# 1.2.3 CAMERA 📷
###################


# 📥 VARIABLES INPUT: INDEX 0
# ✏️ VARIABLES DEFINED: CAP
# 📡 VARIABLES OUTPUT: CAP


# OPEN CAMERA
CAP = cv2.VideoCapture(0) # CAP OPENS THE DEFAULT CAMERA (INDEX 0)


################################################################################################################
# 2. EVENTS (INPUT)
################################################################################################################


# NOW THAT WE HAVE THE SET UP , THE WORKFLOW IS : FINGERS --> GESTURE --> STATE --> ACTION


######################################
# 2.1 CAMERA ( READ FINGERS )
######################################


# Finger tip landmark indices
FINGER_ID = [4, 8, 12, 16, 20]          # TIP_IDS FOR COUNTING FINGERS


# COUNT EXTENDED FINGERS
# VARIABLES INPUT: landmarks
# VARIABLES OUTPUT: FINGERS


def count_fingers(landmarks):
    FINGERS = 0                       # INITIALIZE COUNT
    # Thumb detection
    if landmarks[FINGER_ID[0]].x < landmarks[FINGER_ID[0] - 1].x: # DIFFERENTIATE THE THUMB INDEX BY X AXIS
        FINGERS += 1
    # Other fingers detection
    for tip in FINGER_ID[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            FINGERS += 1
    return FINGERS                  # RETURN NUMBER OF EXTENDED FINGERS


######################################
# 2.2 CAMERA ( FINGERS --> GESTURE )
######################################

# VARIABLES DEFINED: THUMB_UP, THUMB_DOWN, FIST, OPEN_HAND


FIST = "FIST"                       # FIST STATE
OPEN_HAND = "OPEN_HAND"             # OPEN HAND STATE
THUMB_UP = "THUMB_UP"               # THUMB UP STATE
THUMB_DOWN = "THUMB_DOWN"           # THUMB DOWN STATE
FSM_STATE = "ROTATE"


# VARIABLES INPUT: FSM_STATE, SELECTED_INDEX, MP_HANDS, HANDS, MP_DRAW, CAP, BOX, INITIAL_BOXES
# VARIABLES DEFINED: FSM_STATE, SELECTED_INDEX
# VARIABLES OUTPUT: HANDS


# INITIALIZE FSM_STATE TO IDLE
FSM_STATE = "IDLE"                    # CURRENT FSM STATE
# INITIALIZE SELECTED_INDEX TO NONE
SELECTED_INDEX = None                # CURRENTLY SELECTED SHAPE INDEX


# INITIALIZE HAND TRACKING  
# MP_HANDS HOLDS MEDIAPIPE HANDS SOLUTION
MP_HANDS = mp.solutions.hands
# HANDS IS THE ACTIVE MEDIAPIPE HANDS INSTANCE
HANDS = MP_HANDS.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6)
# MP_DRAW IS THE DRAW UTILITIES MODULE
MP_DRAW = mp.solutions.drawing_utils  # DRAWING LANDMARKS


################################################################################################################
# 3. TRANSITIONS (CONDITIONS)
################################################################################################################
# 📥 VARIABLES INPUT: POINT, SHAPE (dictionary containing shape data), ANGLE, SIZE (half the side length of a box), POS (position as tuple), x (x-coordinate), y (y-coordinate), WORKSPACE_TOP_LEFT (top-left corner of workspace), WORKSPACE_BOTTOM_RIGHT (bottom-right corner of workspace), BUFFER (padding distance)
# ✏️ VARIABLES DEFINED: BOX (list of dictionaries containing box data), WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT , ROTATED_PTS , RECT_I (rectangle i for overlap check), RECT_J (rectangle j for overlap check), RETVAL (intersection result), PX (point x-coordinate), PY (point y-coordinate), X_CLAMPED (clamped x-coordinate), Y_CLAMPED (clamped y-coordinate)
# 📡 VARIABLES OUTPUT: True/False from is_inside_shape (point inside shape check), True/False from is_overlapping (box overlap check), True/False from is_inside_workspace (workspace boundary check), Clamped Coordinates (x_clamped, y_clamped as tuple)

######################################
# 3.1 BOX DISPLAY RULES
######################################




# DONT PLACE A BOX INSIDE ANOTHER BOX  ( OVERLAPPING ! )


def is_inside_shape(point, shape):
    angle = shape["ANGLE"]
    size = shape["SIZE"]
    cx, cy = shape["POS"]
    pts = np.array([
        [cx - size, cy - size],
        [cx + size, cy - size],
        [cx + size, cy + size],
        [cx - size, cy + size]
    ], dtype=np.float32)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_pts = cv2.transform(np.array([pts]), M)[0]
    return cv2.pointPolygonTest(rotated_pts, point, False) >= 0


def is_overlapping(i, x, y):
    angle_i = BOX[i]["ANGLE"]
    size_i = BOX[i]["SIZE"]
    rect_i = ((x, y), (size_i * 2, size_i * 2), angle_i)


    for j, other in enumerate(BOX):
        if i != j:
            angle_j = other["ANGLE"]
            size_j = other["SIZE"]
            cx_j, cy_j = other["POS"]
            rect_j = ((cx_j, cy_j), (size_j * 2, size_j * 2), angle_j)


            retval, _ = cv2.rotatedRectangleIntersection(rect_i, rect_j)
            if retval != cv2.INTERSECT_NONE:
                return True
    return False


# DONT PLACE A BOX OUTSIDE THE WORKSPACE


def is_inside_workspace(x, y, size, angle):
    rect = ((x, y), (size * 2, size * 2), angle)
    box = cv2.boxPoints(rect)
    for point in box:
        px, py = point
        if not (WORKSPACE_TOP_LEFT[0] <= px <= WORKSPACE_BOTTOM_RIGHT[0] and
                WORKSPACE_TOP_LEFT[1] <= py <= WORKSPACE_BOTTOM_RIGHT[1]):
            return False
    return True


def clamp_to_workspace(x, y, size, buffer=0):
    x_clamped = max(WORKSPACE_TOP_LEFT[0] + size + buffer,
                    min(WORKSPACE_BOTTOM_RIGHT[0] - size - buffer, x))
    y_clamped = max(WORKSPACE_TOP_LEFT[1] + size + buffer,
                    min(WORKSPACE_BOTTOM_RIGHT[1] - size - buffer, y))
    return x_clamped, y_clamped




######################################
# 3.2 SAVE
######################################
# 📥 VARIABLES INPUT:  BOXES , POS , SIZE , ANGLE, COLOR , WORKSPACE_TOP_LEFT , WORKSPACE_BOTTOM_RIGHT , EXPORT_DIR
# ✏️ VARIABLES DEFINED: h, w , BLACK_FRAME , M , ROTATED_PTS , PNG_PATH , CSV_PATH , FILE , WRITER , X_NORM, Y_NORM ,
# 📡 VARIABLES OUTPUT: FOR THE FOLLOWING CODE : box_layout.png , box_data.csv ,


def save_layout(boxes, workspace_top_left, workspace_bottom_right, export_dir):
    # Create black background
    h, w = 480, 640  # Adjust as needed
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)


    # Draw workspace
    cv2.rectangle(black_frame, workspace_top_left, workspace_bottom_right, (200, 200, 200), 2)


    # Draw boxes
    for box in boxes:
        x, y = box["POS"]
        s = box["SIZE"]
        angle = box["ANGLE"]
        color = box["COLOR"]


        # Define box corners
        pts = np.array([
            [x - s, y - s],
            [x + s, y - s],
            [x + s, y + s],
            [x - s, y + s]
        ], dtype=np.float32)


        # Rotate box
        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rotated_pts = cv2.transform(np.array([pts]), M)[0].astype(int)


        # Draw box
        cv2.polylines(black_frame, [rotated_pts], isClosed=True, color=color, thickness=2)


    # Save image
    png_path = os.path.join(export_dir, "box_layout.png")
    cv2.imwrite(png_path, black_frame)


    # Save CSV
    csv_path = os.path.join(export_dir, "box_data.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x_norm", "y_norm", "angle"])
        for box in boxes:
            x, y = box["POS"]
            angle = box["ANGLE"]
            x_norm = (x - workspace_top_left[0]) / (workspace_bottom_right[0] - workspace_top_left[0])
            y_norm = (y - workspace_top_left[1]) / (workspace_bottom_right[1] - workspace_top_left[1])
            writer.writerow([x_norm, y_norm, angle])




######################################
# 3.3 MAIN LOOP
######################################
# 📥 VARIABLES INPUT: CAP (camera capture object), FINGER_ID (list of finger landmark indices), WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, INITIAL_BOXES (initial box configurations), BOX (list of dictionaries containing box data), MP_DRAW (MediaPipe drawing utilities), MP_HANDS (MediaPipe hands module), SELECTED_INDEX (index of selected box), HANDS (MediaPipe hands processor), FSM_STATE (finite state machine state)
# ✏️ VARIABLES DEFINED: ret (camera read success flag), frame (current video frame), h (frame height), w (frame width), rgb_frame (RGB-converted frame), results (hand detection results), hand_pos (hand center coordinates), fingers (count of extended fingers), last_rotation_time (timestamp of last rotation), current_time (current timestamp), item (current box dictionary), size (box size), angle (box rotation angle), x (x-coordinate), y (y-coordinate), export_dir (directory for saving layouts), SELECTED_INDEX (index of selected box), frame (processed video frame)
# 📡 VARIABLES OUTPUT: frame (displayed video frame with annotations), FSM_STATE (updated state of finite state machine), box_layout.png (saved layout image), box_data.csv (saved box data file)

# MAIN LOOP STARTS AND RUNS UNTIL ESC PRESSED
show_error = False
error_time = 0
last_rotation_time = 0  # Initialize outside loop to persist

while True:
    ret, frame = CAP.read()  # READ CAMERA FRAME
    if not ret:
        break

    print(frame.shape)

    frame = cv2.flip(frame, 1)  # MIRROR VIEW
    h, w, _ = frame.shape       # GET FRAME DIMENSIONS
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # CONVERT TO RGB
    results = HANDS.process(rgb_frame) # PROCESS HAND LANDMARKS
    hand_pos = None

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        cx = int(landmarks[9].x * w)  # COMPUTE CENTER X
        cy = int(landmarks[9].y * h)  # COMPUTE CENTER Y
        hand_pos = (cx, cy)
        fingers = count_fingers(landmarks)  # COUNT EXTENDED FINGERS

        # FIST: PICK (FILL THE BOX)
        if fingers == 0:
            FSM_STATE = "FIST"
            for i, item in enumerate(BOX):
                if is_inside_shape(hand_pos, item):
                    item["SELECTED"] = True
                    SELECTED_INDEX = i
                    break
        
        # OPEN HAND: PLACE
        elif fingers == 5:
            FSM_STATE = "OPEN_HAND"
            if SELECTED_INDEX is not None:
                item = BOX[SELECTED_INDEX]
                size = item["SIZE"]
                angle = item["ANGLE"]
                x, y = item["POS"]
                if is_inside_workspace(x, y, size, angle) and not is_overlapping(SELECTED_INDEX, x, y):
                    item["POS"] = [x, y]
                    item["SELECTED"] = False
                    SELECTED_INDEX = None
                else:
                    item["POS"] = INITIAL_BOXES[SELECTED_INDEX]["POS"]
                    item["ANGLE"] = INITIAL_BOXES[SELECTED_INDEX]["ANGLE"]
                    item["SELECTED"] = False
                    SELECTED_INDEX = None
                    show_error = True
                    error_time = time.time()

        # 2 FINGERS: ROTATE (IF HELD)
        elif fingers == 2 and SELECTED_INDEX is not None:
            current_time = time.time()
            if current_time - last_rotation_time >= 0.5:  # CHANGE TO 0.5 SECONDS
                FSM_STATE = "ROTATE"
                BOX[SELECTED_INDEX]["ANGLE"] += 45
                last_rotation_time = current_time
                notify_event("ROTATE")  # Notify Grasshopper of rotation

        # THUMBS UP: SAVE BLUEPRINT
        elif fingers == 1 and landmarks[FINGER_ID[0]].y < landmarks[FINGER_ID[0] - 1].y:
            FSM_STATE = "THUMB_UP"
            export_dir = r"C:\Users\billm\Desktop\CODE\coding\HARDWARE_3\GROUP HARDWARE 3\FSM_RESULTS"
            os.makedirs(export_dir, exist_ok=True)
            save_layout(BOX, WORKSPACE_TOP_LEFT, WORKSPACE_BOTTOM_RIGHT, export_dir)
            print("Saved box_layout.png and box_data.csv")

        # THUMBS DOWN: RESET
        elif fingers == 1 and landmarks[FINGER_ID[0]].y > landmarks[FINGER_ID[0] - 1].y:
            FSM_STATE = "THUMB_DOWN"
            reset_and_notify()
            SELECTED_INDEX = None
            print("Reset to initial layout")

        else:
            FSM_STATE = "IDLE"

        # MOVE ITEM IF PICKED
        if SELECTED_INDEX is not None and FSM_STATE == "FIST":
            BOX[SELECTED_INDEX]["POS"] = [cx, cy]

        MP_DRAW.draw_landmarks(frame,
                               results.multi_hand_landmarks[0],
                               MP_HANDS.HAND_CONNECTIONS)

    # Draw workspace and boxes
    cv2.rectangle(frame,
                  WORKSPACE_TOP_LEFT,
                  WORKSPACE_BOTTOM_RIGHT,
                  (200, 200, 200),
                  2)

    for item in BOX:
        angle = item["ANGLE"]
        pts = np.array([
            [item["POS"][0] - item["SIZE"], item["POS"][1] - item["SIZE"]],
            [item["POS"][0] + item["SIZE"], item["POS"][1] - item["SIZE"]],
            [item["POS"][0] + item["SIZE"], item["POS"][1] + item["SIZE"]],
            [item["POS"][0] - item["SIZE"], item["POS"][1] + item["SIZE"]]
        ], dtype=np.float32)
        M = cv2.getRotationMatrix2D((item["POS"][0], item["POS"][1]), angle, 1.0)
        pts2 = cv2.transform(np.array([pts]), M)[0].astype(int)

        if item["SELECTED"]:
            cv2.fillPoly(frame, [pts2], item["COLOR"])
        else:
            cv2.polylines(frame, [pts2], isClosed=True, color=item["COLOR"], thickness=2)

    send_box_data()

    # DISPLAY FSM STATE
    cv2.putText(frame, f'STATE: {FSM_STATE}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ERROR MESSAGE
    if show_error and time.time() - error_time < 5:
        cv2.putText(frame, "ERROR: Invalid placement", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif show_error:
        show_error = False

    cv2.imshow("2D HAND INTERACTION", frame)

    # EXIT ON ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


CAP.release()            # RELEASE CAMERA
cv2.destroyAllWindows()  # CLOSE ALL WINDOWS
SEND_MESSAGE.close()

