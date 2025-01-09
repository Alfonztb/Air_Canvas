import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the actual frame size from the webcam
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    exit()

# Create a black canvas with the same dimensions as the frame
canvas = np.zeros(frame.shape, dtype=np.uint8)

# Previous coordinates of the index finger
px, py = 0, 0

# Define colors and create color buttons
colors = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Cyan": (255, 255, 0),
    "Purple": (255, 0, 255),
    "Yellow": (0, 255, 255),
    "Eraser": (0, 0, 0)  # Black color for eraser
}
button_width = 80
button_height = 60
button_margin = 10
button_y = 10

# Exit button configuration
exit_button_x = 10
exit_button_y = 10
exit_button_width = 80
exit_button_height = 40

# Current drawing color and thickness
current_color = list(colors.values())[0]
current_thickness = 2
max_thickness = 20  # Set maximum thickness
min_thickness = 1  # Set minimum thickness


def create_buttons(frame):
    # Draw exit button
    cv2.rectangle(frame, (exit_button_x, exit_button_y),
                  (exit_button_x + exit_button_width, exit_button_y + exit_button_height), (50, 50, 50), -1)
    cv2.putText(frame, "Exit", (exit_button_x + 10, exit_button_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2)

    # Draw color buttons
    for i, (color_name, color) in enumerate(colors.items()):
        x = exit_button_x + exit_button_width + (i * (button_width + button_margin)) + button_margin
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, button_y), (x + button_width, button_y + button_height), color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Add color name text
        text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        text_color = (0, 0, 0) if color_name != "Eraser" else (255, 255, 255)
        cv2.putText(frame, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    return frame


def draw_instructions(frame):
    instructions = (
        "Instructions:\n"
        "- Use Index Finger to draw.\n"
        "- Use 2 fingers to increase thickness.\n"
        "- Use 3 fingers to decrease thickness.\n"
        "- Close palm to erase.\n"
        "- Press 'q' to quit."
    )

    # Define the position and size for the instruction box
    box_x = frame.shape[1] - 300
    box_y = 10
    box_width = 290
    box_height = 120

    # Draw the instruction box
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)

    # Put the instructions text inside the box
    text_position_y = box_y + 20  # Start a bit lower to avoid overlap with the box's top
    for line in instructions.split('\n'):
        cv2.putText(frame, line, (box_x + 10, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_position_y += 20  # Move down for the next line


def check_button_press(x, y):
    for i, color in enumerate(colors.values()):
        button_x = exit_button_x + exit_button_width + (i * (button_width + button_margin)) + button_margin
        if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
            return color
    return None


def check_exit_button_press(x, y):
    """Check if the exit button is pressed based on the fingertip position."""
    if exit_button_x < x < exit_button_x + exit_button_width and exit_button_y < y < exit_button_y + exit_button_height:
        return True
    return False


def is_palm_open(hand_landmarks):
    """Check if the palm is open based on the positions of the fingers."""
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    bases = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
             mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]

    for tip, base in zip(tips, bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            return False
    return True


def count_raised_fingers(hand_landmarks):
    """Count how many fingers are raised (index, middle, ring, pinky)."""
    fingers = []

    # Check if index, middle, ring, and pinky are raised
    for tip, base in zip([mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP],
                         [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                          mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:  # If tip is higher than the base
            fingers.append(1)  # Finger is raised
        else:
            fingers.append(0)  # Finger is not raised

    return fingers.count(1)


def get_palm_bbox(hand_landmarks, frame_width, frame_height):
    """Get the bounding box of the palm based on hand landmarks."""
    palm_landmarks = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                      mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP,
                      mp_hands.HandLandmark.WRIST]

    x_min = min([hand_landmarks.landmark[lm].x for lm in palm_landmarks]) * frame_width
    y_min = min([hand_landmarks.landmark[lm].y for lm in palm_landmarks]) * frame_height
    x_max = max([hand_landmarks.landmark[lm].x for lm in palm_landmarks]) * frame_width
    y_max = max([hand_landmarks.landmark[lm].y for lm in palm_landmarks]) * frame_height

    return int(x_min), int(y_min), int(x_max), int(y_max)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Create color selection buttons and exit button
    frame = create_buttons(frame)

    # Draw instructions
    draw_instructions(frame)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            # Check if palm is open (eraser mode)
            if is_palm_open(hand_landmarks):
                # Get bounding box of the palm
                x_min, y_min, x_max, y_max = get_palm_bbox(hand_landmarks, frame.shape[1], frame.shape[0])
                # Erase the area covered by the palm on the canvas
                cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
            else:
                # Check if a color button is pressed
                new_color = check_button_press(x, y)
                if new_color is not None:
                    current_color = new_color
                    px, py = 0, 0  # Reset previous coordinates

                # Check if exit button is pressed
                if check_exit_button_press(x, y):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Handle thickness adjustments
                raised_fingers = count_raised_fingers(hand_landmarks)
                if raised_fingers == 2:  # If 2 fingers raised, increase thickness
                    current_thickness = min(current_thickness + 1, max_thickness)
                elif raised_fingers == 3:  # If 3 fingers raised, decrease thickness
                    current_thickness = max(current_thickness - 1, min_thickness)
                else:
                    # Draw on the canvas
                    if px == 0 and py == 0:
                        px, py = x, y
                    else:
                        cv2.line(canvas, (px, py), (x, y), current_color, current_thickness)
                    px, py = x, y
    else:
        # Reset previous coordinates when hand is not detected
        px, py = 0, 0

    # Combine the canvas and the frame
    combined_image = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Display the result
    cv2.imshow("Air Canvas", combined_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
