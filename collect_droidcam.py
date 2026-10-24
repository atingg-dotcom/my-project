
import cv2
import mediapipe as mp
import time
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent / "dataset_extra"

SIGNS = {
    "1": "call_me",
    "2": "fist",
    "3": "heart",
    "4": "i_love_you",
    "5": "open_palm",
    "6": "peace",
    "7": "pointing",
    "8": "small",
    "9": "thumbs_up",
}

CAMERA_INDEXES = range(5)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Change this if the camera image is sideways:
# 0 = normal, 90 = clockwise, 180 = upside down,
# 270 = counter-clockwise.
ROTATION = 0

MIRROR_PREVIEW = True
SAVE_DELAY = 0.30
WINDOW_NAME = "DroidCam - Two-Hand Dataset Collector"

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def orient_frame(frame):
    """Rotate and optionally mirror the camera frame."""
    if ROTATION == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATION == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif ROTATION == 270:
        frame = cv2.rotate(
            frame, cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    if MIRROR_PREVIEW:
        frame = cv2.flip(frame, 1)

    return frame


def image_count(sign):
    folder = BASE_DIR / sign
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def draw_text(
    image, text, position,
    color=(255, 255, 255),
    scale=0.65, thickness=2
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# OPEN CAMERA
# ============================================================

camera = None
camera_index = None

try:
    for index in CAMERA_INDEXES:
        test_camera = cv2.VideoCapture(
            index, cv2.CAP_DSHOW
        )

        if test_camera.isOpened():
            ok, test_frame = test_camera.read()

            if ok and test_frame is not None:
                camera = test_camera
                camera_index = index
                break

        test_camera.release()

    if camera is None:
        raise RuntimeError(
            "Could not open a camera. Open DroidCam on your "
            "phone and connect it using the PC client."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # ========================================================
    # STATE
    # ========================================================

    current_sign = None
    last_saved_time = 0.0
    message = "Select a sign using keys 1-9"
    message_time = time.time()

    def show_message(text):
        global message, message_time
        message = text
        message_time = time.time()

    # State variables are initialized before the loop.
    hand_detected = False
    hand_inside_box = False
    hands_count = 0

    print("\nDroidCam connected!")
    print(f"Camera index: {camera_index}")
    print(f"Images will be saved to: {BASE_DIR}")
    print(f"Rotation: {ROTATION} degrees")
    print("\nCONTROLS")
    print("1-9 : Select a sign")
    print("SPACE : Capture an image")
    print("R : Clear selected sign")
    print("Q : Quit")
    print("\nLandmarks enabled for up to TWO hands.")

    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while True:
        ok, frame = camera.read()

        if not ok or frame is None:
            show_message("Cannot read camera frame")
            cv2.waitKey(50)
            continue

        # Orient the frame once. Use the same orientation for
        # detection, landmarks, box, preview, and saved images.
        clean_frame = orient_frame(frame)
        display = clean_frame.copy()

        height, width = display.shape[:2]

        # ----------------------------------------------------
        # HAND PLACEMENT BOX
        # ----------------------------------------------------

        box_w = int(width * 0.68)
        box_h = int(height * 0.76)

        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        # ----------------------------------------------------
        # MEDIAPIPE DETECTION
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            clean_frame, cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        hand_detected = bool(
            results.multi_hand_landmarks
        )

        hand_inside_box = False
        hands_count = (
            len(results.multi_hand_landmarks)
            if hand_detected else 0
        )

        # Draw BOTH hands, if detected, onto the displayed image.
        if hand_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                wrist = hand_landmarks.landmark[
                    mp_hands.HandLandmark.WRIST
                ]

                middle_base = hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                ]

                palm_x = int(
                    ((wrist.x + middle_base.x) / 2) * width
                )
                palm_y = int(
                    ((wrist.y + middle_base.y) / 2) * height
                )

                if x1 < palm_x < x2 and y1 < palm_y < y2:
                    hand_inside_box = True

        # ----------------------------------------------------
        # BOX AND STATUS
        # ----------------------------------------------------

        if not hand_detected:
            box_color = (0, 0, 255)
            hand_status = "NO HAND DETECTED"
        elif hand_inside_box:
            box_color = (0, 255, 0)
            hand_status = (
                f"{hands_count} HAND(S) TRACKED - READY"
            )
        else:
            box_color = (0, 255, 255)
            hand_status = "MOVE HAND(S) INSIDE BOX"

        cv2.rectangle(
            display, (x1, y1), (x2, y2),
            box_color, 3
        )

        if current_sign is None:
            sign_status = "No sign selected - press 1 to 9"
            count_status = ""
        else:
            sign_status = f"Sign: {current_sign}"
            count_status = (
                f"Images saved: {image_count(current_sign)}"
            )

        # Top panel
        cv2.rectangle(
            display, (0, 0), (width, 115),
            (25, 25, 25), -1
        )

        draw_text(
            display, sign_status, (15, 30),
            (0, 255, 255), 0.75, 2
        )
        draw_text(
            display, count_status, (15, 60),
            (255, 255, 255), 0.65, 2
        )
        draw_text(
            display, hand_status, (15, 92),
            box_color, 0.65, 2
        )

        # Bottom panel
        cv2.rectangle(
            display,
            (0, height - 42),
            (width, height),
            (25, 25, 25),
            -1,
        )

        draw_text(
            display,
            "1-9: Sign | SPACE: Capture | R: Reset | Q: Quit",
            (15, height - 14),
            (255, 255, 255),
            0.58,
            1,
        )

        # Temporary feedback message
        if time.time() - message_time < 2.5:
            cv2.rectangle(
                display, (0, 120), (width, 165),
                (25, 25, 25), -1
            )
            draw_text(
                display, message, (15, 150),
                (0, 255, 255), 0.65, 2
            )

        cv2.imshow(WINDOW_NAME, display)

        # ----------------------------------------------------
        # KEYBOARD INPUT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            current_sign = None
            show_message("Sign selection cleared")

        elif key < 128 and chr(key) in SIGNS:
            current_sign = SIGNS[chr(key)]

            (BASE_DIR / current_sign).mkdir(
                parents=True, exist_ok=True
            )

            show_message(f"Selected {current_sign}")
            print(f"Selected sign: {current_sign}")

        elif key == 32:
            if current_sign is None:
                show_message("Select a sign first using keys 1-9")
                continue

            if not hand_detected:
                show_message("Cannot save: no hand detected")
                continue

            if not hand_inside_box:
                show_message("Move your hand(s) inside the box")
                continue

            now = time.time()

            if now - last_saved_time < SAVE_DELAY:
                continue

            folder = BASE_DIR / current_sign
            folder.mkdir(parents=True, exist_ok=True)

            # Find the next unused number to avoid overwriting.
            existing = image_count(current_sign)
            filename = folder / (
                f"{current_sign}_{existing + 1:05d}.jpg"
            )

            # Save the clean image without overlays or landmarks.
            saved = cv2.imwrite(
                str(filename),
                clean_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if saved:
                last_saved_time = now
                show_message(
                    f"Saved {current_sign} #{existing + 1}"
                )
                print(f"Saved: {filename}")
            else:
                show_message("ERROR: Could not save image")

finally:
    if camera is not None:
        camera.release()

    hands.close()
    cv2.destroyAllWindows()

print("\nDataset collection stopped.")
print(f"Images are saved in: {BASE_DIR}")
