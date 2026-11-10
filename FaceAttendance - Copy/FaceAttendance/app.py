import cv2
import os
import csv
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PHONE_CAMERA_URL = "http://put ur webcam adress/video"

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"

MODEL_FILE = "face_model.yml"
NAMES_FILE = "face_names.txt"

CSV_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

# Lower value = stricter recognition
RECOGNITION_THRESHOLD = 70


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ============================================================
# FACE DETECTOR
# ============================================================

CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    print("ERROR: Could not load Haar Cascade.")
    exit()


# ============================================================
# TRAIN FACE RECOGNITION MODEL
# ============================================================


def train_model():

    print("\n========================================")
    print("        TRAINING FACE MODEL")
    print("========================================")

    if not hasattr(cv2, "face"):
        print("\nERROR: OpenCV face module is missing.")
        print("Install opencv-contrib-python.")
        exit()

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    names = {}
    current_id = 0

    student_folders = os.listdir(KNOWN_FACES_DIR)

    for student_name in student_folders:

        student_path = os.path.join(KNOWN_FACES_DIR, student_name)

        if not os.path.isdir(student_path):
            continue

        print(f"\nLoading student: {student_name}")

        names[current_id] = student_name

        image_files = os.listdir(student_path)

        student_face_count = 0

        for image_file in image_files:

            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(student_path, image_file)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_file}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            detected_faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(detected_faces) == 0:

                print(f"  No face found: {image_file}")

                continue

            if len(detected_faces) > 1:

                print(f"  Multiple faces found: {image_file}")

                continue

            x, y, w, h = detected_faces[0]

            face = gray[y : y + h, x : x + w]

            faces.append(face)
            labels.append(current_id)

            student_face_count += 1

            print(f"  Added: {image_file}")

        print(f"  Faces loaded: {student_face_count}")

        current_id += 1

    if len(faces) == 0:

        print("\nERROR: No usable faces found.")

        print("\nAdd photos like:")

        print("""
known_faces/
├── akash/
│   └── akash.jpg
│
├── sunil/
│   └── sunil.jpg
│
└── majhi/
    └── majhi.jpg
""")

        exit()

    print("\nTraining model...")

    recognizer.train(faces, np.array(labels))

    recognizer.save(MODEL_FILE)

    with open(NAMES_FILE, "w", encoding="utf-8") as file:

        for student_id, student_name in names.items():

            file.write(f"{student_id}|{student_name}\n")

    print("\nModel trained successfully.")
    print(f"Total training images: {len(faces)}")

    return recognizer, names


# ============================================================
# LOAD / TRAIN MODEL
# ============================================================

recognizer, names = train_model()


# ============================================================
# ATTENDANCE CSV
# ============================================================


def create_csv():

    if not os.path.exists(CSV_FILE):

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow(["Name", "Date", "Time"])


# ============================================================
# MARK ATTENDANCE
# ============================================================


def mark_attendance(name):

    create_csv()

    today = datetime.now().strftime("%Y-%m-%d")

    # Check whether already marked today
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:

        reader = csv.reader(file)

        next(reader, None)

        for row in reader:

            if len(row) >= 2:

                existing_name = row[0]
                existing_date = row[1]

                if existing_name == name and existing_date == today:

                    return False

    current_time = datetime.now().strftime("%H:%M:%S")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([name, today, current_time])

    print(f"[ATTENDANCE] {name} marked present at {current_time}")

    return True


# ============================================================
# START PHONE CAMERA
# ============================================================

print("\n========================================")
print("       FACE ATTENDANCE SYSTEM")
print("========================================")

print("\nConnecting to phone camera...")

print(f"Camera URL: {PHONE_CAMERA_URL}")

camera = cv2.VideoCapture(PHONE_CAMERA_URL)

if not camera.isOpened():

    print("\nERROR: Could not connect to phone camera.")

    print("\nCheck:")
    print("1. Phone and laptop are on the same Wi-Fi.")
    print("2. IP Webcam server is running.")
    print("3. Browser can open:")
    print("   http://:8080")
    print("4. The IP address has not changed.")

    exit()


print("\nPhone camera connected.")
print("Starting attendance system...")
print("Press Q to quit.")


# ============================================================
# CAMERA SETTINGS
# ============================================================

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("\nERROR: Could not read camera frame.")

        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for x, y, w, h in detected_faces:

        # Face recognition
        face = gray[y : y + h, x : x + w]

        student_id, confidence = recognizer.predict(face)

        name = "Unknown"

        if confidence < RECOGNITION_THRESHOLD:

            if student_id in names:

                name = names[student_id]

                mark_attendance(name)

        # Convert coordinates back to original frame
        x *= 2
        y *= 2
        w *= 2
        h *= 2

        # Unknown / recognized status
        if name == "Unknown":

            box_color = (0, 0, 255)

        else:

            box_color = (0, 255, 0)

        # Face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Name background
        cv2.rectangle(frame, (x, y + h - 40), (x + w, y + h), box_color, cv2.FILLED)

        # Name
        cv2.putText(
            frame,
            name,
            (x + 8, y + h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Confidence
        cv2.putText(
            frame,
            f"Score: {confidence:.1f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )

    # Header
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (30, 30, 30), -1)

    cv2.putText(
        frame,
        "FACE ATTENDANCE SYSTEM",
        (20, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (frame.shape[1] - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    # Display
    cv2.imshow("Face Attendance System", frame)

    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nAttendance system closed.")
print(f"Attendance file: {CSV_FILE}")
