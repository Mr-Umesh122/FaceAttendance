import cv2
import face_recognition
import os
import csv
from datetime import datetime

KNOWN_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
CSV_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def load_known_faces():
    encodings = []
    names = []

    for name in os.listdir(KNOWN_DIR):
        person_dir = os.path.join(KNOWN_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image)

            if len(locations) != 1:
                print(f"[SKIP] {path}: expected 1 face, found {len(locations)}")
                continue

            encoding = face_recognition.face_encodings(image, locations)[0]
            encodings.append(encoding)
            names.append(name)
            print(f"[OK] Loaded: {name} <- {filename}")

    return encodings, names

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    ensure_csv()
    today = datetime.now().strftime("%Y-%m-%d")

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    for row in rows[1:]:
        if len(row) >= 2 and row[0] == name and row[1] == today:
            return False

    now = datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

    print(f"[ATTENDANCE] {name} marked present.")
    return True

print("Loading known faces...")
known_encodings, known_names = load_known_faces()

if not known_encodings:
    print("\nNo valid student faces found.")
    print("Add photos inside known_faces/<StudentName>/ and restart.")
    raise SystemExit

print(f"\nLoaded {len(known_encodings)} face(s).")
print("Starting camera...")

# Default camera. Change to 1, 2, etc. if needed.
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open camera.")
    print("If using a phone camera app, make sure it exposes the phone camera as a webcam.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

process_this_frame = True
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera.")
        break

    # Process every other frame to reduce CPU usage.
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=0.50
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )

            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            frame, name, (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )

    cv2.putText(
        frame, "Face Attendance System | Press Q to quit",
        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
