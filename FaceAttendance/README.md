# Face Attendance System

A simple real-time face recognition attendance system using Python, OpenCV and face_recognition.

## Features

- Real-time webcam recognition
- Student-specific face folders
- CSV attendance storage
- One attendance entry per student per day
- Unknown face detection
- CPU-friendly HOG face detector

## Project structure

FaceAttendance/
├── app.py
├── requirements.txt
├── known_faces/
│   ├── Umesh/
│   │   └── umesh.jpg
│   └── Rahul/
│       └── rahul.jpg
└── attendance/
    └── attendance.csv

## Setup

1. Install Python 3.10 or 3.11.
2. Open PowerShell in this project folder.
3. Create a virtual environment:

   python -m venv .venv

4. Activate it:

   .\.venv\Scripts\Activate.ps1

5. Upgrade pip:

   python -m pip install --upgrade pip

6. Install dependencies:

   pip install -r requirements.txt

7. Add student photos:

   known_faces/
   └── StudentName/
       └── photo.jpg

Use a clear front-facing image with only one person.

## Run

Activate the environment and run:

   python app.py

Press Q to quit.

## Phone camera

The app starts with CAMERA_INDEX = 0.

If your phone is connected to the PC as a webcam and it is not camera 0, open app.py and change:

   CAMERA_INDEX = 0

to:

   CAMERA_INDEX = 1

or 2, etc.

Important: the phone must be exposed to Windows as a webcam/video capture device. A normal phone USB connection does not automatically make its camera available to OpenCV.

For Android, use a webcam/phone-camera app that provides a Windows webcam device, or Windows-supported phone webcam functionality. After connecting, check Windows Camera app first.

## CSV

Attendance is saved automatically to:

attendance/attendance.csv

Columns:

Name, Date, Time

A student is marked only once per calendar day.

## Troubleshooting

Camera doesn't open:
- Test the phone camera in the Windows Camera app.
- Try CAMERA_INDEX = 1 or 2.
- Close other applications using the camera.

No faces loaded:
- Check the folder structure.
- Use JPG/PNG images.
- Make sure each photo contains exactly one face.

Slow recognition:
- This project uses the CPU HOG detector.
- Keep the camera resolution moderate.
- Avoid putting many faces into the database while testing.
