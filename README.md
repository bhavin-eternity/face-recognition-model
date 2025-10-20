# Face Recognition System

A **GUI-based Face Recognition System** built using Python, OpenCV, Tkinter, NumPy, and Pandas.  
This system allows users to **take images, train the recognizer, and detect faces in real-time**. Unknown faces are automatically saved for future reference.

---

## Features
- **Take Images**: Capture multiple images of a user with a unique ID and name.  
- **Train Images**: Train the LBPH Face Recognizer with captured images.  
- **Recognize Faces**: Detect and recognize faces in real-time using webcam.  
- **Save Unknown Faces**: Automatically stores images of faces not recognized by the system.  

---

## Folder Structure
FaceRecognitionSystem/
├── main.py # Main code
├── data/ # Haar cascade file
│ └── haarcascade_frontalface_default.xml
├── TrainingImage/ # Captured images of users
├── TrainingImageLabel/ # Trained recognizer file (Trainer.yml)
├── ImagesUnknown/ # Unknown faces captured automatically
└── UserDetails/ # CSV storing user IDs and Names
└── UserDetails.csv

## How to Run

1. **Install Python and Required Libraries**
   ```bash
pip install opencv-python numpy pandas pillow

2.Run the Program
 -> python main.py

3.Using the GUI

Enter ID (Numbers Only) and Name (Alphabets Only).

Click Take Samples to capture images.

Click Train Images to train the recognizer.

Click Recognize Face to start real-time detection.

Click Quit to close the application.
