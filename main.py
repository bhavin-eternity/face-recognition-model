# importing libraries
import tkinter as tk
from tkinter import Message
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd

# ----------------- AUTO FOLDER CREATION -----------------
folders = ["data", "TrainingImage", "TrainingImageLabel", "UserDetails", "ImagesUnknown"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create CSV file if not exists
csv_path = os.path.join("UserDetails", "UserDetails.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Name"])

# ----------------- GUI SETUP -----------------
window = tk.Tk()
window.title("Face Recogniser")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Face Recognition System",
                   bg="green", fg="white", width=50,
                   height=3, font=('times', 30, 'bold'))
message.place(x=200, y=20)

lbl = tk.Label(window, text="ID (Numbers Only)", width=20, height=2,
               fg="green", bg="white", font=('times', 15, ' bold '))
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg="white", fg="green", font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Name (Alphabets Only)", width=20,
                fg="green", bg="white", height=2,
                font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg="white", fg="green",
                font=('times', 15, ' bold '))
txt2.place(x=700, y=315)

# ----------------- HELPER FUNCTIONS -----------------
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# ----------------- TAKE IMAGES -----------------
def TakeImages():
    Id = txt.get()
    name = txt2.get()

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = os.path.join("data", "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(os.path.join("TrainingImage", f"{name}.{Id}.{sampleNum}.jpg"),
                            gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break

        cam.release()
        cv2.destroyAllWindows()

        row = [int(Id), name]  # Save ID as integer
        with open(csv_path, "a+", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")

    else:
        message.configure(text="Enter Numeric ID and Alphabetical Name only!")

# ----------------- TRAIN IMAGES -----------------
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save(os.path.join("TrainingImageLabel", "Trainer.yml"))
    message.configure(text="Image Training Completed")

# ----------------- TEST IMAGES -----------------
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join("TrainingImageLabel", "Trainer.yml"))
    harcascadePath = os.path.join("data", "haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv(csv_path)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            # Debug info
            print(f"Predicted ID: {Id}, Confidence: {conf}")

            if conf < 80:  # more forgiving threshold
                name = df.loc[df['Id'] == Id]['Name'].values[0]
                text = f"{Id} - {name}"
            else:
                text = "Unknown"
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(os.path.join("ImagesUnknown", f"Image{noOfFile}.jpg"),
                            im[y:y + h, x:x + w])

            cv2.putText(im, str(text), (x, y + h + 30), font, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', im)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC to exit
            break

    cam.release()
    cv2.destroyAllWindows()

# ----------------- BUTTONS -----------------
takeImg = tk.Button(window, text="Sample", command=TakeImages,
                    fg="white", bg="green", width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Training", command=TrainImages,
                     fg="white", bg="green", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)

trackImg = tk.Button(window, text="Testing", command=TrackImages,
                     fg="white", bg="green", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy,
                       fg="white", bg="green", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

window.mainloop()