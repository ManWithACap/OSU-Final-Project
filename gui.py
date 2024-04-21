import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        self.btn_snapshot = tk.Button(window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_predict = tk.Button(window, text="Predict", width=10, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)
        
        self.photo_label = tk.Label(window)
        self.photo_label.pack()

        self.predict_label = tk.Label(window, text="PREDICTION RESULT")
        self.predict_label.pack()
        
        self.model = load_model("img_classifier.keras")
        
        self.update()
        self.window.mainloop()
    
    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            Image.fromarray(frame_rgb).save("snapshot.png")  # Save RGB image
            self.show_snapshot("snapshot.png")
    
    def show_snapshot(self, image_path):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        self.photo_label.config(image=photo)
        self.photo_label.image = photo
    
    def predict(self):
        image = cv2.imread("snapshot.png")
        image = cv2.resize(image, (32, 32))  # Resize image to match the input size of your model
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)[0][0]  # Make prediction
        print(prediction)

        if prediction <= 0.5:
            result_label = "Real"
            color = "green"
        else:
            result_label = "Fake"
            color = "red"
        
        self.predict_label.config(text=result_label, foreground=color)
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

# Create a window and pass it to the CameraApp class
root = tk.Tk()
app = CameraApp(root, "Camera App")
