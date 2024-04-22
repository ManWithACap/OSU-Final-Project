import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import dalle as ai
from tensorflow.keras.models import load_model

class CameraApp:
    def __init__(self, window, window_title):

        self.aiImage = "neither"

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

        self.btn_external_function = tk.Button(window, text="Generate Image", width=15, command=self.predictAI)
        self.btn_external_function.pack(anchor=tk.CENTER)
        
        self.param_entry = tk.Entry(window, width=20)
        self.param_entry.pack(anchor=tk.CENTER, expand=True)
        self.param_entry.insert(0, "a picture of an apple")
        
        self.photo_label = tk.Label(window)
        self.photo_label.pack(anchor=tk.CENTER, expand=True)

        self.predict_label = tk.Label(window, text="PREDICTION RESULT", width=20, font=("Consolas", 24))
        self.predict_label.place(x=100, y=100)
        
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
        self.aiImage = "no"
    
    def predict(self):
        root.config(cursor="wait")
        image = ""
        if self.aiImage == "yes":
            image = cv2.imread("test_resized.jpg")
        elif self.aiImage == "no":
            image = cv2.imread("snapshot.png")
            image = cv2.resize(image, (32, 32))
        else:
            return
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)[0][0]  # Make prediction
        print(prediction)

        if prediction >= 0.5:
            result_label = "Real"
            color = "green"
        else:
            result_label = "Fake"
            color = "red"
        
        self.predict_label.config(text=result_label, foreground=color)
        root.config(cursor="")
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def predictAI(self):
        root.config(cursor="wait")
        param = self.param_entry.get()
        filename = ai.generate_and_resize_image(param)
        self.show_snapshot(filename.replace("_resized", ""))
        self.aiImage = "yes"
        root.config(cursor="")

# Create a window and pass it to the CameraApp class
root = tk.Tk()
root.state("zoomed")
app = CameraApp(root, "Camera App")
