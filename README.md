# OSU-Final-Project

## DALL-E Image Generation and Resizing

This application leverages the DALL-E model by OpenAI to generate and resize images based on user prompts. The provided Python script `generate_and_resize_image.py` facilitates these functionalities.

### Dependencies
- `requests`: For HTTP requests
- `shutil`: For file operations
- `openai`: Access to the OpenAI API
- `cv2`: OpenCV for image processing
- `os`: OS interactions
- `dotenv`: Loading `.env` variables

### Setup
1. Create an OpenAI account and get an API key.
2. Save your API key in a `.env` file within the project directory as follows:

OPENAI_KEY=your_api_key_here

### Usage
- Execute the script `generate_and_resize_image.py`.
- Enter a prompt when prompted.
- An image will be generated and resized to 32x32 pixels, then saved as `test_resized.jpg`.

#### Example Command
bash
python generate_and_resize_image.py

## Real-Time Image Prediction Camera Application

This project develops a real-time camera application that identifies real or fake images and generates new images based on user prompts. It incorporates Python, Tkinter, OpenCV, and TensorFlow/Keras.

### Features
- Real-time video streaming
- Snapshot capture and image prediction
- Image generation using DALL-E

### Dependencies
- tkinter
- cv2
- PIL
- numpy
- tensorflow

### Setup and Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install numpy opencv-python pillow tensorflow

- Make sure your camera is connected and functioning properly.
- For image prediction, the application uses a pre-trained CNN model (`img_classifier.keras`). Ensure this model file is present in the project directory.
- The DALL-E image generation feature requires an internet connection and an API key for accessing the OpenAI API. Make sure to set up an account with OpenAI and obtain an API key.
- The application window is resizable and can be adjusted according to the user's preferences.


# Image Classifier using Convolutional Neural Networks

This project is aimed at building a convolutional neural network (CNN) to classify images into real and fake categories. The dataset consists of images of real and fake objects.

## Dependencies

This project utilizes the following Python libraries:

- **OpenCV (cv2)**: For reading and processing images.
- **NumPy**: For numerical computing and array manipulation.
- **TensorFlow**: Deep learning framework for building and training neural networks.
- **scikit-learn**: For data splitting and other utility functions.

Ensure these dependencies are installed before running the code.

## Dataset Structure

The dataset is organized into training and testing sets, each containing real and fake images. The images are stored in separate directories:

- **Training Set**:
  - Real images: Located in the "./data/train/REAL" directory.
  - Fake images: Located in the "./data/train/FAKE" directory.

- **Testing Set**:
  - Real images: Located in the "./data/test/REAL" directory.
  - Fake images: Located in the "./data/test/FAKE" directory.

## Data Preprocessing

The training and testing images are read into arrays using OpenCV (cv2). The images are then resized to a standard size of 32x32 pixels and stored in separate arrays for real and fake images.

## Model Architecture

The CNN model architecture consists of the following layers:

1. Convolutional Layer with 32 filters, kernel size (3, 3), and ReLU activation.
2. MaxPooling Layer (2, 2).
3. Convolutional Layer with 64 filters, kernel size (3, 3), and ReLU activation.
4. MaxPooling Layer (2, 2).
5. Convolutional Layer with 128 filters, kernel size (3, 3), and ReLU activation.
6. Flatten Layer to convert 2D feature maps to a 1D vector.
7. Dense Layer with 64 neurons and ReLU activation.
8. Output Layer with 1 neuron and Sigmoid activation.

The model is compiled with the Adam optimizer and binary crossentropy loss function.

## Training

The model is trained using the training set with 30 epochs and a validation split of 0.2. Training progress and metrics are printed after each epoch.

## Evaluation

The trained model is evaluated using the testing set to calculate loss and accuracy metrics.

- Test Loss: 0.2907
- Test Accuracy: 0.9154

- Train Loss: 0.1116
- Train Accuracy: 0.9600

## Saving and Exporting the Model

The trained model is saved as "img_classifier.keras" for future use.

## Prediction

New images can be classified using the trained model. The saved model is loaded, and new images are resized to the required dimensions (32x32) before prediction. Predictions are printed for each image.

## Example Usage

```python
# Load the trained model
model = models.load_model("img_classifier.keras")

# Predict new images
new_imgs = []

for filename in os.listdir("./data"):
    img_path = os.path.join("./data", filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            new_imgs.append(img)

new_imgs = np.array(new_imgs)

predictions = model.predict(new_imgs)

for i, pred in enumerate(predictions):
    print(f"Image {i+1} prediction: {pred[0]:.10f}")

## Example Output

Image 1 prediction: 0.0003023147 (Fake)
Image 2 prediction: 0.0000000217 (Real)


