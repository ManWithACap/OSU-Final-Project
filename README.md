# OSU-Final-Project
# Image Classifier using Convolutional Neural Networks

This project is aimed at building a convolutional neural network (CNN) to classify images into real and fake categories. The dataset consists of images of real and fake objects.

## Dependencies

This project utilizes the following Python libraries:

- **OpenCV (cv2)**: For reading and processing images.
- **NumPy**: For numerical computing and array manipulation.
- **TensorFlow**: Deep learning framework for building and training neural networks.
- **scikit-learn**: For data splitting and other utility functions.

Ensure these dependencies are installed before running the code.

## Dataset

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
Image 3 prediction: 0.0000016659 (Real)
Image 4 prediction: 0.0000041612 (Real)
Image 5 prediction: 0.0000000001 (Real)
Image 6 prediction: 0.0000958420 (Real)
Image 7 prediction: 0.0000008647 (Real)
Image 8 prediction: 0.0000013811 (Real)

