# Assignment 03 – Fashion-MNIST Classification using CNN

## Aim

To implement a **Convolutional Neural Network (CNN)** using TensorFlow and Keras for classifying images from the **Fashion-MNIST dataset** into different clothing and footwear categories.

## Dataset

The project uses the **Fashion-MNIST dataset** provided by TensorFlow/Keras.

The dataset contains grayscale images of fashion items.

* **Image Size:** 28 × 28 pixels
* **Number of Classes:** 10
* **Training Dataset:** 60,000 images
* **Testing Dataset:** 10,000 images
* **Task:** Multi-class image classification

### Classes

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

## CNN Model Architecture

The assignment uses the following CNN architecture:

```text
Input Image
28 × 28 × 1
      ↓
Conv2D – 32 Filters
3 × 3 Kernel
ReLU Activation
      ↓
MaxPooling2D
2 × 2
      ↓
Conv2D – 64 Filters
3 × 3 Kernel
ReLU Activation
      ↓
MaxPooling2D
2 × 2
      ↓
Flatten
      ↓
Dense – 128 Neurons
ReLU Activation
      ↓
Dropout – 50%
      ↓
Dense – 10 Neurons
Softmax Activation
      ↓
Predicted Fashion Class
```

## Data Preprocessing

The images are converted to floating-point values and normalized from the range **0–255 to 0–1**.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

The images are then reshaped to include the channel dimension:

```python
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

This produces the input format:

```text
28 × 28 × 1
```

where `1` represents the grayscale channel.

## Model Configuration

The CNN is compiled using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Evaluation Metric:** Accuracy
* **Epochs:** 10
* **Batch Size:** 64
* **Validation Split:** 20%
* **Dropout:** 0.5

## Methodology

The notebook follows these steps:

1. Import TensorFlow, NumPy, and Matplotlib.
2. Load the Fashion-MNIST dataset.
3. Normalize the image pixel values.
4. Reshape the images for CNN input.
5. Define the 10 Fashion-MNIST class names.
6. Visualize sample images from the dataset.
7. Build the CNN model.
8. Compile the model using Adam and sparse categorical crossentropy.
9. Train the model for 10 epochs.
10. Use 20% of the training data for validation.
11. Evaluate the trained model using the test dataset.
12. Display test accuracy and test loss.
13. Plot training and validation accuracy.
14. Plot training and validation loss.
15. Make predictions on test images.
16. Compare a predicted class with its actual class.
17. Save the trained CNN model.

## Model Training

The model is trained using:

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
```

The training history is used to visualize the performance of the model during training.

## Evaluation

The trained model is evaluated on the test dataset:

```python
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)
```

The following metrics are displayed:

* Test Accuracy
* Test Loss

## Visualizations

### 1. Sample Images

Five Fashion-MNIST images are displayed along with their corresponding class names.

### 2. Model Accuracy

A graph is generated showing:

* Training Accuracy
* Validation Accuracy

This helps understand how the model's classification performance changes across epochs.

### 3. Model Loss

A graph is generated showing:

* Training Loss
* Validation Loss

This helps analyze the learning behavior of the CNN.

### 4. Prediction

The model predicts the class of a selected test image and displays:

```text
Predicted : <Class>
Actual    : <Class>
```

## Prediction

Predictions are generated using:

```python
prediction = model.predict(x_test)
```

The predicted class is obtained using `np.argmax()`:

```python
predicted_label = np.argmax(prediction[index])
```

The predicted class is then compared with the actual test label.

## Model Saving

After training, the CNN model is saved as:

```text
fashion_mnist_cnn.keras
```

using:

```python
model.save("fashion_mnist_cnn.keras")
```

This allows the trained model to be reused later without retraining it.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## Installation

Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib
```

## How to Run

1. Clone or download the repository.
2. Open the Assignment 3 folder.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```text
Assignment03_DL.ipynb
```

5. Run all cells sequentially.

The Fashion-MNIST dataset will be loaded automatically through TensorFlow/Keras.

## Project Structure

```text
Assignment-03/
│
├── Assignment03_DL.ipynb
├── fashion_mnist_cnn.keras
└── README.md
```

> `fashion_mnist_cnn.keras` will be created after running the notebook.

## Result

The CNN learns visual features from Fashion-MNIST images and classifies them into one of the **10 fashion categories**. The notebook displays the test accuracy and loss, training/validation graphs, and an example prediction showing the predicted and actual class.

## Conclusion

This assignment demonstrates the use of a **Convolutional Neural Network for image classification**. CNN layers are used to automatically learn image features, while pooling reduces the spatial dimensions. A Dense layer performs the final classification, and Dropout helps reduce overfitting.

The trained model is evaluated on unseen Fashion-MNIST test images and saved as a `.keras` model for future use.
