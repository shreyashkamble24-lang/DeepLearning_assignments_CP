# Assignment 01 – MNIST Digit Classification using Neural Network

## Aim

To implement a **Neural Network using TensorFlow and Keras** for recognizing handwritten digits from the **MNIST dataset**.

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains grayscale images of handwritten digits from **0 to 9**.

Each image has a size of:

```text
28 × 28 pixels
```

The dataset is divided into training and testing data using the built-in TensorFlow/Keras MNIST dataset loader.

## Neural Network Model

A Sequential Neural Network is created using TensorFlow/Keras.

### Model Architecture

```text
MNIST Image (28 × 28)
        ↓
     Flatten
        ↓
Dense Layer – 128 Neurons
     ReLU Activation
        ↓
Dense Layer – 10 Neurons
    Softmax Activation
        ↓
Predicted Digit (0–9)
```

### Model Configuration

* **Input:** 28 × 28 grayscale image
* **Flatten Layer:** Converts the image into a 1D vector
* **Hidden Layer:** 128 neurons
* **Activation:** ReLU
* **Output Layer:** 10 neurons
* **Output Activation:** Softmax
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Epochs:** 3
* **Metric:** Accuracy

## Methodology

The following steps are performed in the notebook:

1. Import TensorFlow and Matplotlib.
2. Load the MNIST dataset.
3. Separate the dataset into training and testing sets.
4. Normalize pixel values from **0–255 to 0–1**.
5. Visualize the dataset and display a sample handwritten digit.
6. Build a Sequential neural network.
7. Compile the model using the Adam optimizer and sparse categorical crossentropy loss.
8. Train the model for 3 epochs.
9. Evaluate the model on the test dataset.
10. Display the test accuracy.
11. Save the trained model.

## Data Preprocessing

The pixel values of the MNIST images originally range from **0 to 255**.

They are normalized using:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

This scales the pixel values to a range between **0 and 1**, making the data more suitable for neural network training.

## Model Training

The model is trained using:

```python
my_model.fit(train_images, train_labels, epochs=3)
```

The training process runs for **3 epochs** and tracks the model's accuracy during training.

## Model Evaluation

After training, the model is evaluated using the test dataset:

```python
val_loss, val_acc = my_model.evaluate(test_images, test_labels)
```

The notebook prints the resulting **test accuracy**.

## Saving the Model

The trained model is saved using:

```python
my_model.save('my_mnist_model')
```

This allows the trained model to be reused later without training it again.

## Visualization

The notebook visualizes an example MNIST image using Matplotlib:

```python
plt.imshow(train_images[1], cmap='gray')
```

This helps verify that the input data contains handwritten digit images.

## Technologies Used

* Python
* TensorFlow
* Keras
* Matplotlib
* Jupyter Notebook

## Libraries Required

Install the required libraries using:

```bash
pip install tensorflow matplotlib
```

## How to Run

1. Clone or download this repository.
2. Open the assignment folder.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```text
Assignment01_DL.ipynb
```

5. Run all cells sequentially.

The MNIST dataset will be downloaded automatically by TensorFlow/Keras when the dataset loader is executed.

## Project Structure

```text
Assignment-01/
│
├── Assignment01_DL.ipynb
└── README.md
```

## Result

The neural network successfully learns patterns from handwritten MNIST digit images and predicts one of the **10 digit classes (0–9)**. The final test accuracy is displayed after evaluating the trained model on the test dataset.

## Conclusion

This assignment demonstrates the basic implementation of a **feed-forward neural network for image classification** using TensorFlow and Keras. The MNIST dataset is normalized, passed through a Flatten layer and two Dense layers, and the trained model is evaluated on unseen test data.

The trained model is also saved for future use.
