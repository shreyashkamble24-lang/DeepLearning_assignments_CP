# Assignment 06 – Tomato Leaf Disease Classification using CNN

## Aim

To design and implement a **Convolutional Neural Network (CNN)** for image classification using a **Tomato Leaf Disease Dataset**.

The model is trained to classify tomato leaf images into the disease classes available in the dataset.

## Dataset

The project uses the **Tomato Leaf Dataset for Disease Detection** downloaded through KaggleHub.

* **Dataset:** Tomato Leaf Dataset for Disease Detection
* **Image Type:** Tomato leaf images
* **Image Size:** 128 × 128 pixels
* **Image Channels:** 3 (RGB)
* **Batch Size:** 32
* **Validation Split:** 20%
* **Training Split:** 80%
* **Random Seed:** 123

The dataset is loaded automatically using TensorFlow's `image_dataset_from_directory()` function.

## CNN Architecture

The CNN model consists of multiple convolutional and pooling layers followed by fully connected layers.

```text
Input Image
128 × 128 × 3
       ↓
Rescaling (1/255)
       ↓
Conv2D – 32 Filters
3 × 3, ReLU
       ↓
MaxPooling2D
2 × 2
       ↓
Conv2D – 64 Filters
3 × 3, ReLU
       ↓
MaxPooling2D
2 × 2
       ↓
Conv2D – 128 Filters
3 × 3, ReLU
       ↓
MaxPooling2D
2 × 2
       ↓
Flatten
       ↓
Dense – 128 Neurons
ReLU
       ↓
Dropout – 0.5
       ↓
Output Layer
Softmax
```

## Model Configuration

The CNN is compiled with:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy
* **Epochs:** 10
* **Batch Size:** 32
* **Dropout:** 0.5
* **Image Size:** 128 × 128

The output layer automatically uses the number of classes detected from the dataset.

## Data Preprocessing

The images are resized to:

```text
128 × 128
```

Pixel values are normalized using:

```python
layers.Rescaling(1./255)
```

This converts the original pixel values from the range **0–255** to approximately **0–1**.

The dataset is divided into:

```text
80% → Training
20% → Validation
```

## Methodology

The notebook follows these steps:

1. Download the tomato leaf dataset using KaggleHub.
2. Locate the dataset directory.
3. Load the images using TensorFlow.
4. Resize images to 128 × 128 pixels.
5. Divide the dataset into training and validation sets.
6. Detect the class names automatically.
7. Cache and prefetch the datasets for better performance.
8. Build the CNN architecture.
9. Compile the model using Adam optimizer.
10. Train the CNN for 10 epochs.
11. Record training and validation accuracy.
12. Record training and validation loss.
13. Plot the accuracy and loss curves.

## Model Training

The model is trained for **10 epochs**:

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```

The training history is stored in the `history` variable and is used to analyze the model's performance.

## Performance Visualization

Two graphs are generated after training.

### 1. Accuracy

The accuracy graph compares:

* Training Accuracy
* Validation Accuracy

This helps observe how well the CNN learns the classification task.

### 2. Loss

The loss graph compares:

* Training Loss
* Validation Loss

This helps identify the learning behavior of the model and possible overfitting.

## Dataset Performance Optimization

The TensorFlow datasets are optimized using:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

Caching and prefetching help improve the data input pipeline during model training.

## Technologies Used

* Python
* TensorFlow
* Keras
* Matplotlib
* KaggleHub
* Jupyter Notebook

## Installation

Install the required libraries using:

```bash
pip install tensorflow matplotlib kagglehub
```

## How to Run

1. Clone or download the repository.
2. Open the Assignment 6 folder.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```text
Assignment06_DL.ipynb
```

5. Run all cells sequentially.

The dataset will be downloaded automatically using KaggleHub.

## Project Structure

```text
Assignment-06/
│
├── Assignment06_DL.ipynb
└── README.md
```

The dataset does not need to be manually placed in the project because it is downloaded using KaggleHub during execution.

## Result

The CNN learns image features from tomato leaf images and performs multi-class classification based on the classes detected from the dataset.

The notebook displays:

* Detected dataset classes
* CNN model summary
* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Accuracy and loss graphs

## Conclusion

This assignment demonstrates how a **Convolutional Neural Network can be designed and implemented for plant disease image classification**.

The CNN uses convolutional layers to extract visual features from tomato leaf images, pooling layers to reduce spatial dimensions, and dense layers to perform the final classification. Dropout is also used to help reduce overfitting.

The model's learning performance is analyzed using training and validation accuracy and loss curves.
