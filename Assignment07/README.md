# Assignment 07 – Transfer Learning on CIFAR-10

## Aim

To implement and compare **transfer learning using pretrained deep learning models** for image classification on the **CIFAR-10 dataset**.

The following pretrained architectures are evaluated:

* AlexNet
* VGG16
* ResNet50
* EfficientNetB0

The models use pretrained ImageNet weights, with their final classification layers replaced to perform **10-class CIFAR-10 classification**.

---

## Dataset

The project uses the **CIFAR-10 dataset** provided by Torchvision.

CIFAR-10 contains images belonging to 10 different classes.

### Classes

| Label | Class      |
| ----: | ---------- |
|     0 | Airplane   |
|     1 | Automobile |
|     2 | Bird       |
|     3 | Cat        |
|     4 | Deer       |
|     5 | Dog        |
|     6 | Frog       |
|     7 | Horse      |
|     8 | Ship       |
|     9 | Truck      |

### Dataset Configuration

* **Training samples used:** 5,000
* **Validation samples:** 1,000
* **Testing samples:** 1,000
* **Training/Validation split:** 80% / 20%
* **Image size:** 224 × 224
* **Number of classes:** 10
* **Batch size:** 16
* **Random state:** 42

> The notebook intentionally uses a subset of CIFAR-10 rather than the complete dataset to make comparison between the four pretrained models more manageable.

---

## Transfer Learning

Transfer learning uses knowledge learned from a previously trained model and adapts it to a new classification problem.

In this assignment, all four models use **pretrained ImageNet weights**.

The pretrained parameters are frozen, and the final classification layer is replaced with a new layer containing **10 output classes** for CIFAR-10.

### Transfer Learning Process

```text
Pretrained ImageNet Model
          ↓
Freeze Pretrained Parameters
          ↓
Replace Final Classification Layer
          ↓
10 CIFAR-10 Classes
          ↓
Train New Classifier
          ↓
Evaluate Model
```

---

# Models Used

## 1. AlexNet

A classic convolutional neural network architecture pretrained on ImageNet.

The final classifier layer is replaced with a 10-class output layer.

```text
AlexNet
   ↓
Pretrained Feature Extractor
   ↓
New Classifier
   ↓
10 CIFAR-10 Classes
```

---

## 2. VGG16

VGG16 is a deep CNN architecture consisting of multiple convolutional layers.

The pretrained feature parameters are frozen and the final classification layer is modified for CIFAR-10.

```text
VGG16
   ↓
Pretrained Features
   ↓
Modified Classifier
   ↓
10 Classes
```

---

## 3. ResNet50

ResNet50 is a deep residual network that uses residual connections to improve the training of deep networks.

The original 1000-class ImageNet output layer is replaced with a 10-class CIFAR-10 classifier.

```text
ResNet50
   ↓
Pretrained Feature Extractor
   ↓
New Fully Connected Layer
   ↓
10 Classes
```

---

## 4. EfficientNetB0

EfficientNetB0 is a computationally efficient CNN architecture that provides a strong balance between model size and performance.

The final classifier is replaced with a 10-class output layer.

```text
EfficientNetB0
      ↓
Pretrained Features
      ↓
Modified Classifier
      ↓
10 Classes
```

---

# Image Preprocessing

Since the pretrained ImageNet models expect larger input images, CIFAR-10 images are resized from:

```text
32 × 32
```

to:

```text
224 × 224
```

### Training Transform

Training images undergo:

* Resize to 224 × 224
* Random horizontal flip
* Conversion to tensor
* ImageNet normalization

### Testing/Validation Transform

Validation and test images undergo:

* Resize to 224 × 224
* Conversion to tensor
* ImageNet normalization

ImageNet normalization uses:

```text
Mean = [0.485, 0.456, 0.406]

Std  = [0.229, 0.224, 0.225]
```

---

# Model Configuration

The main configuration used in the notebook is:

| Parameter          | Value              |
| ------------------ | ------------------ |
| Image Size         | 224 × 224          |
| Batch Size         | 16                 |
| Epochs             | 3                  |
| Learning Rate      | 0.001              |
| Optimizer          | Adam               |
| Loss Function      | Cross Entropy Loss |
| Number of Classes  | 10                 |
| Random State       | 42                 |
| Pretrained Weights | ImageNet           |

---

# Methodology

The notebook follows these steps:

1. Import PyTorch and supporting libraries.
2. Select CPU or CUDA GPU automatically.
3. Load the CIFAR-10 dataset.
4. Apply image preprocessing and ImageNet normalization.
5. Select 5,000 training samples.
6. Split the training samples into training and validation sets.
7. Create DataLoaders.
8. Load pretrained AlexNet, VGG16, ResNet50, and EfficientNetB0 models.
9. Freeze the pretrained model parameters.
10. Replace the final classification layer with a 10-class classifier.
11. Train each model for 3 epochs.
12. Save the best validation weights during training.
13. Evaluate each model on the test dataset.
14. Record accuracy, loss, and training time.
15. Compare the four models.
16. Generate accuracy and training-time graphs.
17. Generate training/validation accuracy curves.
18. Generate training/validation loss curves.
19. Generate confusion matrices.
20. Generate classification reports.

---

# Training

The model uses:

```python
criterion = nn.CrossEntropyLoss()
```

and the Adam optimizer:

```python
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
```

Only the newly added classifier parameters are trained because the pretrained parameters are frozen.

---

# Evaluation Metrics

Each model is evaluated using:

* **Test Accuracy**
* **Test Loss**
* **Training Time**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

The notebook also tracks:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

---

# Model Comparison

The notebook creates a final comparison table containing:

```text
Model
Accuracy (%)
Test Loss
Training Time (sec)
```

Example structure:

| Model          | Accuracy (%) | Test Loss | Training Time (sec) |
| -------------- | -----------: | --------: | ------------------: |
| AlexNet        |            — |         — |                   — |
| VGG16          |            — |         — |                   — |
| ResNet50       |            — |         — |                   — |
| EfficientNetB0 |            — |         — |                   — |

The actual values are generated when the notebook is executed.

---

# Visualizations

## 1. Model Accuracy Comparison

A bar chart compares the test accuracy of:

* AlexNet
* VGG16
* ResNet50
* EfficientNetB0

## 2. Training Time Comparison

A bar chart compares the training time required by each model.

## 3. Training vs Validation Accuracy

Separate accuracy curves are generated for every model.

```text
Training Accuracy
        vs
Validation Accuracy
```

## 4. Training vs Validation Loss

Separate loss curves are generated for every model.

```text
Training Loss
       vs
Validation Loss
```

## 5. Confusion Matrix

A confusion matrix is generated for each model to show the relationship between actual and predicted CIFAR-10 classes.

## 6. Classification Report

A classification report is generated for every model containing:

* Precision
* Recall
* F1-score
* Support

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# Installation

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

For GPU training, install the appropriate PyTorch version for your CUDA environment.

---

# How to Run

### 1. Clone or download the repository

### 2. Open Jupyter Notebook

```bash
jupyter notebook
```

### 3. Open the notebook

```text
Assignment_7_CIFAR_10.ipynb
```

### 4. Run all cells

The CIFAR-10 dataset will be automatically downloaded by Torchvision:

```python
torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
```

---

# Project Structure

```text
Assignment-07/
│
├── Assignment_7_CIFAR_10.ipynb
├── README.md
│
└── data/
    └── CIFAR-10 dataset
```

The `data` folder is created automatically when the notebook downloads the CIFAR-10 dataset.

---

# Hardware Support

The notebook automatically checks whether a CUDA-compatible GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Therefore, the models can run on:

* NVIDIA GPU with CUDA
* CPU

GPU execution is recommended because four pretrained CNN models are trained sequentially.

---

# Best and Fastest Model

After all four models are evaluated, the notebook automatically identifies:

### Best Performing Model

The model with the highest test accuracy.

### Fastest Model

The model with the lowest training time.

The notebook prints both results at the end of execution.

---

# Expected Output

After execution, the notebook produces:

```text
✓ Dataset loaded
✓ Training/validation/test datasets created
✓ AlexNet trained and evaluated
✓ VGG16 trained and evaluated
✓ ResNet50 trained and evaluated
✓ EfficientNetB0 trained and evaluated
✓ Accuracy comparison
✓ Training-time comparison
✓ Accuracy curves
✓ Loss curves
✓ Confusion matrices
✓ Classification reports
✓ Best performing model
✓ Fastest model
```

---

# Conclusion

This assignment demonstrates the practical application of **transfer learning for image classification** using pretrained CNN architectures.

AlexNet, VGG16, ResNet50, and EfficientNetB0 are adapted from ImageNet classification to the **10-class CIFAR-10 problem** by replacing their original classification layers.

The models are compared using accuracy, loss, training time, confusion matrices, and classification reports. This provides a practical understanding of how different pretrained architectures behave when applied to the same image classification task.
