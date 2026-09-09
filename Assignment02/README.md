# Assignment 02 – Wine Classification using MLP

## Aim

To implement a **Multi-Layer Perceptron (MLP) Neural Network** for classifying wine samples into different wine classes using the Wine dataset available in Scikit-learn.

## Dataset

The project uses the built-in **Wine Dataset** from `sklearn.datasets`.

* **Dataset:** Wine Dataset
* **Source:** Scikit-learn
* **Task:** Multi-class Classification
* **Target:** Classification of wine samples into different classes
* **Train-Test Split:** 80% Training and 20% Testing
* **Random State:** 42

The dataset is loaded using `load_wine()` and separated into features (`X`) and target labels (`y`).

## Model Used

A **Multi-Layer Perceptron Classifier (MLPClassifier)** is used for classification.

### Model Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
```

### Architecture

```text
Input Features
      ↓
Hidden Layer 1 – 10 Neurons
      ↓
Hidden Layer 2 – 10 Neurons
      ↓
Output Layer – Wine Classes
```

## Methodology

The following steps are performed:

1. Load the Wine dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Standardize the input features using `StandardScaler`.
5. Create the MLP neural network.
6. Train the model using the training data.
7. Predict wine classes for the test data.
8. Calculate model accuracy.
9. Generate the confusion matrix.
10. Generate the classification report.
11. Visualize the training loss curve.
12. Compare actual and predicted classes.
13. Visualize classification metrics using a heatmap.

## Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1-Score**
* **Training Loss Curve**

The notebook also displays an **Actual vs Predicted** comparison for the test samples.

## Visualizations

The following visualizations are generated:

### 1. Confusion Matrix Heatmap

Shows the number of correctly and incorrectly classified samples for each wine class.

### 2. Training Loss Curve

Shows how the model's loss changes during the training iterations.

### 3. Actual vs Predicted

Compares the actual wine classes with the classes predicted by the MLP model.

### 4. Classification Report Heatmap

Displays precision, recall, and F1-score for each wine class.

## Technologies Used

* Python
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

## Libraries Required

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
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
Assignment02_DL.ipynb
```

5. Run all cells sequentially.

## Project Structure

```text
Assignment-02/
│
├── Assignment02_DL.ipynb
└── README.md
```

## Result

The MLP neural network is trained on the Wine dataset and evaluated on the test dataset. The notebook provides accuracy, confusion matrix, classification report, training loss visualization, and actual-vs-predicted results to analyze the classification performance.

## Author

**Abhishek Kadam**

---

### Conclusion

The assignment demonstrates how a **Multi-Layer Perceptron neural network** can be used for multi-class wine classification. Feature standardization and evaluation through multiple metrics and visualizations help in understanding the performance of the trained neural network.

