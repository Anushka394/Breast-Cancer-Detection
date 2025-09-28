````markdown
# Breast Cancer Detection using Machine Learning

This project involves building and evaluating several machine learning models to predict whether a breast cancer tumor is malignant or benign based on medical data. The primary goal is to apply various classification techniques and analyze their performance using key evaluation metrics.

This project was developed by following a guided learning path from a **Machine Learning playlist on Unstop**.

## Topics & Concepts Covered

This project implements a full machine learning workflow, covering the following key topics:

-   **Logistic Regression:** Understanding and implementing a foundational classification algorithm.
-   **Model Implementation:** Building, training, and testing the models.
-   **Error Analysis:** Differentiating between False Positives and False Negatives and understanding their critical importance in medical diagnoses.
-   **Performance Metrics:** Using a **Confusion Matrix** to evaluate model performance beyond simple accuracy.
-   **Accuracy Paradox:** Understanding the limitations of the accuracy metric, especially with imbalanced datasets.
-   **CAP Curve & Analysis:** Using a Cumulative Accuracy Profile (CAP) curve to visualize and assess the predictive power of a model.
-   **Model Selection:** Comparing different classification algorithms to select the best-performing one for the given task.

## Models Implemented

Three different classification models were trained and compared:
1.  **Logistic Regression**
2.  **K-Nearest Neighbors (KNN)**
3.  **Support Vector Machine (SVM)**

## Results

After training and evaluation, the **Support Vector Machine (SVM) model** was identified as the most effective, achieving an **accuracy of 97%** on the test set.

The performance of all three models was visualized using confusion matrices:

![Comparison of Confusion Matrices](confusion_matrices_comparison.png)

## Technologies Used
- Python
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- Scikit-learn (for machine learning models and metrics)
- Matplotlib & Seaborn (for data visualization)
- OpenPyXL (for reading Excel files)

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn openpyxl
    ```
3.  **Place the dataset:** Make sure your Excel file named `Data.xlsx` is in the same directory as the `Implementation.py` script.

4.  **Execute the script:**
    ```bash
    python Implementation.py
    ```
5.  **Check the output:** The script will print the accuracy and confusion matrix for each model in the terminal and save the comparison graphs (`confusion_matrices_comparison.png` and `cap_curve_svm.png`) in the project folder.

````
