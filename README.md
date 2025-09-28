Breast Cancer Detection using Machine Learning

This project involves building and evaluating several machine learning models to predict whether a breast cancer tumor is malignant or benign based on medical data. The primary goal is to apply various classification techniques and analyze their performance using key evaluation metrics.

This project was developed by following a guided learning path from a playlist on Unstop.

Topics & Concepts Covered

This project implements a full machine learning workflow, covering the following key topics:

- **Logistic Regression:** Understanding and implementing a foundational classification algorithm.
- **Model Implementation:** Building, training, and testing the models.
- **Error Analysis:** Differentiating between False Positives and False Negatives and understanding their critical importance in medical diagnoses.
- **Performance Metrics:** Using a **Confusion Matrix** to evaluate model performance beyond simple accuracy.
- **Accuracy Paradox:** Understanding the limitations of the accuracy metric, especially with imbalanced datasets.
- **CAP Curve & Analysis:** Using a Cumulative Accuracy Profile (CAP) curve to visualize and assess the predictive power of a model.
- **Model Selection:** Comparing different classification algorithms to select the best-performing one for the given task.

Models Implemented

Three different classification models were trained and compared:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**

Results

After training and evaluation, the **Support Vector Machine (SVM) model** was identified as the most effective, achieving an **accuracy of 97%** on the test set.

The performance of all three models was visualized using confusion matrices.

Technologies Used
- Python
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- Scikit-learn (for machine learning models and metrics)
- Matplotlib & Seaborn (for data visualization)
- OpenPyXL (for reading Excel files)

How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Anushka394/Breast-Cancer-Detection.git](https://github.com/Anushka394/Breast-Cancer-Detection.git)
