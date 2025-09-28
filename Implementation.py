# Import necessary libraries for data handling, machine learning, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- 1. Data Loading and Preprocessing ---

# Use a try-except block to handle potential file not found errors gracefully.
try:
    # Define the path to the dataset.
    file_path = 'Data.xlsx'
    # Read the Excel file into a pandas DataFrame.
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please make sure the Excel file is in the same directory as the script.")
    exit()

# Drop the 'id' column as it is just an identifier and not a predictive feature.
if 'id' in df.columns:
    df = df.drop('id', axis=1)
# Drop the 'Unnamed: 32' column which is often an empty column created when exporting from Excel.
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Ensure the 'diagnosis' column (our target variable) is correctly formatted.
if 'diagnosis' in df.columns:
    # Remove any rows that might have a missing value in the 'diagnosis' column.
    df.dropna(subset=['diagnosis'], inplace=True)
    # Convert the column to integer type (e.g., 1 or 0) for the model.
    df['diagnosis'] = df['diagnosis'].astype(int)
else:
    print("Error: 'diagnosis' column not found in the Excel file.")
    exit()

# Separate the dataset into features (X) and the target variable (y).
# X contains all columns except 'diagnosis'.
X = df.drop('diagnosis', axis=1)
# y contains only the 'diagnosis' column.
y = df['diagnosis']

# Split the data into a training set (75%) and a testing set (25%).
# The model will learn from the training set and be evaluated on the unseen testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scale the features using StandardScaler.
# This standardizes all feature columns to have a mean of 0 and a standard deviation of 1,
# which is crucial for the performance of many algorithms like Logistic Regression and SVM.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- 2. Model Training and Evaluation ---

print("--- 1. Logistic Regression Model ---")
# Initialize the Logistic Regression model.
log_reg = LogisticRegression(random_state=0)
# Train the model using the training data.
log_reg.fit(X_train, y_train)
# Make predictions on the test data.
y_pred_log_reg = log_reg.predict(X_test)
# Create the confusion matrix to evaluate performance.
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
# Print the accuracy score, rounded to two decimal places.
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")


print("\n--- 2. K-Nearest Neighbors (KNN) Model ---")
# Initialize the KNN model with 5 neighbors.
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model.
knn.fit(X_train, y_train)
# Make predictions.
y_pred_knn = knn.predict(X_test)
# Create the confusion matrix.
cm_knn = confusion_matrix(y_test, y_pred_knn)
# Print the accuracy.
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")


print("\n--- 3. Support Vector Machine (SVM) Model ---")
# Initialize the SVM model with an RBF kernel. 'probability=True' is needed for the CAP curve.
svm = SVC(kernel='rbf', random_state=0, probability=True)
# Train the model.
svm.fit(X_train, y_train)
# Make predictions.
y_pred_svm = svm.predict(X_test)
# Create the confusion matrix.
cm_svm = confusion_matrix(y_test, y_pred_svm)
# Print the accuracy.
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")


# --- 3. Visualization ---

# Apply a professional and clean style to all subsequent plots.
sns.set_theme(style="whitegrid", palette="muted")
plt.style.use('seaborn-v0_8-talk')

# --- Plotting Confusion Matrices with better spacing ---
# Create a figure to hold the three subplots.
plt.figure(figsize=(22, 8))
# Add a main title for the entire figure.
plt.suptitle("Model Performance: Confusion Matrix Comparison", fontsize=24)
# Define a consistent font size for the numbers inside the heatmaps.
annot_kws = {"size": 16}

# Subplot 1: Logistic Regression
plt.subplot(1, 3, 1)
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws=annot_kws,
            xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('Logistic Regression (96% Acc)', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Subplot 2: K-Nearest Neighbors (KNN)
plt.subplot(1, 3, 2)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws=annot_kws,
            xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('K-Nearest Neighbors (95% Acc)', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Subplot 3: Support Vector Machine (SVM)
plt.subplot(1, 3, 3)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', cbar=False, annot_kws=annot_kws,
            xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('Support Vector Machine (97% Acc)', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Use subplots_adjust to create proper spacing for the main title, preventing it from being cut off.
plt.subplots_adjust(top=0.85, wspace=0.3)
# Save the final figure to a file with high resolution (dpi=300).
plt.savefig('confusion_matrices.png', dpi=300)
print("\nComparison of confusion matrices saved as 'confusion_matrices.png'")


# --- CAP Curve Analysis for the best performing model (SVM) ---
def plot_cap_curve(y_true, y_pred_proba, model_name):
    """
    This function calculates and plots the Cumulative Accuracy Profile (CAP) curve.
    """
    n_samples = len(y_true)
    # Sort instances by the model's predicted probabilities (descending order).
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    total_positives = sum(y_true_sorted)
    
    # Calculate the cumulative sum of positive cases found.
    x_axis = np.arange(n_samples + 1)
    y_axis = np.zeros(n_samples + 1)
    for i in range(n_samples):
        y_axis[i+1] = y_axis[i] + y_true_sorted[i]
        
    # Create the plot.
    plt.figure(figsize=(10, 8))
    # Plot the model's curve.
    plt.plot(x_axis / n_samples * 100, y_axis / total_positives * 100, label=f'{model_name} CAP Curve', color='navy', lw=2.5)
    # Plot the perfect model's curve (ideal case).
    plt.plot([0, total_positives / n_samples * 100, 100], [0, 100, 100], 'k--', label='Perfect Model', lw=2)
    # Plot the random model's curve (baseline).
    plt.plot([0, 100], [0, 100], 'r-', label='Random Model', lw=2)
    
    # Add labels and title for clarity.
    plt.xlabel('Percentage of Patients Screened', fontsize=14)
    plt.ylabel('Percentage of Malignant Cases Captured', fontsize=14)
    plt.title(f'Cumulative Accuracy Profile (CAP) Curve - {model_name}', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the final CAP curve figure.
    plt.savefig('cap_curve.png', dpi=300)
    print(f"CAP Curve for {model_name} saved as 'cap_curve.png'")

# Generate the predicted probabilities for the positive class (1) from the SVM model.
y_pred_proba_svm = svm.predict_proba(X_test)[:, 1]
# Call the function to plot the CAP curve.
plot_cap_curve(y_test, y_pred_proba_svm, "SVM")

# Display the plots on the screen (useful if running in an interactive environment).
plt.show()