# Machine Learning Model Implementation for Spam Detection

## Overview

This project implements a spam detection system using a Naive Bayes classifier. The model is trained on a dataset of SMS messages, classifying them as either "ham" (not spam) or "spam". The implementation utilizes various libraries for data manipulation, visualization, and machine learning.

## Features

- Loads and preprocesses a dataset of SMS messages.
- Converts categorical labels into binary format.
- Splits the dataset into training and testing sets.
- Vectorizes text data using Count Vectorization.
- Trains a Naive Bayes classifier on the training data.
- Evaluates the model's performance using accuracy, classification report, and confusion matrix.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/vaibhavxom/MACHINE_LEARNING_MODEL_IMPLEMENTATION.git
    cd MACHINE_LEARNING_MODEL_IMPLEMENTATION
    ```
#
2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    ```
#
3. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
#
4. Install dependencies:
    ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
    ```
#
5. Ensure you have the dataset file named SMSSpamCollection in the same directory as the script. The dataset should be in tab-separated format with two columns: label and message.

#

## Usage
Run the spam detection script:

```bash
python spam_detection.py
```
The script will display the first few rows of the dataset, the accuracy of the model, a classification report, and a confusion matrix visualizing the model's performance.

## Code Explanation
 - Imports: The script imports necessary libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
 - Data Loading: The dataset is loaded from a local file using pandas.
 - Data Preprocessing: The labels are converted from categorical to binary format (0 for ham, 1 for spam).
 - Train-Test Split: The dataset is split into training and testing sets using train_test_split.
 - Text Vectorization: The text messages are vectorized using CountVectorizer to convert them into a format suitable for model training.
 - Model Training: A Naive Bayes classifier (MultinomialNB) is trained on the vectorized training data.
 - Prediction and Evaluation: The model makes predictions on the test set, and its performance is evaluated using accuracy, a classification report, and a confusion matrix.

## Example Output
Upon running the script, you will see output similar to the following:  
![image](https://github.com/user-attachments/assets/2b07ed75-9be3-427a-930c-fbc54dab4b46)


