# Artificial-neural-network-Sigmoid-

Customer Churn Prediction using Artificial Neural Network
This project aims to predict customer churn for a bank using an Artificial Neural Network (ANN). The model is built using Keras with a TensorFlow backend and trained on the "Churn Modelling" dataset to identify customers who are likely to leave the bank.

📝 Table of Contents
About the Project

Dataset

Methodology

Model Architecture

Getting Started

Prerequisites

Installation

Usage

Results

📖 About the Project
Customer churn is a significant challenge for businesses. Predicting which customers are at high risk of churning allows the business to take proactive steps to retain them. This project implements a deep learning model to tackle this problem.

The core of this project is the ANN_Sigmoid.py script, which performs the following steps:

Loads and preprocesses the customer data from Churn_Modelling.csv.

Builds a Sequential Artificial Neural Network.

Trains the network on the preprocessed data.

Evaluates the model's performance on a test set.

📊 Dataset
The project uses the Churn_Modelling.csv dataset. It contains 10,000 records of bank customers with 14 attributes.

Key Attributes:

CreditScore: The customer's credit score.

Geography: The country where the customer resides (France, Spain, Germany).

Gender: The customer's gender.

Age: The customer's age.

Tenure: The number of years the customer has been with the bank.

Balance: The customer's account balance.

NumOfProducts: The number of bank products the customer uses.

HasCrCard: Whether the customer has a credit card (1 for yes, 0 for no).

IsActiveMember: Whether the customer is an active member (1 for yes, 0 for no).

EstimatedSalary: The customer's estimated salary.

Exited: The target variable, indicating whether the customer has churned (1 for yes, 0 for no).

🛠️ Methodology
Data Preprocessing:

Irrelevant columns (RowNumber, CustomerId, Surname) are dropped.

Categorical features (Geography, Gender) are converted into numerical format using one-hot encoding.

The dataset is split into features (X) and the target variable (y).

The data is divided into a training set (70%) and a testing set (30%).

Feature scaling is applied to the training and testing sets using StandardScaler to normalize the data.

Model Training:

A Sequential ANN is constructed.

The model is compiled with the adam optimizer, binary_crossentropy loss function (suitable for binary classification), and accuracy as the evaluation metric.

The model is trained on the training data for 10 epochs with a batch size of 10.

🧠 Model Architecture
The neural network consists of the following layers:

Layer Type

Units/Neurons

Activation Function

Details

Input/Dense

11

ReLU

Input dimension matches number of features

Hidden/Dense

6

ReLU

First hidden layer

Hidden/Dense

4

ReLU

Second hidden layer

Output/Dense

1

Sigmoid

Outputs a probability between 0 and 1

The model is initialized with a uniform distribution for the kernel weights.

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites

You need to have Python and pip installed. The required Python libraries are listed below.

numpy

pandas

tensorflow

keras

scikit-learn

Installation

Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>

Install the required packages:

pip install numpy pandas tensorflow keras scikit-learn

🏃 Usage
To run the model training and prediction script, navigate to the project directory and execute the following command:

python ANN_Sigmoid.py

The script will load the Churn_Modelling.csv file, preprocess the data, train the model, and print a classification report to the console.

📈 Results
After training, the model's performance is evaluated on the test set. The script will output the predicted probabilities for the test set and a detailed classification report, which includes:

Precision: The ratio of correctly predicted positive observations to the total predicted positives.

Recall (Sensitivity): The ratio of correctly predicted positive observations to all observations in the actual class.

F1-Score: The weighted average of Precision and Recall.

The final output will look similar to this:

              precision    recall  f1-score   support

           0       0.88      0.96      0.92      2396
           1       0.75      0.49      0.59       604

    accuracy                           0.86      3000
   macro avg       0.81      0.72      0.75      3000
weighted avg       0.85      0.86      0.85      3000

(Note: The exact values may vary slightly due to the random initialization of weights.)

