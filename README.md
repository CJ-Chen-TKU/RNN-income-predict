# Tabular Binary Classifier with PyTorch and Streamli

🧠 Key Learning Objectives
Data Preprocessing
Label encode categorical features
Standardize continuous features
Custom Dataset & DataLoader
Define TabularDataset for PyTorch
Model Architecture
Use embeddings for categorical inputs
Combine with continuous features
Build a feedforward neural net (MLP)
Training Loop
Binary classification with BCELoss
Epoch loop and optimizer
Single Row Inference
Preprocess new row
Run through model and return prediction

# This Streamlit app lets you upload your own tabular CSV file, preprocess it automatically, and train a binary classifier using a PyTorch-based feedforward neural network (MLP).

Key features:
Handles mixed data types:
Encodes categorical features using label encoding and embeddings
Normalizes continuous numeric features
Uses embeddings to represent categorical variables effectively
Trains a multi-layer feedforward neural network with dropout and batch normalization
Supports real-time single-row predictions with automatic preprocessing
Runs efficiently on CPU or GPU
This app is ideal for users who want to quickly build a deep learning model on tabular data without extensive manual feature engineering or coding.

# Usage Instructions
1. Upload Your CSV File
Click the Upload button and select your tabular CSV file.
The CSV should contain:
One binary target column (e.g., income: >50K or <=50K)
Multiple categorical columns (strings or categories)
Multiple continuous numeric columns

2. Preprocessing
The app will automatically:
Encode categorical columns using label encoding
Normalize continuous numeric columns with standard scaling
Convert the target column into binary labels (0 and 1)

3. Train the Model
Click Train to start training the feedforward neural network.
Training progress and loss per epoch will be displayed.
The model uses embeddings for categorical variables and batch normalization/dropout in hidden layers.

4. Make Predictions
After training, you can enter new data in the provided input form.
The app will preprocess the input, run inference, and display the predicted probability and class.

5. Download / Save Model (Optional)
Optionally, save the trained model for later use (if implemented).
