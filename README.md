# RNN-income-predict

# 🧠 Train an RNN Tabular Classifier from Your CSV
This Streamlit app lets you upload your own tabular CSV file, preprocess it, and train a binary classifier using a PyTorch-based RNN-style neural network.

## 🔧 Features
- 📁 Upload any CSV file with:
  - A required **target column** named `income`
  - One or more **categorical columns**
  - One or more **continuous numeric columns**
- ⚙️ Interactive controls for:
  - Selecting categorical features
  - Adjusting training parameters (batch size, learning rate, epochs)
- 🧪 Model training with real-time loss & accuracy plots
- 🧠 Predict new values via dynamic input form
- 💾 Download trained PyTorch model (`.pth` file)

## 📦 Requirements
Install required packages with:
```bash
pip install streamlit torch pandas scikit-learn seaborn matplotlib
