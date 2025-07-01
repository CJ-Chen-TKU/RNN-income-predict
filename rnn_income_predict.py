 # py code beginning
import streamlit as st
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

st.title("📄 Train an RNN Tabular Classifier from Your CSV")

st.markdown("""
Upload a CSV file with the following layout:

- ✅ One target column: **`income`**
- 🧩 Multiple **categorical columns** (e.g., strings or categories)
- 🔢 Multiple **continuous numeric columns**

This app will automatically encode, scale, and train an RNN-based classifier on your tabular data.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    df.replace('?', pd.NA, inplace=True)

    st.subheader("📈 Correlation Heatmap (Numeric Columns Only)")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to show correlation.")

    all_columns = df.columns.tolist()
    
    target_column = "income"
    st.markdown(f"🎯 Target column is fixed to **{target_column}**")
    
    
    
#    target_column = st.selectbox("🎯 Choose your target column", all_columns)

    cat_cols = st.multiselect("🧩 Categorical columns", all_columns, default=[
        col for col in all_columns if df[col].dtype == 'object' and col != target_column
    ])
    cont_cols = [col for col in all_columns if col not in cat_cols + [target_column]]

    st.subheader("⚙️ Training Parameters")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=64)
    learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=0.001, format="%.6f")
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)

    if st.button("✅ Train Model"):
        st.write("開始訓練")
        df = df.dropna()
        st.write(f"資料筆數: {len(df)}")

        # Label Encoding
        label_encoders = {}
        for col in cat_cols + [target_column]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Standardize
        scaler = StandardScaler()
        df[cont_cols] = scaler.fit_transform(df[cont_cols])

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        class TabularDataset(Dataset):
            def __init__(self, df):
                self.cats = df[cat_cols].values.astype('int64')
                self.conts = df[cont_cols].values.astype('float32')
                self.y = df[target_column].values.astype('float32')

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return (
                    torch.tensor(self.cats[idx]),
                    torch.tensor(self.conts[idx]),
                    torch.tensor(self.y[idx])
                )

        train_loader = DataLoader(TabularDataset(train_df), batch_size=64, shuffle=True, drop_last=True)
        test_loader = DataLoader(TabularDataset(test_df), batch_size=64, shuffle=False, drop_last=True)

        class TabularModel(nn.Module):
            def __init__(self, emb_dims, n_cont, hidden_layers=[100, 50], dropout=0.5):
                super().__init__()
                self.embeds = nn.ModuleList([
                    nn.Embedding(categories, size)
                    for categories, size in emb_dims
                ])
                self.emb_drop = nn.Dropout(dropout)
                self.bn_cont = nn.BatchNorm1d(n_cont)

                input_size = sum([e[1] for e in emb_dims]) + n_cont
                layers = []
                for i in range(len(hidden_layers)):
                    in_size = input_size if i == 0 else hidden_layers[i-1]
                    layers += [
                        nn.Linear(in_size, hidden_layers[i]),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_layers[i]),
                        nn.Dropout(dropout)
                    ]
                layers.append(nn.Linear(hidden_layers[-1], 1))
                self.layers = nn.Sequential(*layers)

            def forward(self, x_cat, x_cont):
                x = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
                x = self.emb_drop(x)
                x_cont = self.bn_cont(x_cont)
                x = torch.cat([x, x_cont], dim=1)
                return torch.sigmoid(self.layers(x)).squeeze()

        emb_dims = [(df[col].nunique(), min(50, (df[col].nunique() + 1) // 2)) for col in cat_cols]
        model = TabularModel(emb_dims, len(cont_cols))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        progress = st.progress(0)
        status = st.empty()
        train_losses, test_losses, test_accuracies = [], [], []

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for x_cat, x_cont, y in train_loader:
                optimizer.zero_grad()
                out = model(x_cat, x_cont)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))

            model.eval()
            val_loss = 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for x_cat, x_cont, y in test_loader:
                    if x_cont.size(0) == 1:
                    # skip batch size = 1 to avoid BatchNorm error
                      continue


                    out = model(x_cat, x_cont)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    y_true.extend(y.tolist())
                    y_pred.extend((out > 0.5).float().tolist())

            val_loss /= len(test_loader)
            acc = accuracy_score(y_true, y_pred)
            test_losses.append(val_loss)
            test_accuracies.append(acc)

            progress.progress((epoch + 1) / epochs)
            status.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.2%}")

        st.success("🎉 Model training complete!")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Val Loss")
        ax.plot(test_accuracies, label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / Accuracy")
        ax.legend()
        st.pyplot(fig)

        st.session_state.model = model
        st.session_state.label_encoders = label_encoders
        st.session_state.scaler = scaler
        st.session_state.cat_cols = cat_cols
        st.session_state.cont_cols = cont_cols

        torch.save(model.state_dict(), "rnn_income_predict_model.pth")
        st.download_button("⬇️ Download Model", open("rnn_income_predict_model.pth", "rb"), "rnn_income_predict_model.pth")

    if "model" in st.session_state:
        st.header("🔍 Predict New Row")

        def build_input_ui(df, cat_cols, cont_cols, label_encoders):
            user_input = {}
            for col in cat_cols:
                options = list(label_encoders[col].classes_)
                user_input[col] = st.selectbox(f"{col}", options)

            for col in cont_cols:
                val = float(df[col].dropna().mean())
                user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), val)
            return user_input

        user_input = build_input_ui(df, st.session_state.cat_cols, st.session_state.cont_cols, st.session_state.label_encoders)

        if st.button("🚀 Predict Income Class"):
            for col in st.session_state.cat_cols:
                le = st.session_state.label_encoders[col]
                user_input[col] = le.transform([user_input[col]])[0]

            cont_array = [[user_input[col] for col in st.session_state.cont_cols]]
            scaled = st.session_state.scaler.transform(cont_array)[0]
            for i, col in enumerate(st.session_state.cont_cols):
                user_input[col] = scaled[i]

            x_cat = torch.tensor([[user_input[col] for col in st.session_state.cat_cols]], dtype=torch.long)
            x_cont = torch.tensor([[user_input[col] for col in st.session_state.cont_cols]], dtype=torch.float)

            model = st.session_state.model
            model.eval()
            with torch.no_grad():
                pred = model(x_cat, x_cont).item()
                st.success(f"Predicted probability: **{pred:.2%}**")
                st.info("Prediction: " + (">50K ✅" if pred > 0.5 else "≤50K ❌"))
else:
    st.info("⬆️ Upload and train the model to unlock prediction.")

