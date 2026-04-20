import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Dashboard", layout="wide")

# Load model
model_path = r"C:\Users\moide\Downloads\new\model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

weights = model["weights"]
bias = model["bias"]
mean = model["mean"]
std = model["std"]

def predict(features):
    features = np.array(features)
    features = (features - mean) / std
    return np.dot(features, weights) + bias

# ---------------- HEADER ---------------- #
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🏠 House Price Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------------- MENU ---------------- #
menu = st.radio(
    "Select a Feature",
    ["Predict Price", "Batch Prediction", "Model Insights"],
    horizontal=True
)

# =========================================================
# 🔹 1. SINGLE PREDICTION
# =========================================================
if menu == "Predict Price":
    st.subheader("📊 Single Property Prediction")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("Living Area", 500, 5000, 1500)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)

    with col2:
        garage = st.slider("Garage Area", 0, 1000, 500)
        quality = st.slider("Overall Quality", 1, 10, 5)
        year = st.slider("Year Built", 1900, 2025, 2000)
        basement = st.slider("Basement Area", 0, 2000, 800)

    if st.button("Predict Price"):
        features = [area, bedrooms, bathrooms, garage, quality, year, basement]
        price = predict(features)

        st.success(f"💰 Estimated Price: ${price:,.2f}")

# =========================================================
# 🔹 2. BATCH PREDICTION
# =========================================================
elif menu == "Batch Prediction":
    st.subheader("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.write("📊 Uploaded Data", data.head())

        try:
            required_cols = [
                "area", "bedrooms", "bathrooms",
                "garage", "quality", "year", "basement"
            ]

            if not all(col in data.columns for col in required_cols):
                st.error("⚠️ CSV must contain columns: " + ", ".join(required_cols))
            else:
                X = data[required_cols].values
                std_safe = np.where(std == 0, 1, std)
                X = (X - mean) / std_safe

                preds = np.dot(X, weights) + bias
                data["Predicted Price"] = preds

                st.success("✅ Predictions completed")
                st.write(data)

                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results", csv, "predictions.csv")

        except Exception as e:
            st.error(f"Error: {e}")
# =========================================================
# 🔹 3. MODEL INSIGHTS
# =========================================================
elif menu == "Model Insights":
    st.subheader("📈 Model Insights")

    file = st.file_uploader("Upload CSV with actual prices", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        required_cols = [
            "area", "bedrooms", "bathrooms",
            "garage", "quality", "year", "basement", "price"
        ]

        if not all(col in data.columns for col in required_cols):
            st.error("⚠️ CSV must include features + 'price' column")
        else:
            X = data[required_cols[:-1]].values
            y_actual = data["price"].values

            std_safe = np.where(std == 0, 1, std)
            X = (X - mean) / std_safe

            y_pred = np.dot(X, weights) + bias

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.scatter(y_actual, y_pred)
            ax.plot(
                [y_actual.min(), y_actual.max()],
                [y_actual.min(), y_actual.max()],
                'r--'
            )

            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Model Performance")

            st.pyplot(fig, use_container_width=False)  # ✅ SAME INDENT LEVEL
