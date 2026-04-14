# 🏠 House Price Prediction Dashboard

An interactive machine learning dashboard that predicts house prices based on property features.  
Built with **Python, NumPy, Streamlit, and Matplotlib**.

---

## 📌 Overview

This project demonstrates how machine learning can be applied to real estate valuation.

It includes:

- A regression model trained on housing data  
- Interactive dashboard for single and batch predictions  
- Visual insights for model performance  

---

## ⚙️ Features

- 📊 **Single Property Prediction**  
  Enter property details (area, bedrooms, bathrooms, etc.) and get price instantly  

- 📂 **Batch Prediction**  
  Upload CSV file and predict prices for multiple houses  

- 📈 **Model Insights**  
  Visualize predicted vs actual prices  

- 🎨 **Clean UI**  
  Built using Streamlit with interactive sliders and layout  

---

## 🧠 Model

- **Algorithm:** Linear Regression (from scratch using NumPy)  

### Features Used:
- GrLivArea (Living Area)  
- BedroomAbvGr  
- FullBath  
- GarageArea  
- OverallQual  
- YearBuilt  
- TotalBsmtSF  

---

## 🖼️ Dashboard Screenshots
<img width="1918" height="959" alt="image" src="https://github.com/user-attachments/assets/507a2eed-b982-4433-ab24-bf62faa3813e" />
<img width="1914" height="717" alt="image" src="https://github.com/user-attachments/assets/3772848d-2cab-4b5c-9497-135bd79281c8" />
<img width="1881" height="947" alt="image" src="https://github.com/user-attachments/assets/2b883d25-8ecf-4f27-b562-1037902202ed" />



## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy pandas streamlit matplotlib scikit-learn
