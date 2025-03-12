# 📄 [Comparison of Transformer and LLM Performance and Forecasting LMS Login Frequency]
**Authors:** Seonmi Lee, Yoonsuh Jung
**Conference / Journal:** KDISS 2025

## 📌 Abstract
contents

## 📂 Dataset

## 🔍 Methodology
### **1️⃣ Models Used**
- **Baseline Model:** Random Forest  
- **Deep Learning Model:** LSTM with Attention Mechanism  
- 
## 🔍 Model Validation & Hyperparameter Tuning
To ensure model robustness, **cross-validation and hyperparameter tuning** were applied.

### **1️⃣ Cross-Validation Method**

### **2️⃣ Hyperparameter Tuning**

### **3️⃣ Model Architecture**
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
```

## 🏆 Conclusion

## 🔧 Reproducibility
