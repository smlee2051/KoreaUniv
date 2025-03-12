# 📄 [Comparison of Transformer and LLM Performance and Forecasting LMS Login Frequency]
**Authors:** Seonmi Lee<sup>1, 2</sup>, Yoonsuh Jung<sup>2</sup> 
**Affiliations:**  
¹ Korea University, Department of Statistics, Seoul, South Korea  
² Korea University, New Energy Industry Convergence and Open Sharing System
**Journal:** KDISS 2025

## 📌 Abstract
contents

## 📂 Dataset

## 🔍 Methodology
### **1️⃣ Models Used**
- **Transformer-Based Time Series Forecasting Models:** Transformer, Reformer, Informer, Autoformer
- **LLM-Based Time Series Forecasting Models:** PromptCast, LLMTIME, Time-LLM

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
