# 📄 [Comparison of Transformer and LLM Performance and Forecasting LMS Login Frequency]
**Authors:** Seonmi Lee<sup>1, 2</sup>, Yoonsuh Jung<sup>1</sup>  
**Affiliations:**  
¹ Korea University, Department of Statistics  
² Korea University, New Energy Industry Convergence and Open Sharing System  
**Journal:** KDISS 2025

## 📌 Abstract
contents

## 🔍 Models Used
We evaluate the performance of multiple machine learning models for login frequency forecasting.
The following models were used in this study:

- **Transformer-Based Time Series Forecasting Models:**
  - Transformer
  - Reformer
  - Informer
  - Autoformer
    
- **LLM-Based Time Series Forecasting Models:**
  - PromptCast
  - LLMTIME
  - Time-LLM

## ✅ Empirical data study

### **📂 Dataset**

### **🔍 Model Validation & Hyperparameter Tuning**
To ensure model robustness, **cross-validation and hyperparameter tuning** were applied exclusively to Transformer, Reformer, Informer, Autoformer.****

#### **1️⃣ Cross-Validation Method**

#### **2️⃣ Hyperparameter Tuning**

#### **3️⃣ Model Architecture**
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
To ensure the reproducibility of our results, the following computing environment was used:

- **Operating System:** 
- **CPU:**  
- **GPU:** 
- **RAM:** 
- **Storage:** 

## 📌 Open-Source Code Usage
This project incorporates open-source code from the following repositories:

- **[Autoformer](https://github.com/thuml/Autoformer)** - Licensed under **MIT License**.  
- **[LLMTIME](https://github.com/ngruver/llmtime)** - Licensed under **MIT License**.  
- **[Time-LLM](https://github.com/KimMeen/Time-LLM)** -Licensed under **Apache License 2.0**.  

We acknowledge the authors of these repositories for their contributions.
