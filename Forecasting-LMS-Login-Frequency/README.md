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
The dataset consists of **77 days of login frequency data** collected from a Learning Management System (LMS) during the **2023 Fall semester**. The dataset includes login activity records for **500 users**.

- **Source:** LMS , SIS
- **Time Period:** 2023 Fall Semester (77 days)
- **Number of Users:** 832
- **Data Type:** Login frequency per user
  
### **🔍 Model Validation & Hyperparameter Tuning**
To ensure model robustness, **cross-validation and hyperparameter tuning** were applied exclusively to Transformer, Reformer, Informer, Autoformer.

#### **1️⃣ Cross-Validation Method**
For time-series forecasting, we applied **Rolling Window Cross-Validation** instead of K-Fold Cross-Validation. The model is trained on past data and evaluated on a **7-day prediction horizon**, ensuring that future values are never used in training.

#### **2️⃣ Hyperparameter Tuning**
Hyperparameter tuning was performed **separately for the LMS and SIS datasets** to optimize model performance.  
A comprehensive grid search was conducted to fine-tune key hyperparameters, including `batch_size`, `sequence length (seq_len)`, `model dimension (d_model)`, `number of attention heads (n_heads)`, `encoder layers (e_layers)`, and `decoder layers (d_layers)`.

##### 📌 Best Hyperparameter Configurations (LMS)
The best configurations obtained from tuning on the **LMS dataset**:

| Model   | Batch Size | Sequence Length | d_model | n_heads | e_layers | d_layers | Validation Loss (LMS) |
|---------|------------|----------------|---------|---------|---------|---------|----------------------|
| Transformer | 32         | 14             | 256     | 16      | 2       | 2       | **0.745** |
| Reformer | 32        | 21             | 1024     | 16       | 2       | 2       | **0.734** |
| Informer | 8        | 14             | 128     | 16       | 2       | 2       | **0.73** |
| Autoformer | 8        | 21             | 256     | 16       | 2       | 1       | **0.746** |

##### 📌 Best Hyperparameter Configurations (SIS)
The best configurations obtained from tuning on the **SIS dataset**:

| Model   | Batch Size | Sequence Length | d_model | n_heads | e_layers | d_layers | Validation Loss (SIS) |
|---------|------------|----------------|---------|---------|---------|---------|----------------------|
| Transformer | 8         | 14             | 128     | 16      | 2       | 1       | **0.59** |
| Reformer | 8        | 21             | 256     | 16       | 6       | 2       | **0.58** |
| Informer | 8        | 14             | 128     | 16       | 2       | 2       | **0.585** |
| Autoformer | 32        | 14             | 128     | 8       | 2       | 1       | **0.594** |


Detailed tuning logs and configurations are available in:  
📂 **[`Results/Hyperparameter_Tuning/LMS`](Results/Hyperparameter_Tuning/LMS/)**  
📂 **[`Results/Hyperparameter_Tuning/SIS`](Results/Hyperparameter_Tuning/SIS/)**  

- **[`LMS/val_losses_test_Transformer.txt`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Transformer.txt)** → LMS tuning results for Transformer
- **[`LMS/val_losses_test_Reformer.tx`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Reformer.txt)** → LMS tuning results for Reformer
- **[`LMS/val_losses_test_Informer.tx`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Informer.txt)** → LMS tuning results for Informer
- **[`LMS/val_losses_test_Autoformer.tx`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Autoformer.txt)** → LMS tuning results for Autoformer



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

## 🔧 Open-Source Code Usage
This project incorporates open-source code from the following repositories:

- **[Autoformer](https://github.com/thuml/Autoformer)** - Licensed under **MIT License**.  
- **[LLMTIME](https://github.com/ngruver/llmtime)** - Licensed under **MIT License**.  
- **[Time-LLM](https://github.com/KimMeen/Time-LLM)** -Licensed under **Apache License 2.0**.  

We acknowledge the authors of these repositories for their contributions.
