<h1 align="center">Comparison of Transformer and LLM Performance and <br> Forecasting LMS Login Frequency</h1>

<p align="center">
  <strong>Authors:</strong>  
  <br> Seonmi Lee<sup>1,2</sup>, Yoonsuh Jung<sup>1</sup>
  <br> <sup>1</sup> Korea University, Department of Statistics  
  <sup>2</sup> Korea University, New Energy Industry Convergence and Open Sharing System  
</p>

<p align="center">
  <strong>Published in:</strong>  
  <br> <em>name</em>  
  <br> [View Paper](https://)
</p>

## Abstract
contents

## Models Used
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

## Empirical data study

### **📂 Dataset Description**
The dataset consists of **77 days of login frequency data** collected from a Learning Management System (LMS) during the **2023 Fall semester**. The dataset includes login activity records for **832 users**.

- **Data Source:** LMS , SIS
- **Time Period:** 2023 Fall Semester (77 days)
- **Number of Users:** 832
- **Feature Description:** Login frequency per user, Timestamps

## **🔧 Experimental Setup**
To ensure the reproducibility of our results, the following computing environment is used:

- **Computing Environment:** 
- **GPU:** 2-NVIDIA-RTX-A6000
- **CPU:** 16 
- **RAM:** 64
  
### **🔍 Model Validation & Hyperparameter Tuning**
To ensure model robustness, **cross-validation and hyperparameter tuning** were applied exclusively to Transformer, Reformer, Informer, Autoformer.

#### **1️⃣ Cross-Validation Method**
For time-series forecasting, we applied **Rolling Window Cross-Validation** instead of K-Fold Cross-Validation. The model is trained on past data and evaluated on a **7-day prediction horizon**, ensuring that future values are never used in training.

#### **2️⃣ Hyperparameter Tuning**
Hyperparameter tuning is performed **separately for the LMS and SIS datasets** to optimize model performance.  
A comprehensive grid search is conducted to fine-tune key hyperparameters, including `batch_size`, `sequence length (seq_len)`, `model dimension (d_model)`, `number of attention heads (n_heads)`, `encoder layers (e_layers)`, and `decoder layers (d_layers)`.  
All other hyperparameters were set to their default values as specified in the original open-source implementation.  

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
- **[`LMS/val_losses_test_Reformer.txt`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Reformer.txt)** → LMS tuning results for Reformer
- **[`LMS/val_losses_test_Informer.txt`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Informer.txt)** → LMS tuning results for Informer
- **[`LMS/val_losses_test_Autoformer.txt`](Results/Hyperparameter_Tuning/LMS/val_losses_test_Autoformer.txt)** → LMS tuning results for Autoformer

- **[`SIS/val_losses_test_Transformer.txt`](Results/Hyperparameter_Tuning/SIS/val_losses_test_Transformer.txt)** → SIS tuning results for Transformer
- **[`SIS/val_losses_test_Reformer.txt`](Results/Hyperparameter_Tuning/SIS/val_losses_test_Reformer.txt)** → SIS tuning results for Reformer
- **[`SIS/val_losses_test_Informer.txt`](Results/Hyperparameter_Tuning/SIS/val_losses_test_Informer.txt)** → SIS tuning results for Informer
- **[`SIS/val_losses_test_Autoformer.txt`](Results/Hyperparameter_Tuning/SIS/val_losses_test_Autoformer.txt)** → SIS tuning results for Autoformer

### **📌 Results & Discussion**
- Table of Performance Comparison

## Conclusion
contents

## Open-Source Code Usage
This project incorporates open-source code from the following repositories:

- **[Autoformer](https://github.com/thuml/Autoformer)** - Licensed under **MIT License**.  
- **[LLMTIME](https://github.com/ngruver/llmtime)** - Licensed under **MIT License**.  
- **[Time-LLM](https://github.com/KimMeen/Time-LLM)** -Licensed under **Apache License 2.0**.  

We acknowledge the authors of these repositories for their contributions.
