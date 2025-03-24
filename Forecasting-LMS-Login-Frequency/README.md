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

- **Transformer-Based Time Series Models:**
  - Transformer
  - Reformer
  - Informer
  - Autoformer
    
- **LLM-Based Time Series Models:**
  - PromptCast
  - LLMTIME
  - Time-LLM

## Real Data Analysis

### **📂 Dataset Description**
This study utilizes **login frequency data** collected from a LMS and a SIS. 

- **Data Source:** LMS & SIS logs
- **Time Period:** 2023 Fall Semester (77 days)
- **Number of Users:** 832
- **Feature Description:** Login frequency per user, Timestamps
  
##### 📌 **Data Availability:**
Due to privacy regulations, this dataset **cannot be publicly shared**.

## **🔧 Experimental Setup**
To ensure the reproducibility of our results, the following computing environment is used:

- **GPU:** 2-NVIDIA-RTX-A6000
- **CPU:** 16-core
- **RAM:** 64GB
  
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

### **📌 Results**
- Comparison of Model Performance and Computing Time in Predicting Logins on the LMS Platform
  | Models             | Performance indicator | Mean of 7 days | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 | Computing time (min) |
  |--------------------|-----|------ |-------|-------|-------|-------|-------|-------|-------|-----------|
  | Transformer        | MSE | 1.474 | 1.036 | 0.849 | 1.243 | 2.143 | 1.840 | 1.893 | 1.316 |  1375.333 |
  |                    | NAE | 0.670 | 0.558 | 0.457 | 0.596 | 0.917 | 0.823 | 0.698 | 0.643 |           |
  | Reformer           | MSE | 1.489 | 1.026 | 0.833 | 1.280 | 2.111 | 1.898 | 1.911 | 1.365 | 982.283   |
  |                    | NAE | 0.647 | 0.548 | 0.429 | 0.590 | 0.868 | 0.790 | 0.678 | 0.625 |           |
  | Informer           | MSE | 1.478 | 1.055 | 0.864 | 1.224 | 2.218 | 1.834 | 1.871 | 1.276 | 893.867   |
  |                    | NAE | 0.692 | 0.546 | 0.448 | 0.587 | 0.903 | 0.846 | 0.794 | 0.719 |           |
  | Autoformer         | MSE | 1.475 | 1.008 | 0.858 | 1.150 | 2.213 | 1.894 | 1.911 | 1.291 | 951.500   |
  |                    | NAE | 0.660 | 0.511 | 0.462 | 0.593 | 0.881 | 0.820 | 0.712 | 0.644 |           |
  | PromptCast GPT-3.5 | MSE | 2.070 | 1.917 | 1.91 | 1.791 | 2.653 | 2.143 | 2.347 | 1.732 | 19.917     |
  |                    | NAE | 0.997 | 1.004 | 1.056 | 0.981 | 1.078 | 0.980 | 0.960 | 0.917 |           |
  | LLMTIME GPT-3.5    | MSE | 1.944 | 1.738 | 1.504 | 1.703 | 2.736 | 2.279 | 2.167 | 1.480 | 21.283    |
  |                    | NAE | 0.766 | 0.690 | 0.641 | 0.782 | 0.957 | 0.861 | 0.744 | 0.686 |           |
  | LLMTIME GPT-4o     | MSE | 2.368 | 2.160 | 1.657 | 2.478 | 3.031 | 2.636 | 2.556 | 2.060 | 35.133    |
  |                    | NAE | 0.884 | 0.898 | 0.703 | 0.875 | 0.998 | 0.929 | 0.862 | 0.921 |           |
  | Time-LLM BERT      | MSE | 1.855 | 1.221 | 0.945 | 1.480 | 3.031 | 2.419 | 2.234 | 1.651 | 15.850    |
  |                    | NAE | 0.689 | 0.538 | 0.409 | 0.643 | 0.988 | 0.845 | 0.710 | 0.69 |            |
  | Time-LLM GPT-2     | MSE | 1.855 | 1.203 | 0.930 | 1.507 | 3.014 | 2.454 | 2.236 | 1.643 | 15.817    |
  |                    | NAE | 0.688 | 0.53 | 0.406 | 0.649 | 0.986 | 0.844 | 0.714 | 0.684 |            |
  | Time-LLM LLAMA     | MSE | 1.849 | 1.201 | 0.954 | 1.505 | 2.989 | 2.425 | 2.209 | 1.660 | 88.267    |
  |                    | NAE | 0.686 | 0.53 | 0.411 | 0.651 | 0.977 | 0.844 | 0.704 | 0.686 |            |
  
- Comparison of Model Performance and Computing Time in Predicting Logins on the SIS Platform
  | Models             | Performance indicator | Mean of 7 days | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 | Computing time (min) |
  |--------------------|-----|----------------|-------|-------|-------|-------|-------|-------|-------|----------------------|
  | Transformer        | MSE | 
  |                    | NAE |
  | Reformer           | MSE | 
  |                    | NAE | 
  | Informer           | MSE | 
  |                    | NAE | 
  | Autoformer         | MSE | 
  |                    | NAE |
  | PromptCast GPT-3.5 | MSE |
  |                    | NAE |
  | LLMTIME GPT-3.5    | MSE |
  |                    | NAE |
  | LLMTIME GPT-4o     | MSE |
  |                    | NAE |
  | Time-LLM BERT      | MSE |
  |                    | NAE |
  | Time-LLM GPT-2     | MSE |
  |                    | NAE |
  | Time-LLM LLAMA     | MSE |
  |                    | NAE |
  
## Conclusion
contents

## Open-Source Code Usage
This project incorporates open-source code from the following repositories:

- **[Autoformer](https://github.com/thuml/Autoformer)** - Licensed under **MIT License**.  
- **[LLMTIME](https://github.com/ngruver/llmtime)** - Licensed under **MIT License**.  
- **[Time-LLM](https://github.com/KimMeen/Time-LLM)** -Licensed under **Apache License 2.0**.  

We acknowledge the authors of these repositories for their contributions.
