<h1 align="center">Predicting LMS Login Frequency using Transformer- and LLM-Based Time Series Models</h1>

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

## Overview
This study introduces a Transformer-based time series model and an LLM-based time series model, and applies these models to predict LMS login frequency.

## Models Used
We evaluate the performance of several machine learning models for login frequency forecasting.
The models considered in this study include the following:

- **Transformer-Based Time Series Models:** Transformer, Reformer, Informer, Autoformer
    
- **LLM-Based Time Series Models:** Promptcast, LLMTime, Time-LLM

## Real Data Analysis

### **📂 Dataset Description**
This study utilizes **login frequency data** collected from a LMS and a SIS. 

- **Data Source:** LMS & SIS logs
- **Time Period:** 2023 Fall Semester (77 days)
- **Number of Users:** 832
- **Feature Description:** Login frequency per user, Timestamps
  
##### **Data Availability:** Due to privacy regulations, this dataset **cannot be publicly shared**.

### **🔧 Experimental Setup**
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

##### Best Hyperparameter Configurations (LMS)
The best configurations obtained from tuning on the **LMS dataset**:

| Model   | Batch Size | Sequence Length | d_model | n_heads | e_layers | d_layers | Validation Loss (LMS) |
|---------|------------|----------------|---------|---------|---------|---------|----------------------|
| Transformer | 32         | 14             | 256     | 16      | 2       | 2       | **0.745** |
| Reformer | 32        | 21             | 1024     | 16       | 2       | 2       | **0.734** |
| Informer | 8        | 14             | 128     | 16       | 2       | 2       | **0.73** |
| Autoformer | 8        | 21             | 256     | 16       | 2       | 1       | **0.746** |

##### Best Hyperparameter Configurations (SIS)
The best configurations obtained from tuning on the **SIS dataset**:

| Model   | Batch Size | Sequence Length | d_model | n_heads | e_layers | d_layers | Validation Loss (SIS) |
|---------|------------|----------------|---------|---------|---------|---------|----------------------|
| Transformer | 8         | 14             | 128     | 16      | 2       | 1       | **0.59** |
| Reformer | 8        | 21             | 256     | 16       | 6       | 2       | **0.58** |
| Informer | 8        | 14             | 128     | 16       | 2       | 2       | **0.585** |
| Autoformer | 32        | 14             | 128     | 8       | 2       | 1       | **0.594** |


Detailed tuning logs and configurations are available in:  
- ➡️ **[`Hyperparameter_Tuning/LMS`](Hyperparameter_Tuning/LMS/)**
- ➡️ **[`Hyperparameter_Tuning/SIS`](Hyperparameter_Tuning/SIS/)**
  <details>
  <summary>Click to view files</summary>
    
  - **[`LMS/val_losses_test_Transformer.txt`](Hyperparameter_Tuning/LMS/val_losses_test_Transformer.txt)** → LMS tuning results for Transformer
  - **[`LMS/val_losses_test_Reformer.txt`](Hyperparameter_Tuning/LMS/val_losses_test_Reformer.txt)** → LMS tuning results for Reformer
  - **[`LMS/val_losses_test_Informer.txt`](Hyperparameter_Tuning/LMS/val_losses_test_Informer.txt)** → LMS tuning results for Informer
  - **[`LMS/val_losses_test_Autoformer.txt`](Hyperparameter_Tuning/LMS/val_losses_test_Autoformer.txt)** → LMS tuning results for Autoformer
  
  - **[`SIS/val_losses_test_Transformer.txt`](Hyperparameter_Tuning/SIS/val_losses_test_Transformer.txt)** → SIS tuning results for Transformer
  - **[`SIS/val_losses_test_Reformer.txt`](Hyperparameter_Tuning/SIS/val_losses_test_Reformer.txt)** → SIS tuning results for Reformer
  - **[`SIS/val_losses_test_Informer.txt`](Hyperparameter_Tuning/SIS/val_losses_test_Informer.txt)** → SIS tuning results for Informer
  - **[`SIS/val_losses_test_Autoformer.txt`](Hyperparameter_Tuning/SIS/val_losses_test_Autoformer.txt)** → SIS tuning results for Autoformer
    

### **📌 Results**
- Comparison of Model Performance and Computing Time in Predicting Logins on the LMS Platform
  <details>
  <summary>Click to show table</summary>

  | Models             | Metrics | Mean of<br>7 days | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 | Computing<br>time (min) |
  |--------------------|-----|------ |-------|-------|-------|-------|-------|-------|-------|-----------|
  | Transformer        | MSE | 1.474 | 1.036 | 0.849 | 1.243 | 2.143 | 1.840 | 1.893 | 1.316 |  1375.333 |
  |                    | NAE | 0.670 | 0.558 | 0.457 | 0.596 | 0.917 | 0.823 | 0.698 | 0.643 |           |
  | Reformer           | MSE | 1.489 | 1.026 | 0.833 | 1.280 | 2.111 | 1.898 | 1.911 | 1.365 | 982.283   |
  |                    | NAE | 0.647 | 0.548 | 0.429 | 0.590 | 0.868 | 0.790 | 0.678 | 0.625 |           |
  | Informer           | MSE | 1.478 | 1.055 | 0.864 | 1.224 | 2.218 | 1.834 | 1.871 | 1.276 | 893.867   |
  |                    | NAE | 0.692 | 0.546 | 0.448 | 0.587 | 0.903 | 0.846 | 0.794 | 0.719 |           |
  | Autoformer         | MSE | 1.475 | 1.008 | 0.858 | 1.150 | 2.213 | 1.894 | 1.911 | 1.291 | 951.500   |
  |                    | NAE | 0.660 | 0.511 | 0.462 | 0.593 | 0.881 | 0.820 | 0.712 | 0.644 |           |
  | Promptcast GPT-3.5 | MSE | 2.070 | 1.917 | 1.91 | 1.791 | 2.653 | 2.143 | 2.347 | 1.732 | 19.917     |
  |                    | NAE | 0.997 | 1.004 | 1.056 | 0.981 | 1.078 | 0.980 | 0.960 | 0.917 |           |
  | LLMTime GPT-3.5    | MSE | 1.944 | 1.738 | 1.504 | 1.703 | 2.736 | 2.279 | 2.167 | 1.480 | 21.283    |
  |                    | NAE | 0.766 | 0.690 | 0.641 | 0.782 | 0.957 | 0.861 | 0.744 | 0.686 |           |
  | LLMTime GPT-4o     | MSE | 2.368 | 2.160 | 1.657 | 2.478 | 3.031 | 2.636 | 2.556 | 2.060 | 35.133    |
  |                    | NAE | 0.884 | 0.898 | 0.703 | 0.875 | 0.998 | 0.929 | 0.862 | 0.921 |           |
  | Time-LLM BERT      | MSE | 1.855 | 1.221 | 0.945 | 1.480 | 3.031 | 2.419 | 2.234 | 1.651 | 15.850    |
  |                    | NAE | 0.689 | 0.538 | 0.409 | 0.643 | 0.988 | 0.845 | 0.710 | 0.69 |            |
  | Time-LLM GPT-2     | MSE | 1.855 | 1.203 | 0.930 | 1.507 | 3.014 | 2.454 | 2.236 | 1.643 | 15.817    |
  |                    | NAE | 0.688 | 0.53 | 0.406 | 0.649 | 0.986 | 0.844 | 0.714 | 0.684 |            |
  | Time-LLM LLAMA     | MSE | 1.849 | 1.201 | 0.954 | 1.505 | 2.989 | 2.425 | 2.209 | 1.660 | 88.267    |
  |                    | NAE | 0.686 | 0.53 | 0.411 | 0.651 | 0.977 | 0.844 | 0.704 | 0.686 |            |
  
- Comparison of Model Performance and Computing Time in Predicting Logins on the SIS Platform
  <details>
  <summary>Click to show table</summary>
    
  | Models             | Metrics | Mean of<br>7 days | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 | Computing<br>time (min) |
  |--------------------|-----|----------------|-------|-------|-------|-------|-------|-------|-------|----------------------|
  | Transformer        | MSE | 1.088 | 0.834 | 0.671 | 0.931 | 1.741 | 1.423 | 1.093 | 0.923 | 1107.150 |
  |                    | NAE | 0.631 | 0.496 | 0.406 | 0.570 | 0.825 | 0.804 | 0.700 | 0.614 |          |
  | Reformer           | MSE | 1.109 | 0.843 | 0.689 | 0.886 | 1.739 | 1.516 | 1.122 | 0.97  | 1043.967 |
  |                    | NAE | 0.606 | 0.487 | 0.402 | 0.543 | 0.794 | 0.761 | 0.652 | 0.599 |          |
  | Informer           | MSE | 1.088 | 0.814 | 0.657 | 0.946 | 1.757 | 1.424 | 1.076 | 0.942 | 732.533  |
  |                    | NAE | 0.629 | 0.480 | 0.399 | 0.570 | 0.839 | 0.802 | 0.710 | 0.605 |          |
  | Autoformer         | MSE | 1.111 | 0.836 | 0.648 | 0.892 | 1.787 | 1.464 | 1.159 | 0.993 | 1068.300 |
  |                    | NAE | 0.617 | 0.501 | 0.381 | 0.545 | 0.837 | 0.758 | 0.692 | 0.602 |          |
  | Promptcast GPT-3.5 | MSE | 2.052 | 1.714 | 1.740 | 1.708 | 2.483 | 2.245 | 2.572 | 1.905 | 20.150   |
  |                    | NAE | 0.994 | 0.975 | 1.024 | 0.969 | 1.057 | 1.011 | 0.972 | 0.951 |          |
  | LLMTime GPT-3.5    | MSE | 1.776 | 1.433 | 1.181 | 1.547 | 2.651 | 2.195 | 2.093 | 1.335 | 19.600   |
  |                    | NAE | 0.752 | 0.676 | 0.607 | 0.749 | 0.947 | 0.870 | 0.743 | 0.672 |          |
  | LLMTime GPT-4o     | MSE | 1.944 | 1.696 | 1.406 | 1.690 | 2.623 | 2.190 | 2.324 | 1.676 | 32.483   |
  |                    | NAE | 0.828 | 0.800 | 0.657 | 0.808 | 0.975 | 0.877 | 0.854 | 0.823 |          |
  | Time-LLM BERT      | MSE | 1.459 | 1.035 | 0.767 | 1.187 | 2.508 | 1.947 | 1.454 | 1.312 | 19.617   |
  |                    | NAE | 0.658 | 0.543 | 0.394 | 0.601 | 0.935 | 0.793 | 0.676 | 0.664 |          |
  | Time-LLM GPT-2     | MSE | 1.442 | 1.005 | 0.760 | 1.186 | 2.494 | 1.943 | 1.455 | 1.252 | 16.367   |
  |                    | NAE | 0.649 | 0.533 | 0.387 | 0.595 | 0.928 | 0.792 | 0.675 | 0.635 |          |
  | Time-LLM LLAMA     | MSE | 1.445 | 1.008 | 0.748 | 1.205 | 2.478 | 1.948 | 1.453 | 1.271 | 97.350   |
  |                    | NAE | 0.651 | 0.529 | 0.377 | 0.610 | 0.924 | 0.799 | 0.675 | 0.642 |          |

## Conclusion
1. Transformer-based models outperform LLM-based models in predictive accuracy; however, they require significantly longer computational time.
2. Within LLM-based models, Time-LLM demonstrates the highest predictive accuracy.
3. In LLMTime, GPT-3.5 outperforms GPT-4o in both predictive accuracy and computational efficiency.
4. The daily login prediction analysis reveals no consistent patterns across models or datasets.

---

## Code Structure: Model Groups and Source Repositories

This repository organizes time series models into three experimental groups.  
Each group is based on an open-source GitHub repository, and has been modified for unified experimentation and benchmarking.

- **models_group1 – Transformer-based**  
  - **Source Repository:** [Autoformer](https://github.com/thuml/Autoformer)  
  - **License:** MIT License
  - **Included Models:** Transformer, Reformer, Informer, Autoformer

- **models_group2 – LLM-based**  
  - **Source Repository:** [LLMTime](https://github.com/ngruver/LLMTime)  
  - **License:** MIT License
  - **Included Models:** Promptcast, LLMTime

- **models_group3 – LLM-based**  
  - **Source Repository:** [Time-LLM](https://github.com/KimMeen/Time-LLM)  
  - **License:** Apache License 2.0
  - **Included Models:** Time-LLM

## Modified Code Summary

We have made the following modifications to support unified experimental settings and evaluation:

- **models_group1 (based on Autoformer)**  
  - `data_provider/data_loader.py`: Added or modified `Dataset_Custom`, `Dataset_Pred` classes for custom data handling
  - `exp/exp_main.py`: Implemented or extended `Exp_Main` class to control model training and evaluation

- **models_group2 (based on LLMTime)**  
  - `models/gpt.py`: Implemented `gpt_completion_fn`, `gpt_nll_fn` for GPT-based inference and loss
  - `models/llms.py`: Refactored main LLM interface functions

- **models_group3 (based on Time-LLM)**  
  - `data_provider/data_factory.py`: Modified `data_provider` function for flexible dataset input and batch handling

All modified code follows the respective open-source licenses.
In particular, `main.py` in each model group was adjusted to support unified execution with our dataset structure and configurations.

## License Notice
This repository includes modified components from the following open-source projects:  
Autoformer (MIT License), LLMTime (MIT License), and Time-LLM (Apache 2.0).  
For license details, refer to each repository listed in the sections above.

## How to Run (via `run.sh`)

To run experiments, use the `run.sh` script by specifying a model group ID and a model name.

- **GROUP_ID**:
  - `1` → `models_group1`: Transformer-based models (Autoformer, Informer, Reformer)
  - `2` → `models_group2`: LLM-based models (LLMTime GPT-3.5, LLMTime GPT-4o, PromptCast GPT-3.5)
  - `3` → `models_group3`: **Time-LLM** models using different LLM backbones (LLAMA, GPT2, BERT)

- **MODEL_NAME**:
  - The name of the model to run.
  - For group 3, this selects the **LLM backbone** used by Time-LLM.
  - If the name contains spaces, wrap it in quotes (`"`).

### Example usage

```bash
# Run Autoformer in group 1
./run.sh 1 Autoformer

# Run LLMTime GPT-3.5 in group 2 (quotes required)
./run.sh 2 "LLMTime GPT-3.5"

# Run Time-LLM with LLAMA backbone
./run.sh 3 LLAMA
```

---

