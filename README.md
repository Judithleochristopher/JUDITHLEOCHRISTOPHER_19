# Customer Churn Prediction — End-to-End ML Pipeline
**Dataset Source:** [Kaggle — Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  
**Notebook:** `Judithleochristopher.ipynb`  
**Final Dataset:** `customer_churn_final.csv`
## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Source](#2-dataset-source)
3. [Dataset Creation — From Kaggle to Churn Dataset](#3-dataset-creation--from-kaggle-to-churn-dataset)
4. [Final Dataset Description](#4-final-dataset-description)
5. [Data Loading & Inspection](#5-data-loading--inspection)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Data Preprocessing & Feature Encoding](#7-data-preprocessing--feature-encoding)
8. [Train-Test Split](#8-train-test-split)
9. [Baseline Model Training](#9-baseline-model-training)
10. [Hyperparameter Tuning](#10-hyperparameter-tuning)
11. [Rule-Based System](#11-rule-based-system)
12. [Model Evaluation — Detailed Results](#12-model-evaluation--detailed-results)
13. [Model Comparison & Why Logistic Regression Wins](#13-model-comparison--why-logistic-regression-wins)
14. [Prediction on New Customer](#14-prediction-on-new-customer)
## 1. Project Overview

This project builds a complete **Customer Churn Prediction System** using a real-world marketing dataset from Kaggle. Since the original dataset did not include a churn label and the required columns, a **synthetic churn target** was engineered using behavioral logic based on customer recency, spending, income, and age and the other columns were engineered using logics.
Three ML models are trained, tuned, and evaluated:

Logistic Regression
Random Forest Classifier
XGBoost Classifier

A Rule-Based System is also implemented as a non-ML baseline to compare against the learned models.
The project covers the full ML lifecycle: data sourcing → feature engineering → churn label creation → EDA → preprocessing → training → hyperparameter tuning → evaluation → prediction.

## 2. Dataset Source
Kaggle Dataset: Customer Personality Analysis

## 3. Dataset Creation — From Kaggle to Churn Dataset

**Feature 1: Age**
df['Age'] = current_date.year - df['Year_Birth']
**Feature 2: Purchases (Total Spending)**
df['Purchases'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                   df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
**Feature 3: Membership (Customer Tier)**

<img width="573" height="243" alt="image" src="https://github.com/user-attachments/assets/3d9ffb59-5dfe-4515-88fa-95e659a54429" />

**Feature 4: Churn (Synthetic Target Label)**
churn_prob = (
    0.45 * Recency_norm +           # Highest weight: hasn't purchased recently
    0.25 * (1 - Purchases_norm) +   # Low total spending
    0.20 * (1 - Income_norm) +      # Low income customers churn more
    0.10 * (Age < 25)               # Younger customers churn more
)
# Small noise added for realism
churn_prob += np.random.normal(0, 0.02)
df['Churn'] = (np.random.rand(len(df)) < churn_prob).astype(int)

**Class Distribution**

<img width="524" height="145" alt="image" src="https://github.com/user-attachments/assets/9595acd6-7d41-4470-89a0-def76285c836" />

The dataset is well-balanced — no SMOTE or resampling is needed.

**Membership Distribution**

<img width="440" height="251" alt="image" src="https://github.com/user-attachments/assets/6f7ab31b-3eb8-4074-b9f9-9f9f25a90911" />

## 4.Save Final Dataset
df_final[['Age', 'Income', 'Purchases', 'Membership', 'Recency', 'Churn']].to_csv(
    'customer_churn_final.csv', index=False)

## 5. Data Loading & Inspection
df = pd.read_csv("customer_churn_final.csv")
**Data Types**

<img width="236" height="183" alt="image" src="https://github.com/user-attachments/assets/699c19e8-088b-401b-bde7-1cbf5f16bc4f" />

## 6. Exploratory Data Analysis
 **Histogram Analysis**
All numerical features were plotted with a KDE curve, mean line (red dashed), and median line (green dashed):
**Age**: Roughly bell-shaped, centered around 57. Most customers are middle-aged to senior (40–75). A few extreme outliers above 120.
**Income**: Right-skewed — most customers earn in the ₹35K–₹70K range. Long right tail due to outliers. Mean (52,238) > Median (51,382) — characteristic of right skew.
**Purchases**: Heavily right-skewed — many customers have low total spending (<200). A smaller group of high-value customers spend 1,500–2,500. Mean=605.8 vs Median=396 shows strong skew.
**Recency**: Roughly uniform across 0–99 days. Mean ≈ Median ≈ 49 — symmetric and well-distributed.
**Churn**: Near-equal split between 0 and 1, confirming the synthetic label is balanced.

**Box Plot Analysis (Outlier Detection)**
Box plots confirmed:
**Age**: Outliers at 126, 127, 133 (far beyond upper whisker)
**Income**: Multiple extreme outliers above ₹200,000
**Purchases**: Long upper whisker; several customers spending 2,000+
**Recency**: Clean, no extreme outliers

**Correlation Heatmap**
Recency ↔ Churn: Positive — the longer since last purchase, the more likely the customer churned
Purchases ↔ Churn: Negative — higher spenders are less likely to churn
Income ↔ Purchases: Positive — higher income customers spend more overall
Age ↔ Churn: Weak — age alone is not a strong predictor in this dataset

<img width="830" height="443" alt="image" src="https://github.com/user-attachments/assets/2b9b5cdd-ea62-400c-8cbc-86ba74251e77" />

**Membership Count Plot**

<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/5590a48b-eb97-4f01-9b38-6e6e3de3fded" />

## 7. Data Preprocessing & Feature Encoding
**Label Encoding (Exploratory)**

<img width="305" height="133" alt="image" src="https://github.com/user-attachments/assets/cd3513fa-473f-4866-967b-ca0898db4998" />

**One-Hot Encoding (Used for Models)**
df_encoded = pd.get_dummies(df_final, columns=['Membership'])
**Feature Scaling**
StandardScaler normalizes each feature to mean=0, std=1. This is critical for Logistic Regression which is sensitive to feature scale differences.
**Final feature matrix:**
8 features: Age, Income, Purchases, Recency, Membership_Loyal, Membership_New, Membership_Regular, Membership_VIP

## 8. Train-Test Split
1.80/20 split — industry standard for datasets of this size
2.stratify=y — preserves the 48.5% churn rate in both train and test sets
3.random_state=42 — fixed seed ensures reproducibility

## 9. Baseline Model Training
Three ML models are trained, tuned, and evaluated:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

<img width="596" height="647" alt="image" src="https://github.com/user-attachments/assets/bf5b6957-671a-4e26-b35a-e6772dc86d5e" />

## 10. Hyperparameter Tuning
**Logistic Regression**

<img width="500" height="53" alt="image" src="https://github.com/user-attachments/assets/2b0921ab-af3a-4bac-b18a-59fe6919a06c" />
**Random Forest**

<img width="963" height="54" alt="image" src="https://github.com/user-attachments/assets/6f27391c-c68c-48d2-b58d-d993fdca4fe9" />
**XG Boost**

<img width="1158" height="42" alt="image" src="https://github.com/user-attachments/assets/f3ab949b-b8f9-4436-a444-f0e22fe6e260" />


## 11. Rule-Based System
Implemented both a rule-based system and machine learning models. The rule-based system provides interpretability, while ML models offer data-driven generalization. Comparing both helped evaluate model effectiveness.
This performs well for clearly defined behavioral patterns but struggles with ambiguous cases, where machine learning models provide better generalization.

## 12. Model Evaluation — Detailed Results

<img width="547" height="647" alt="image" src="https://github.com/user-attachments/assets/d6d9e960-126a-4b1b-b117-187ae07681f3" />

**LR**

<img width="802" height="517" alt="image" src="https://github.com/user-attachments/assets/cb777a22-d455-4ca4-a9de-88509e44133c" />
**RF**

<img width="813" height="527" alt="image" src="https://github.com/user-attachments/assets/d75ddf02-282e-4d26-aa1c-cfc8aada9dcc" />

**XGB**

<img width="823" height="524" alt="image" src="https://github.com/user-attachments/assets/45f65fe7-2c0a-4bcb-a0cc-56cb2180ab78" />

## 13. Model Comparison & Why Logistic Regression is better
1. The Data Has Linear Structure — By Design
The churn target was synthetically created from a weighted linear formula:
python
churn_prob = 0.45*Recency + 0.25*(1-Purchases) + 0.20*(1-Income) + 0.10*(Age<25)
This is a linear combination of normalized features. Logistic Regression is specifically designed to find exactly this kind of linear decision boundary. It learns the correct underlying pattern directly. Random Forest and XGBoost search for complex non-linear interactions and conditional splits — but those don't exist here.

2. Complex Models Overfit Noise
RF and XGBoost are powerful because they capture non-linear interactions. When no such interactions exist, they end up fitting noise in the training set.
 Evidence: RF's AUC drops to 0.5751 (barely above random), while LR's stays at 0.6664.

4. Best Hyperparameters Confirm Data Simplicity
LR best C=0.01 → heavy regularization, very simple boundary
RF best max_depth=5 → shallow trees, not deep complexity
XGB best max_depth=3 → very shallow, slow learning rate (0.05)
All three models converged on their simplest, most constrained configurations. This collectively confirms the data does not reward complexity.

5. AUC-ROC Gap Is Significant
ModelAUC-ROCGap vs Random (0.5)Logistic Regression0.6664+0.1664XGBoost0.6398+0.1398Random Forest0.5751+0.0751
LR's AUC advantage over RF (+0.09) is substantial. AUC measures rank-ordering ability across all thresholds — it's the most reliable single metric for binary classifiers on balanced datasets.

6. Better Confusion Matrix for Business
From the confusion matrices:
ModelChurners Caught (TP)False Alarms (FP)Logistic Regression12071 XGBoost 11478 Random Forest11890
LR catches the most churners (120) while generating the fewest false alarms (71). In a real business, false alarms waste retention budget; missed churners mean lost customers. LR achieves the best trade-off.

7. Explainability Advantage
Logistic Regression coefficients are directly interpretable:
Positive coefficient for Recency → as days since last purchase increase, churn probability rises
Negative coefficient for Purchases → as spending increases, churn probability falls
This transparency is valuable in internship presentations and for business stakeholders who need to understand not just who will churn but why.

Root Cause Summary
The churn label was generated from a linear formula with Gaussian noise. With only 8 features (5 real features + 4 binary membership flags after encoding), there is limited information for complex models to work with. XGBoost and RF attempt to model interaction terms and non-linear splits that simply don't exist in this data — they fit noise instead of signal, reducing test-set generalization.
*Model complexity should match data complexity. Since, Linear data - linear model wins.*

## 14. Prediction on New Customer

<img width="741" height="501" alt="image" src="https://github.com/user-attachments/assets/793c1470-5a91-4a04-aaf0-3b2a0af225d3" />

## PROJECT STRUCTURE

├── Judithleochristopher.ipynb   # Main notebook
├── customer_churn_final.csv               # Processed dataset (generated by Cell 2)
├── marketing_campaign_cleaned.xlsx        # Kaggle source data (download separately)
└── README.md                              # This file
