## 3 Dataset

**Source and license.** We use the *Churn for Bank Customers* dataset (`Churn_Modelling.csv`), published on Kaggle under a CC0 Public Domain dedication [7]. It contains 10,000 customer records of a retail bank in three countries (France 5,014; Germany 2,509; Spain 2,477) and 14 columns. The file has no missing values and no duplicate rows. The same dataset is used by [1], which allows a direct comparison with prior work.

**Features and target.** The target `Exited` equals 1 if the customer left the bank and 0 if the customer stayed. After removing the three identifiers (`RowNumber`, `CustomerId`, `Surname`), ten predictors remain (Table 1).

**Table 1.** Predictors and their treatment.

| Feature | Type | Treatment |
|---|---|---|
| CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary | numeric | standardised (zero mean, unit variance) |
| HasCrCard, IsActiveMember | binary | kept as 0/1 |
| Gender | categorical (2) | encoded 0/1 |
| Geography | categorical (3) | one-hot encoded |

This gives 12 model inputs.

**Class distribution.** The classes are imbalanced: 2,037 customers churned (20.37 %) and 7,963 stayed (79.63 %). Because of this, accuracy alone is misleading (always predicting "stay" gives 79.63 %), so we report precision, recall and F1-score of the churn class and ROC-AUC, and no resampling is applied so that the class ratio of the held-out test set stays realistic.

**Preprocessing and split.** We use one stratified 70/15/15 split of the 10,000 records into training, validation and test sets with random seed 42, shared by all models (Table 2). The scaler and encoders are fitted on the training set only and then applied to the validation and test sets, so no information from held-out customers enters preprocessing. The validation set is used only for hyperparameter selection; the test set is used once for the final results. No augmentation is applied.

**Table 2.** Stratified split (seed 42).

| Set | Customers | Churners | Churn rate |
|---|---|---|---|
| Training | 7,000 | 1,426 | 20.37 % |
| Validation | 1,500 | 306 | 20.40 % |
| Test | 1,500 | 305 | 20.33 % |

**Ethical considerations.** `Surname` and `CustomerId` are removed before modelling and kept only to align predictions across models. Gender and country are legitimate predictors in this data but are protected or sensitive attributes in real banking, and churn rates differ markedly between groups (Germany 32.4 % vs. France 16.2 % and Spain 16.7 %; women 25.1 % vs. men 16.5 %). A deployed contact policy based on such scores could treat groups unequally, so we treat our results as an evaluation of methods, not as a deployable system. The origin of the data is not documented in detail, and some secondary sources describe it as a fictitious bank; we therefore make no claim about generalisation to a real institution.
