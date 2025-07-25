# 🎗️ Breast Cancer Survival Prediction

This project aims to predict the survival status of breast cancer patients using various machine learning algorithms. Early detection and accurate prediction can significantly improve patient outcomes, enable better treatment planning, and assist healthcare professionals in clinical decision-making.

---

## Publication

This project has been published as a research paper in collaboration with professors Mr. Samad Najjar-Ghabel and Ms. Shamim Yousefi.

Paper title: A Technical Analysis and Practical Implementation of Machine Learning Algorithms for Predicting Survival in Breast Cancer Patients 
Available at: [IEEE Xplore – View Paper](https://ieeexplore.ieee.org/document/10773521)

---

## Dataset

The dataset used in this project was obtained from the SEER (Surveillance, Epidemiology, and End Results) program – November 2017 update, released by the National Cancer Institute (NCI).

- Focused on female patients diagnosed with invasive ductal and lobular carcinoma (code 8522/3) between 2006 and 2010.
- Final dataset included 4024 patients after preprocessing.

### Preprocessing Steps

- Removed patients with unknown tumor size or unexamined regional lymph nodes.
- Excluded patients with survival time less than one month.
- Converted categorical features to numerical values.
- Normalized the data using Z-score normalization.
- Split the data into:
  - 90% training
  - 10% testing

---

## Objective

The primary objective of this project is to evaluate and compare the performance of various machine learning algorithms for predicting breast cancer patient survival. Accurate predictions can support:

- Informed treatment planning
- Efficient allocation of medical resources
- Better support for patients and their families in decision-making

---

## Machine Learning Models

### Algorithms Implemented

- Naive Bayes  
- k-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM) with the following kernels:
  - Linear
  - Polynomial
  - RBF
  - Sigmoid
- Logistic Regression  
- Artificial Neural Network (ANN)  
- Ensemble Learning Methods:
  - Hard Voting
  - Soft Voting
  - Stacking Classifier

### Data Augmentation

To address class imbalance (i.e., more "alive" than "dead" samples), the following augmentation techniques were applied:

- Random Over Sampling  
- SMOTE (Synthetic Minority Over-sampling Technique)

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score

---

## Performance Comparison

| Algorithm                 | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Naive Bayes               | 0.831    | 0.514     | 0.514  | 0.514    |
| k-Nearest Neighbors (KNN) | 0.870    | 0.687     | 0.471  | 0.559    |
| Decision Tree             | 0.908    | 0.823     | 0.600  | 0.695    |
| Random Forest             | 0.915    | 0.928     | 0.557  | 0.696    |
| SVM (Linear)              | 0.890    | 0.861     | 0.442  | 0.584    |
| SVM (Polynomial)          | 0.885    | 0.852     | 0.414  | 0.557    |
| SVM (RBF)                 | 0.888    | 0.857     | 0.428  | 0.571    |
| SVM (Sigmoid)             | 0.816    | 0.464     | 0.371  | 0.412    |
| Logistic Regression       | 0.900    | 0.875     | 0.500  | 0.636    |
| Artificial Neural Network | 0.905    | 0.880     | 0.528  | 0.660    |
| Ensemble - Hard Voting    | 0.903    | 0.860     | 0.528  | 0.654    |
| Ensemble - Soft Voting    | 0.903    | 0.829     | 0.557  | 0.666    |
| Stacking Classifier       | 0.905    | 0.880     | 0.528  | 0.660    |
| Random Over Sampling      | 0.898    | 0.716     | 0.685  | 0.700    |
| SMOTE Oversampling        | 0.885    | 0.671     | 0.671  | 0.671    |

---

## Best Model

The Random Forest algorithm achieved the highest accuracy (91.5%) and precision (92.8%), making it the top-performing model in this study. However, other models such as Decision Tree, ANN, and ensemble methods like Stacking and Soft Voting also demonstrated strong performance.

---

## Conclusion

This project demonstrates the effectiveness of various machine learning algorithms, particularly ensemble learning techniques, in predicting the survival of breast cancer patients. The findings show the potential of such models to assist in medical prognosis, treatment optimization, and decision support systems in healthcare environments.

---

## Technologies Used

- Python  
- scikit-learn  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Imbalanced-learn (for SMOTE)  
- Machine Learning, Data Mining

---
