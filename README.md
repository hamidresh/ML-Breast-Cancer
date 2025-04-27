Breast Cancer Survival Prediction
Breast cancer is one of the most common types of cancer among women, and early detection can significantly improve the chances of survival.
With advancements in technology and the use of machine learning techniques, faster and more accurate detection of the disease has become possible.
In this project, various machine learning algorithms are implemented and evaluated on a breast cancer dataset to provide accurate and reliable results for predicting the survival status of patients.

Dataset
The dataset used in this project is obtained from the November 2017 update of the SEER (Surveillance, Epidemiology, and End Results) program by the National Cancer Institute (NCI).
This dataset contains information about female patients diagnosed with invasive ductal and lobular carcinoma (code 8522/3) between 2006 and 2010.

* Data preprocessing steps:
Removal of patients with unknown tumor size or unexamined regional lymph nodes

Exclusion of patients with survival times less than one month

Finally, 4024 patients were included in the study.

Metadata of the dataset is provided in the image below.

Project Objective
The objective of this project is to evaluate and compare the performance of different machine learning algorithms for predicting the survival of breast cancer patients. These predictions can help:

Improve treatment planning,

Better management of medical resources,

Enable patients and their families to make more informed decisions.

Data Preprocessing
Removed noise from the data

Converted ordinal and nominal features into numeric values

Normalized the data using Z-Score

Split the dataset into:

90% for training

10% for testing

Implemented Machine Learning Algorithms :
1. Naive Bayes
2. k-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
* Support Vector Machine (SVM) with kernels:
5. Linear
6. Polynomial
7. RBF
8. Sigmoid
9. Logistic Regression
10. Artificial Neural Network (ANN)
11. Ensemble Learning Techniques:
12. Hard Voting
13. Soft Voting
14. Stacking Classifier
15. Data Augmentation
* To address the imbalance between the number of positive (dead) and negative (alive) samples, the following data augmentation techniques were applied:
16. Random Over Sampling
17. SMOTE (Synthetic Minority Over-sampling Technique)

Model Evaluation Metrics :
* Accuracy
* Precision
* Recall
* F1-Score

Algorithm | Accuracy | Precision | Recall | F1-Score
Naive Bayes | 0.831 | 0.514 | 0.514 | 0.514
KNN | 0.870 | 0.687 | 0.471 | 0.559
Decision Tree | 0.908 | 0.823 | 0.600 | 0.695
Random Forest | 0.915 | 0.928 | 0.557 | 0.696
SVM (linear) | 0.890 | 0.861 | 0.442 | 0.584
SVM (poly) | 0.885 | 0.852 | 0.414 | 0.557
SVM (rbf) | 0.888 | 0.857 | 0.428 | 0.571
SVM (sigmoid) | 0.816 | 0.464 | 0.371 | 0.412
Logistic Regression | 0.900 | 0.875 | 0.500 | 0.636
ANN | 0.905 | 0.880 | 0.528 | 0.660
Ensemble (Hard Voting) | 0.903 | 0.860 | 0.528 | 0.654
Ensemble (Soft Voting) | 0.903 | 0.829 | 0.557 | 0.666
Stacking Classifier | 0.905 | 0.880 | 0.528 | 0.660
Random Over Sampling | 0.898 | 0.716 | 0.685 | 0.700
SMOTE Oversampling | 0.885 | 0.671 | 0.671 | 0.671

Best Model: Decision Tree {
Accuracy: 91%
Precision: 82.3%
Recall: 60%
F1-Score: 0.695 }

Analysis of Results 
The Decision Tree model outperformed other models with an accuracy of 91%.
The Precision and Recall scores indicate that the model was able to correctly identify positive cases (dead patients), although some positive cases were misclassified.
The F1-Score of 0.7 reflects a good balance between Precision and Recall.

Conclusion
Machine learning algorithms can play a crucial role in predicting the survival of breast cancer patients, and this project demonstrates their potential for assisting in medical decision-making. These predictions could contribute to better treatment planning and healthcare management.

Publication
This project has been published in the form of a research paper, the link to which is provided below. The paper was conducted in collaboration with professors Mr. Samad Najjar-Ghabel and Ms. Shamim Yousefi.
https://ieeexplore.ieee.org/document/10773521

