# AI-Powered-Predictive-Maintenance-with-NASA-Bearing-Sensor-Data
This project focuses on detecting faults in rolling element bearings using the NASA IMS dataset. Vibration signals were preprocessed and statistical as well as frequency-domain features were extracted. Machine learning models such as Random Forest and XGBoost were trained, with XGBoost + FFT + SMOTE achieving the best performance (≈90% accuracy).
Project Overview: Bearing Fault Detection using Vibration Data
1. Introduction
This project focuses on fault diagnosis of rolling element bearings using vibration signals from the NASA IMS bearing dataset. Bearings are critical components in rotating machinery, and their failure can lead to costly downtime. Hence, early detection of faults is crucial.
The goal of this project is to build a machine learning pipeline that can classify bearing conditions into NORMAL or FAULTY using vibration signals.
2. Dataset
Source: NASA IMS Bearing Dataset (1st, 2nd, and 4th test runs).
Each file contains time-domain vibration signals recorded from 4 channels (accelerometers).
Preprocessing:
Resampled to 2048 data points per file for uniformity.
Extracted signals from 6324 (4th test), 984 (2nd test), and 2156 (1st test) files.
Final dataset:
9,464 samples
7,573 Normal (label 0)
1,891 Faulty (label 1)
3. Feature Engineering
Two types of features were extracted:
(a) Statistical Features (per channel)
RMS (Root Mean Square)
Kurtosis
Skewness
Peak-to-Peak Value
(b) Frequency-Domain Features (FFT)
Dominant Frequency
Spectral Entropy
Each file generated a 24-dimensional feature vector (6 features × 4 channels).
4. Modeling Approach
Multiple machine learning algorithms were tested:
Random Forest (RF)
Baseline model with statistical features.
Accuracy ~87%, but recall for faulty cases was lower.
Random Forest with FFT Features + SMOTE
Balanced dataset using SMOTE oversampling.
Accuracy ~88%
Better detection of faulty samples, though some misclassifications remained.
XGBoost with FFT + SMOTE (Final Model)
Tuned parameters: n_estimators=300, max_depth=6, learning_rate=0.05.
Accuracy ~90%
Precision (Faulty): 0.75
Recall (Faulty): 0.74
F1-score (Faulty): 0.74
Achieved the best trade-off between normal and faulty detection.
5. Results
Confusion Matrices demonstrate that XGBoost outperforms baseline Random Forest by reducing false negatives for faulty bearings.
Prediction Distribution plots show ~20% of files labeled as faulty across different test sets.
Time-domain visualization of combined normal vs faulty signals shows clear differences in vibration amplitude patterns.
6. Deployment Functions
Two key functions were implemented for practical usage:
predict_bearing_health(file, model) – Predicts whether a single file is NORMAL or FAULTY.
predict_folder_summary(folder, model) – Scans an entire folder, generates:
Pie chart of normal vs faulty predictions.

Combined signal plots for each class.

CSV report of predictions.
