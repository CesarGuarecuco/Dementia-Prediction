### **Project Description (GitHub README)**

# **Predicting Dementia in Adults Using Machine Learning and Neural Networks**

This project addresses the challenge of predicting dementia in adults using a dataset rich in clinical and health-related features. Various Machine Learning approaches and a neural network built with PyTorch are explored to analyze and optimize the classification performance, particularly considering the complexity of an imbalanced minority class.

---

## **Project Contents**
1. **Exploratory Data Analysis (EDA):**
   - Descriptive and visual analysis of the dataset variables.
   - Handling missing values (NaN) and analyzing distributions.
   - Key findings:
     - **Imbalanced classes:** Most patients do not have dementia, impacting model performance.
     - **Key variables:** `EF`, `PS`, `Global`, and SVD scores require scaling and are crucial predictors.
     - **Special encodings:** Variables like `smoking` need specific handling in models.

2. **Clustering:**
   - Using the elbow method to determine the optimal number of clusters (3 and 4).
   - Comparative analysis:
     - **3 Clusters:** Offer a simpler overview of patterns.
     - **4 Clusters:** Capture more detailed subgroups.

3. **Machine Learning Predictive Models:**
   - Training models such as:
     - **Random Forest**
     - **AdaBoost**
     - **Extra Trees**
     - **SVC**
     - **XGBoost**
   - Preprocessing:
     - Scaling with `MinMaxScaler`.
     - Balancing classes using SMOTE to mitigate imbalance.
   - Evaluation:
     - Accuracy, F1-Score, Precision, Recall, and confusion matrix analysis.

4. **Neural Network with PyTorch:**
   - Design of a neural network with:
     - **Dense layers** and ReLU activation functions.
     - **Regularization:** Dropout to prevent overfitting.
     - **Optimization:** Adam optimizer with automated learning rate reduction (`ReduceLROnPlateau`).
   - Early stopping implementation to prevent unnecessary training.
   - Final evaluation of the model.

5. **Model Comparison:**
   - **Key results:**
     - **Extra Trees:** Best overall accuracy (94.02%).
     - **SVC:** Best balance between precision and recall (88.03% accuracy and 0.50 recall for the minority class).
     - **AdaBoost:** Highest recall for the minority class (0.88) at the cost of precision and overall accuracy.

---

## **Dataset**
The dataset includes clinical, cognitive, and MRI variables from studies conducted on adults. Key fields:
- **Demographics:** Age, gender, years of education.
- **Clinical:** Diabetes, hypertension, smoking status.
- **Cognitive:** Executive Function (EF), Processing Speed (PS), Global cognitive score.
- **MRI:** Fazekas scores, lacune count, and SVD scores.

---

## **Main Scripts**
- `eda_and_preprocessing.py`: Data exploration and preprocessing.
- `clustering_analysis.py`: Clustering implementation and group analysis.
- `train_ml_models.py`: Training Machine Learning models.
- `train_neural_network.py`: Neural network implementation and training with PyTorch.
- `model_evaluation.py`: Comparison and visualization of model results.

---

## **Conclusions**
- **Recommended Model:** Extra Trees achieves the best overall accuracy while maintaining balanced performance.
- **Alternative Approaches:**
  - **SVC:** Ideal for scenarios requiring a balance between classes.
  - **AdaBoost:** Excellent for maximizing the detection of the minority class.
- **Future Work:** Explore techniques like threshold adjustment, more complex ensemble models, and additional metrics such as AUC-ROC.

---

## **Requirements**
To run this project, install the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost torch torchvision
```

---

## **Execution**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dementia-prediction.git
   cd dementia-prediction
   ```
2. Run the main script to train and evaluate models:
   ```bash
   python main.py
   ```

---

## **Contribution**
Contributions are welcome! Please open an issue or submit a pull request with suggestions or improvements.

---

Let me know if there's anything else you'd like to add or customize! 😊
