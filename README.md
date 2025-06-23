# Email-Spam-Detection-with-NLP-and-ML

## Project Overview

This project demonstrates the implementation of a **spam email classifier** using supervised machine learning techniques in Python. It uses the **SMS Spam Collection Dataset** to build a model that can accurately distinguish between spam (unwanted) and ham (legitimate) text messages. The goal is to apply Natural Language Processing (NLP) techniques to extract features from text, train a predictive model, and evaluate its performance using industry-standard metrics.

The project is implemented in a clean, step-by-step manner and includes data preprocessing, feature extraction, model training, testing, and evaluation. The final output includes an accuracy score, a classification report, and a confusion matrix — both numerically and visually plotted using Seaborn and Matplotlib.

This type of model can be extended or modified for email spam detection, SMS filtering, or content moderation systems.

## Dataset Information

- **Source**: [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Format**: Two columns — `label` (ham or spam) and `message` (text)
- **Preprocessing**: Labels are mapped to binary format (`ham` → 0, `spam` → 1)

## Technologies Used

- **Python 3.x**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **scikit-learn** – ML model training and evaluation
- **Matplotlib & Seaborn** – data visualization

## Workflow

### Step-by-step Process

1. **Import Libraries**  
   Load required Python libraries and modules.

2. **Load Dataset**  
   The dataset is fetched directly from GitHub and loaded into a pandas DataFrame.

3. **Data Preprocessing**  
   Convert text labels into binary format (0 for ham, 1 for spam).

4. **Feature Extraction**  
   Transform the text data into numeric features using `CountVectorizer`.

5. **Split Dataset**  
   Use `train_test_split` to divide the dataset into training and testing sets.

6. **Model Training**  
   A `MultinomialNB` (Naive Bayes) classifier is trained on the extracted features.

7. **Prediction and Evaluation**  
   - Generate predictions on the test set
   - Calculate accuracy
   - Print classification report (precision, recall, F1-score)
   - Display confusion matrix (text and heatmap)


## Model Performance

Typical results:
- **Accuracy**: ~97–98%
- **High Precision**: Effectively detects spam with minimal false positives
- **High Recall**: Catches most spam messages

---

## 📊 Sample Output

