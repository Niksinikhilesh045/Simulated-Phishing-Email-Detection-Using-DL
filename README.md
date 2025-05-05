# Simulated-Phishing-Email-Detection-Using-DL
This project presents an approach to detecting phishing emails using both traditional machine learning techniques and deep learning architectures. By analysing the textual content of emails, the model aims to classify them as either phishing or legitimate, helping enhance email security against social engineering attacks.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-(EDA))
- [Visualizations](#visualizations)
- [Limitations](#limitations)
- [Installation](#installation)
- [Contributing](#contributing)
- [Conclusion](#conclusion)

---

## Overview

Phishing attacks are one of the most common and dangerous types of cyberattacks, wherein attackers attempt to deceive individuals or organisations into disclosing sensitive information by mimicking legitimate sources. As phishing tactics evolve, identifying malicious emails becomes more complex.

This project addresses this challenge by using deep learning techniques to classify emails as phishing or legitimate based on their textual content. The model utilises:

-Loads and preprocesses a phishing email dataset.
-Applies both traditional ML and deep learning models.
-Evaluates and compares the performance of multiple classifiers.
-Uses LSTM, GRU, and Bidirectional RNNS for deep text-based analysis.
-Text Preprocessing: Cleaning, tokenising, and padding the email text to ensure consistency for deep learning processing.
-Word Embeddings: Converting words into dense vector representations that capture semantic relationships between them.
-LSTM (Long Short-Term Memory) Network: A type of recurrent neural network (RNN) used to capture long-term dependencies in the email content and effectively classify emails based on patterns in the sequence of words.

By training the model on a labelled dataset of phishing and legitimate emails, this project aims to build an effective tool for automatic phishing email detection.

---

## Dataset

The dataset used in this project contains a collection of email messages that are labelled as either phishing or legitimate. It provides a rich set of features based on the content of the emails, which allows the model to learn the distinguishing characteristics of phishing attempts.

### Dataset Details:

Format: CSV file with two columns: email_text (the body of the email) and label (the classification, where 1 represents phishing and 0 represents legitimate).
Content: The dataset includes various types of phishing emails, such as those that try to steal personal information, credentials, or money, as well as legitimate emails from recognised sources.

---

## Preprocessing

Key preprocessing steps included:

-Removing null values and duplicates
-Text normalisation (lowercasing, removing URLs and punctuation)
-Tokenisation and integer encoding
-Word vectorisation using Tokeniser and pad_sequences
-Train-test splitting

Additionally, WordClouds were used to visualise common words in phishing vs. legitimate emails.

---

## Machine Learning Models

The following traditional classifiers were implemented and evaluated:

1. **Naive Bayes**
2. **Logistic Regression**
3. **Stochastic Gradient Descent (SGD)**
4. **XGBoost**
5. **Decision Tree**
6. **Random Forest**
7. **Multi-Layer Perceptron (MLP)**

Each model was assessed for accuracy, precision, recall, and F1-score.

---

## Deep Learning Models

Deep learning models were built using TensorFlow/Keras and included:

1. **Simple RNN**: A basic recurrent layer to model word sequences.
2. **LSTM**: Captures long-term dependencies in email sequences.
3. **GRU**: Efficient alternative to LSTM.
4. **Bidirectional RNN**: Reads sequences in both forward and backwards directions.

All models used an Embedding Layer followed by recurrent layers and a Dense output layer with sigmoid activation for binary classification.

---

## Exploratory Data Analysis (EDA)

A detailed EDA was conducted to:

-Explore the distribution of phishing vs. legitimate samples.
-Visualise frequent terms using WordClouds.
-Analyse performance metrics of different classifiers.

---

## Visualizations

-WordClouds for phishing and legitimate emails
-Confusion matrices for model performance
-Accuracy/loss plots for deep learning training
-Comparative bar charts for ML and DL model performance

---

## Limitations

-Results depend heavily on the quality and size of the dataset.
-Contextual clues in phishing emails may require more advanced transformers (e.g., BERT).
-Real-world deployment would need email header analysis and URL inspection.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Niksinikhilesh045/Simulated-phishing-Email-detection-Using-DL.git
cd phishing-email-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook Phishing-Email-Detection.ipynb
```

---

## Contributing

Feel free to fork the repository and submit pull requests. Contributions are welcome in the areas of:
-Model improvement
-Dataset expansion
-Integration with real-time email systems
-Visualisation enhancements

---

## Conclusion

This project illustrates how deep learning, especially LSTM-based models, can significantly improve phishing email detection. Combining NLP preprocessing with advanced neural architectures helps build resilient models against evolving cyber threats.

---
