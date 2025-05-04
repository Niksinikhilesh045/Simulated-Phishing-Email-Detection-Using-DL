# Simulated-Phishing-Email-Detection-Using-DL
This project focuses on detecting phishing emails using deep learning techniques. It trains a deep learning model to analyse email content and classify it as either legitimate or phishing. The goal is to improve email security by identifying potential phishing attempts using deep learning methods.

# Phishing Email Detection using Deep Learning

This project demonstrates how deep learning can be used to detect phishing emails based on their textual content. Using techniques such as deep learning, embedding layers, and LSTM networks, it aims to identify malicious emails that mimic legitimate sources.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Contributing](#contributing)

---

## Overview

Phishing attacks are one of the most common and dangerous types of cyberattacks, wherein attackers attempt to deceive individuals or organizations into disclosing sensitive information by mimicking legitimate sources. As phishing tactics evolve, identifying malicious emails becomes more complex.

This project addresses this challenge by using deep learning techniques to classify emails as phishing or legitimate based on their textual content. The model utilises:

-Text Preprocessing: Cleaning, tokenising, and padding the email text to ensure consistency for deep learning processing.
-Word Embeddings: Converting words into dense vector representations that capture semantic relationships between them.
-LSTM (Long Short-Term Memory) Network: A type of recurrent neural network (RNN) used to capture long-term dependencies in the email content and effectively classify emails based on patterns in the sequence of words.

By training the model on a labeled dataset of phishing and legitimate emails, this project aims to build an effective tool for automatic phishing email detection.

---

## Dataset

The dataset used in this project contains a collection of email messages that are labelled as either phishing or legitimate. It provides a rich set of features based on the content of the emails, which allows the model to learn the distinguishing characteristics of phishing attempts.

# Dataset Details:

Format: CSV file with four columns: email_text (the body of the email) and label (the classification, where 1 represents phishing and 0 represents legitimate).
Content: The dataset includes various types of phishing emails, such as those that try to steal personal information, credentials, or money, as well as legitimate emails from recognised sources.

---

## Features

- Email text preprocessing: cleaning, tokenising, and padding
- Label encoding for binary classification
- Deep learning model built with Keras (LSTM)
- Accuracy and loss plotting
- Confusion matrix and classification report
- Scalable design for future dataset expansions

---

## Model Architecture

The model follows a sequential architecture:

1. **Embedding Layer** – Converts words into dense vectors.
2. **LSTM Layer** – Captures long-term dependencies in the text.
3. **Dense Output Layer** – Sigmoid activation for binary classification.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Niksinikhilesh045/Simulated-phishing-Email-detection-Using-DL.git
cd phishing-email-detection
