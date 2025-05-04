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

Phishing remains one of the most common cyberattacks. In this project, we leverage deep learning to classify emails as **phishing** or **legitimate** using:
- Text preprocessing (cleaning, tokenisation)
- Word embeddings
- LSTM (Long Short-Term Memory) neural network

The goal is to accurately detect phishing emails through supervised learning.

---

## Dataset

The dataset contains email messages labelled as **phishing** or **legitimate**.  
Due to its large size (~60 MB), the dataset is **not included in the repository**.

> 📥 **[Download CSV dataset here](#)**  
> Place the CSV file inside a folder named `data/` at the root of the project.

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
