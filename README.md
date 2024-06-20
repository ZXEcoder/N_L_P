Quora Question Pair Analysis using LSTMs
Overview

Welcome to the Quora Question Pair Analysis project! This project leverages Long Short-Term Memory (LSTM) networks to analyze pairs of questions from Quora and determine whether they are semantically similar. By addressing this problem, we aim to improve question deduplication, enhance user experience, and optimize content management on the Quora platform.
Table of Contents

    Introduction
    Dataset
    Installation
    Project Structure
    Model Architecture
    Training the Model
    Evaluation
    Results
    Future Work
    Contributing
    License

Introduction

Duplicate questions on Quora can lead to redundancy and inefficiency in knowledge sharing. To combat this, we use LSTM networks, a type of recurrent neural network (RNN) well-suited for sequence prediction tasks, to identify question pairs that are similar. This project not only showcases the power of LSTMs in natural language processing (NLP) but also provides a robust solution for real-world applications.
Dataset

We use the Quora Question Pairs dataset from Kaggle, which contains over 400,000 question pairs with binary labels indicating whether the questions are duplicates.
Features

    id: Unique identifier for the question pair
    qid1, qid2: Unique identifiers for each question
    question1, question2: The actual questions
    is_duplicate: Label (1 if questions are duplicates, 0 otherwise)

Installation

To get started, clone this repository and install the necessary dependencies:

bash

git clone https://github.com/yourusername/quora-question-pair-analysis.git
cd quora-question-pair-analysis
pip install -r requirements.txt

Project Structure

plaintext

quora-question-pair-analysis/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── lstm_model.h5
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── Evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt

Model Architecture

Our LSTM model consists of the following layers:

    Embedding Layer: Converts words into dense vectors of fixed size.
    Bidirectional LSTM Layer: Captures dependencies in both forward and backward directions.
    Dense Layers: Adds a fully connected neural network to process the LSTM outputs.
    Output Layer: A single neuron with sigmoid activation for binary classification.

 (Illustrative purposes, replace with actual model diagram)
Training the Model

To train the model, use the following command:

bash

python src/train.py --epochs 10 --batch_size 64 --embedding_dim 100 --lstm_units 128

This script will preprocess the data, build the model, and train it on the Quora Question Pairs dataset.
Evaluation

Evaluate the trained model's performance using the provided evaluation script:

bash

python src/evaluate.py --model_path models/lstm_model.h5 --data_path data/test.csv

The evaluation metrics include accuracy, precision, recall, and F1-score.
Results

Our LSTM model achieves impressive performance on the Quora Question Pairs dataset, with high accuracy and robustness in detecting duplicate questions. Detailed results can be found in the Evaluation.ipynb notebook.
Future Work

Future improvements could include:

    Experimenting with different neural network architectures (e.g., GRUs, Transformers).
    Utilizing pre-trained embeddings (e.g., GloVe, BERT) for better contextual understanding.
    Enhancing the model with additional features (e.g., question length, common words).

Contributing

We welcome contributions to this project! Feel free to submit issues and pull requests. Please ensure your contributions adhere to our code of conduct.
License

This project is licensed under the MIT License. See the LICENSE file for more details.

Thank you for checking out our Quora Question Pair Analysis project! We hope you find it insightful and valuable. Happy coding! 🚀
