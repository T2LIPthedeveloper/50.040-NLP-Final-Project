# 50.040 Natural Language Processing - Final Project - Hybrid CNN-RNN + Attention Sentiment Analysis

This project implements a sentiment analysis model using a **Hybrid CNN-RNN** architecture with an **Attention Mechanism** for improved performance. The model is trained on the IMDb movie reviews dataset and leverages pre-trained GloVe embeddings for semantic understanding.

## Repository Location

The dataset repository for this project is hosted on GitHub and can be accessed via the following link:

[GitHub Repository](https://github.com/T2LIPthedeveloper/50.040-NLP-Final-Project)

This repository contains all the code, dataset files, and results necessary to replicate the project and analyze the performance of various models.

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required Python Libraries (Install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/T2LIPthedeveloper/50.040-NLP-Final-Project
   cd 50.040-NLP-Final-Project
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Python environment and run the following:
   ```bash
   python sentencesenseis_script.py
   ```

---

## Models and Results

The results for the sentiment analysis task using various models are summarized below:

### Hybrid-CNN-RNN with Attention
- **Accuracy**: 0.9472
- **Precision**: 0.9464
- **Recall**: 0.9480
- **F1 Score**: 0.9472

### TextCNN
- **Accuracy**: 0.8476
- **Precision**: 0.8132
- **Recall**: 0.9026
- **F1 Score**: 0.8556

### BiRNN
- **Accuracy**: 0.8320
- **Precision**: 0.9116
- **Recall**: 0.7352
- **F1 Score**: 0.8140

### Performance Metrics

For detailed performance metrics, refer to the result files:

- [Results CSV](results.csv)
- [Prediction Results (Hybrid-CNN-RNN)](prediction_results_hybrid.csv)
- [Prediction Results (TextCNN)](prediction_results_textcnn.csv)
- [Prediction Results (BiRNN)](prediction_results_birnn.csv)

---

## Contributors

This project was developed as part of the **50.040 Natural Language Processing** course at **SUTD**. The group members are:

- **Ansh Oswal (1006265)**
- **Atul Parida (1006184)**
- **Elvern Neylman Tanny (1006203)**

---

## Acknowledgements

We would like to thank the following:

- **Stanford AI Group** for providing the IMDb dataset.
- **SUTD** and Professor XX for their guidance and resources for the 50.040 NLP course.
