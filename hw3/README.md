# Spoken-SQuAD Question Answering using BERT

This repository contains the implementation of question-answering models for the Spoken-SQuAD dataset using BERT (Bidirectional Encoder Representations from Transformers). The goal is to build a question-answering system that can comprehend spoken inputs even in challenging conditions involving audio noise.

## Dataset

The models are trained and evaluated on the Spoken-SQuAD dataset derived from the original SQuAD dataset. It consists of questions in text form and answers extracted from spoken documents. The dataset is divided into the following files:

- `spoken_train-v1.1.json`: Training set containing 37,111 question-answer pairs.
- `spoken_test-v1.1.json`: Standard test set containing 5,351 question-answer pairs.
- `spoken_test-v1.1_WER44.json`: Test set with added noise resulting in a Word Error Rate (WER) of 44.22%.
- `spoken_test-v1.1_WER54.json`: Test set with higher noise levels, resulting in a WER of 54.82%.

## Models

Three different BERT-based models are implemented in this repository:

1. **Simple BERT** (`bert_qa_simple.ipynb`): A basic BERT model for question answering.
2. **BERT with Doc Stride** (`bert_qa_medium_docstride.ipynb`): BERT model with the addition of doc stride to handle longer contexts.
3. **Advanced Pretrained BERT** (`bert_qa_strong.ipynb`): BERT model initialized with a pre-trained checkpoint (`deepset/bert-base-uncased-squad2`) for improved performance.

## Training

Each model is trained using the Spoken-SQuAD training set. The training process involves the following steps:

1. Loading and preprocessing the dataset.
2. Tokenizing the input using a BERT tokenizer.
3. Training the BERT model with specified hyperparameters.
4. Plotting the training loss and accuracy curves.

## Evaluation

The trained models are evaluated on three test sets: the standard test set (22.73% WER), a noisy test set with 44.22% WER, and another noisy test set with 54.82% WER. The evaluation metrics used are:

- **F1 Score**: Measures the average overlap between the predicted and true answers.
- **Exact Match (EM)**: Calculates the percentage of predictions that exactly match the true answers.

The evaluation results for each model are as follows:

| Model                    | Test Set | F1 Score | Exact Match |
|--------------------------|----------|----------|-------------|
| Simple BERT              | No Noise | 66.69%   | 48.36%      |
|                          | WER 44   | 25.87%   | 4.77%       |
|                          | WER 54   | 20.91%   | 2.88%       |
| BERT with Doc Stride     | No Noise | 69.39%   | 48.42%      |
|                          | WER 44   | 38.04%   | 26.53%      |
|                          | WER 54   | 26.25%   | 18.31%      |
| Advanced Pretrained BERT | No Noise | 73.03%   | 51.09%      |
|                          | WER 44   | 41.26%   | 29.11%      |
|                          | WER 54   | 31.11%   | 22.99%      |

## Conclusion

The advanced pre-trained BERT model (`deepset/bert-base-uncased-squad2`) performs best among the three models, demonstrating improved robustness to ASR errors. Including doc stride in the medium BERT model enhances its ability to handle longer contexts than the simple BERT model.

For more details on each model's implementation, please refer to the Jupyter Notebook files and the submitted report (`Choudavarpu_Prateek_Homework_3_Report.pdf`).
