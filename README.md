# Integrating Probabilities Models and Neural Networks for Enhanced Part-of-Speech Tagging and Spellchecking

## Overview

This project focuses on improving Part-of-Speech (POS) tagging and spellchecking for the Telugu language using a combination of probabilistic models and neural networks. The aim is to provide an efficient solution to handle spelling errors and assign correct POS tags to words, enhancing the accuracy of natural language processing tasks in Telugu.

The project utilizes a BiLSTM model for POS tagging and spellchecking. It uses the Telugu Bible dataset, with data consisting of books, chapters, verses, and their respective POS tags for training and evaluation.

## Project Objectives

- **POS Tagging**: Train and test a BiLSTM model for Part-of-Speech (POS) tagging using a Telugu dataset.
- **Spellchecking**: Correct spelling errors in Telugu text using the trained models.
- **Dataset Integration**: Use a dataset structured with verses and corresponding POS tags, and integrate it directly into the model.
- **Model Evaluation**: Evaluate the models with accuracy, precision, recall, and F1-score metrics to determine their effectiveness.

## Dataset

The dataset used for this project is the **Telugu Bible dataset**. It is structured in JSON format and contains:

- Books, chapters, and verses.
- Each verse is labeled with POS tags corresponding to each word in the sentence.

### Example Dataset Structure:
```json
{
  "book": "Genesis",
  "chapter": 1,
  "verse": 1,
  "Verseid": "1",
  "Verse": "ప్రపంచము ఆర్యుని చేత నెలకొల్పబడింది"
}
```

## Installation and Setup
- Prerequisites
- Python 3.x
- TensorFlow
- Keras
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

## Model Details

### BiLSTM Model for POS Tagging and Spellchecking

The model consists of the following architecture:

- **Embedding Layer**: Converts input words into dense vectors.
- **BiLSTM Layer**: Captures contextual information from both the past and future contexts of a word.
- **Dense Layer**: For classification into different POS tags.
- **Output Layer**: Provides the final POS tag for each word in the sentence.

The model is trained using the **Adam optimizer** and **categorical cross-entropy** as the loss function.

---

## Evaluation Metrics

- **Accuracy**: Measures the proportion of correctly tagged words.
- **Precision**: Measures the precision for each POS tag.
- **Recall**: Measures the recall for each POS tag.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between them.

---

## Results

The model's performance is evaluated on the test dataset, showing improved results in POS tagging and spellchecking:

- **BiLSTM Accuracy**: 89% on the test set.
- **Precision, Recall, F1-Score**: Detailed evaluation metrics for each POS tag.

---

## Usage

### Part-of-Speech Tagging

The trained model can be used to predict POS tags for any input Telugu text. Here's how to use it:

1. Load the trained model.
2. Provide input sentences in Telugu.
3. The model returns the predicted POS tags for each word in the sentence.

Example:

```python
text = "ప్రపంచము ఆర్యుని చేత నెలకొల్పబడింది"
tags = model.predict(text)
print(tags)
```

### Spellchecking

The spellchecking module can be used to identify and correct spelling mistakes in Telugu text:

Example:

```python
sentence = "పారమ్యము"
corrected_sentence = spellcheck(sentence)
print(corrected_sentence)
```

## Future Work

- **Improved Dataset**: Expanding the dataset with more diverse Telugu texts, including modern-day content and informal language, can enhance the model’s robustness and accuracy in real-world applications.
  
- **Multilingual Support**: Integrating support for other Indian languages (e.g., Hindi, Tamil) can broaden the utility of the model, enabling cross-linguistic POS tagging and spellchecking.

- **Contextual Spellchecking**: Enhancing the spellchecking module to account for sentence-level context, improving its ability to correct homophones and contextually inappropriate words.

- **Model Optimization**: Exploring more advanced techniques like attention mechanisms, transformers, or fine-tuning BERT-based models for further improving performance.

- **User Interface**: Developing a simple GUI or web application to make it easier for non-technical users to utilize the POS tagging and spellchecking functionalities.

---

## Conclusion

In this project, we successfully developed a BiLSTM-based model for Part-of-Speech (POS) tagging and spellchecking of Telugu text. The model achieved an accuracy of 89% in POS tagging and demonstrated reliable performance in spelling correction tasks. Through data preprocessing, model training, and evaluation, we showcased the potential of using deep learning techniques in processing complex linguistic tasks for Telugu, a low-resource language. The project has significant implications for improving natural language processing (NLP) tools for Indian languages and can be expanded further with more diverse datasets, multilingual capabilities, and more advanced models.

