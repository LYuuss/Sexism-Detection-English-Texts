# Sexism Detection in English Texts with Naïve Bayes and Linguistic Feature Analysis

This project aims to build a **text classification model** that predicts whether an English text is **sexist** or **non-sexist**.

Unlike very common classification projects such as Reuters topic classification or wine/iris prediction, this project focuses on a more socially relevant and linguistically rich task: **detecting sexist language in online texts**.

## Objective

The main goal is to study whether a **Naïve Bayes classifier** can effectively distinguish sexist from non-sexist texts.

In addition to classification, the project also includes a linguistic analysis of the dataset in order to better understand which textual features are associated with each class.

## Dataset

We use the following Kaggle dataset:  
[Sexism Detection in English Texts](https://www.kaggle.com/datasets/aadyasingh55/sexism-detection-in-english-texts/data)

## Why this project?

This project is relevant for several reasons:

- it is less overused than standard classroom datasets
- it remains simple enough to be tackled with **Naïve Bayes**
- it allows both **machine learning experimentation** and **linguistic analysis**
- it connects NLP methods with a real-world moderation problem

## Method

The classification pipeline will include:

1. **Dataset exploration**
2. **Text preprocessing**
   - lowercasing
   - tokenization
   - stopword removal
   - punctuation handling
   - optional lemmatization
3. **Feature extraction**
   - Bag-of-Words
   - TF-IDF
4. **Model training**
   - Multinomial Naïve Bayes
5. **Evaluation**
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix

## Linguistic Analyses

A key part of the project is to analyze which features characterize each category.

We will study, for example:

- the **most important words** for the sexist and non-sexist classes
- the most frequent **n-grams**
- the distribution of **Part-of-Speech tags** in each class
- the presence of **named entities**
- stylistic signals such as:
  - pronouns
  - modal verbs
  - adjectives
  - insults / offensive vocabulary
  - punctuation patterns

These analyses will help interpret the classifier and connect the project to concepts studied in class.

## Expected Outcome

The final system should be able to classify a new English text as:

- **Sexist**
- **Non-sexist**

We also expect to identify the linguistic patterns that most strongly contribute to this distinction.
