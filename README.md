# Sexism Detection in English Texts with Naïve Bayes and Linguistic Feature Analysis

## CLI Application

The project includes an interactive **command-line interface** to explore and use the sexism detection pipeline from the terminal.

With the CLI, you can:

- choose one or more classification methods
- evaluate selected models on the active train/test/dev CSV datasets
- manage datasets by switching files, duplicating them, or appending labeled examples
- enter a custom English text and get a **sexist / non-sexist** prediction
- adjust CLI options such as debug mode and cleanup behavior

(misses tsne implementation for now)

### Usage
In your python environment :
```python3 main.py```


# Sexism Detection in English Texts

This project aims to build a **text classification system** that predicts whether an English text is **sexist** or **non-sexist**.

Unlike very common classification projects such as Reuters topic classification or wine/iris prediction, this project focuses on a more socially relevant and linguistically rich task: **detecting sexist language in online texts**.

---

## Objective

The main goal is to study whether a **Naïve Bayes classifier** can effectively distinguish sexist from non-sexist texts.

In addition to this baseline approach, the project also explores **modern transformer-based models (BERTweet)** to compare classical and contextual NLP methods.

A second objective is to perform a **linguistic and interpretability analysis** to better understand which textual features are associated with each class.

---

## Dataset

We use the following Kaggle dataset:  
[Sexism Detection in English Texts](https://www.kaggle.com/datasets/aadyasingh55/sexism-detection-in-english-texts/data)

---

## Why this project?

This project is relevant for several reasons:

- it is less overused than standard classroom datasets  
- it remains accessible with **Naïve Bayes as a baseline**  
- it allows both **machine learning experimentation** and **linguistic analysis**  
- it connects NLP methods with a real-world moderation problem  
- it enables comparison between **classical NLP (Bag-of-Words)** and **modern contextual embeddings (BERT)**  

---

## Method

### 1. Classical NLP Pipeline (Baseline)

The first part of the project focuses on interpretable models:

1. **Dataset exploration**
2. **Text preprocessing**
   - lowercasing
   - tokenization
   - stopword removal (with custom variants)
   - punctuation handling
   - optional stemming / lemmatization
3. **Feature extraction**
   - Bag-of-Words
   - TF-IDF
   - n-grams (including bigrams)
4. **Model training**
   - Multinomial Naïve Bayes
5. **Evaluation**
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix

Several configurations are compared (raw vs processed text, unigrams vs bigrams, TF-IDF vs counts).

---

### 2. Transformer-based Approach (BERTweet)

The project also includes a more advanced approach using a **pretrained transformer model**:

- Model: `NLP-LTU/bertweet-large-sexism-detector`
- Architecture: BERTweet (RoBERTa-based model trained on tweets)
- Task: binary sexism classification

This model leverages **contextual embeddings**, allowing it to capture:

- semantic meaning  
- word order  
- subtle linguistic patterns  

---

## Embedding Visualization

To better understand how texts are represented in embedding space, we project high-dimensional embeddings into 2D using:

- **PCA (dimensionality reduction)**
- **t-SNE (non-linear projection)**

### Observations

- A **clear separation** emerges between sexist and non-sexist texts  
- Sexist texts form **dense clusters**, indicating strong semantic similarity  
- Non-sexist texts are more **dispersed**, suggesting higher variability  
- A **mixed region** appears, corresponding to ambiguous or subtle cases  

These visualizations confirm that **BERTweet embeddings encode meaningful structure for the classification task**.

---

## Linguistic Analyses

A key part of the project is to analyze which features characterize each category.

We study:

- the **most discriminative words** learned by Naïve Bayes  
- the most frequent **n-grams (including bigrams)**  
- differences in **lexical usage between classes**  
- stylistic signals such as:
  - pronouns
  - modal verbs
  - adjectives
  - offensive vocabulary
  - punctuation patterns  

These analyses help interpret the model and connect the project to NLP concepts studied in class.

---

## Key Findings

- **Count-based models outperform TF-IDF** for Naïve Bayes in this task  
- **Preprocessing improves unigram models**, but can hurt bigram representations  
- **Bigrams improve detection of sexist texts** when rare features are filtered (`min_df`)  
- **BERTweet embeddings show strong class separation**, confirming the importance of contextual representations  
- The choice of model involves a trade-off between:
  - interpretability (Naïve Bayes)
  - performance and semantic understanding (transformers)

---

## Expected Outcome

The final system should be able to classify a new English text as:

- **Sexist**
- **Non-sexist**

In addition, the project provides:

- an interpretable baseline model  
- insights into linguistic patterns of sexism  
- visual evidence of semantic structure in embedding space  

---

## Future Work

Possible improvements include:

- fine-tuning BERTweet on the dataset  
- testing other classical models (Logistic Regression, SVM)  
- exploring character n-grams  
- improving interpretability of transformer predictions  
