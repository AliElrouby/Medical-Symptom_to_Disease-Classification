# Medical Symptom-to-Disease Classification

An NLP classification project that predicts one of **24 disease categories** from a patient's written symptom description.

The project covers data consolidation, exploratory data analysis, text preprocessing, word-embedding experiments, deep-learning baselines, transformer fine-tuning, and quantitative and qualitative model evaluation.

> **Disclaimer:** This is an educational machine-learning project, not a clinical diagnostic system. Its predictions should not be used for medical decisions.

Developed an end-to-end medical text-classification pipeline in **Python and PyTorch** that consolidated two symptom datasets into **1,988 unique descriptions across 24 disease classes**.

The project compares a **BiLSTM + FastText** baseline with **DistilBERT** using stratified data splits, class-weighted loss, dropout, early stopping, confusion matrices, and per-class evaluation metrics.

DistilBERT achieved the best held-out performance with:

* **95.32% Accuracy**
* **95.53% Macro F1**
* **14 errors out of 299 test examples**

The BiLSTM + FastText baseline achieved:

* **94.98% Accuracy**
* **95.37% Macro F1**
* **15 errors out of 299 test examples**

---

## Project Workflow

1. Combined `symptom_to_diagnosis.csv` and `Symptom2Disease.csv`.
2. Normalized disease labels.
3. Removed **65 exact duplicate label-text pairs**.
4. Cleaned text by:

   * Lowercasing
   * Removing unsupported characters
   * Normalizing whitespace
5. Used NLTK tokenization and stop-word removal for the embedding and BiLSTM pipeline.
6. Explored:

   * Class distributions
   * Text lengths
   * Common terms
   * Word clouds
   * Word2Vec embeddings
   * FastText embeddings
   * PCA projections
7. Created a reproducible **70/15/15 stratified train-validation-test split** using random seed `42`.
8. Trained a **BiLSTM + FastText** baseline.
9. Fine-tuned **DistilBERT** for multi-class text classification.
10. Compared models using:

    * Accuracy
    * Weighted F1
    * Macro F1
    * Learning curves
    * Confusion matrices
    * Misclassified examples

---

## Dataset Summary

| Property                           |       Value |
| ---------------------------------- | ----------: |
| Source datasets                    |           2 |
| Rows before deduplication          |       2,053 |
| Exact duplicates removed           |          65 |
| Final unique descriptions          |       1,988 |
| Disease classes                    |          24 |
| Missing labels                     |           0 |
| Missing descriptions               |           0 |
| Examples per class                 |       41-90 |
| Mean examples per class            |       82.83 |
| Mean description length            | 29.86 words |
| Median description length          |    30 words |
| Description length range           |  8-55 words |
| 95th-percentile description length |    42 words |

The smallest classes are:

* **Dimorphic hemorrhoids:** 41 examples
* **Acne:** 46 examples

The largest classes include:

* **Diabetes:** 90 examples
* **Impetigo:** 90 examples
* **Psoriasis:** 90 examples

The largest class is approximately **2.2×** the size of the smallest class.

---

## Data Strengths

* No missing labels or descriptions.
* Exact duplicate rows were removed.
* Contains **24 disease categories** rather than a simple binary classification task.
* Most classes contain approximately **80-90 examples**.
* Stratified splitting preserves class proportions across training, validation, and test sets.
* Natural-language descriptions allow contextual NLP models to learn more than isolated symptom keywords.
* The dataset supports detailed error analysis and per-class evaluation.

---

## Data Limitations

* The dataset is relatively small for deep learning, containing only **1,988 descriptions**.
* Individual disease classes contain only **41-90 examples**.
* The test set contains only **299 examples**, with approximately **6-14 samples per class**.
* A small number of predictions can therefore noticeably affect the final metrics.
* The merged datasets may contain repeated writing patterns or source-specific templates even after exact duplicate removal.
* Random text-level splitting may produce more optimistic results than evaluation on completely independent clinical data.
* The project covers only **24 diseases** and English-language symptom descriptions.
* It does not include:

  * Medical history
  * Laboratory results
  * Imaging
  * Symptom severity
  * Patient demographics
  * Full clinical notes
* Stop-word removal in the BiLSTM pipeline may remove medically important words such as negation terms.
* There is no external clinical validation, probability calibration, fairness evaluation, or prospective testing.

---

## Experimental Setup

| Split      | Examples | Share |
| ---------- | -------: | ----: |
| Training   |    1,391 | 70.0% |
| Validation |      298 | 15.0% |
| Test       |      299 | 15.0% |

The project uses:

* Stratified data splitting
* Random seed `42`
* Class-weighted cross-entropy
* Early stopping
* Validation macro-F1 checkpointing

### BiLSTM + FastText

The BiLSTM baseline uses:

* **150-dimensional FastText embeddings**
* Trainable embeddings
* Bidirectional LSTM
* Hidden size: `128`
* Dropout: `0.5`
* Class-weighted loss
* Validation macro-F1 model selection

### DistilBERT

The transformer model uses:

* Hugging Face `Transformers`
* Maximum sequence length: `96`
* AdamW optimizer
* Class-weighted classification
* Fine-tuning for **4 epochs**

---

## Test Results

The following results come from the held-out **299-example test set**.

| Model             |   Accuracy | Weighted F1 |   Macro F1 | Test Errors |
| ----------------- | ---------: | ----------: | ---------: | ----------: |
| **DistilBERT**    | **95.32%** |  **95.25%** | **95.53%** |      **14** |
| BiLSTM + FastText |     94.98% |      95.09% |     95.37% |          15 |

DistilBERT improved over the BiLSTM baseline by:

* **+0.34 percentage points in accuracy**
* **+0.16 percentage points in macro F1**
* **1 fewer test error**

The difference between the two models is relatively small, showing that the simpler BiLSTM architecture remains competitive on this dataset.

---

## Validation Results

| Model                 | Validation Accuracy | Validation Macro F1 |
| --------------------- | ------------------: | ------------------: |
| **BiLSTM + FastText** |          **95.97%** |          **96.23%** |
| DistilBERT            |              95.64% |              95.79% |

Both models showed stable learning behavior.

The close validation and test results demonstrate that using a transformer architecture does not automatically guarantee a large improvement when working with a relatively small dataset.

---

## Model Evaluation

The models were evaluated using several complementary metrics.

### Accuracy

Measures the percentage of correctly classified symptom descriptions.

### Macro F1

Calculates the F1 score independently for every disease class and then gives each class equal weight.

Macro F1 is particularly important for this project because the dataset contains moderate class imbalance.

### Weighted F1

Calculates F1 while weighting each disease according to its number of examples.

### Confusion Matrix

Used to identify which diseases are most frequently confused with one another.

### Misclassification Analysis

Incorrect predictions were manually inspected to understand whether errors were caused by:

* Overlapping symptoms
* Similar disease descriptions
* Limited examples
* Text preprocessing
* Model limitations

---

## Interpretation

DistilBERT achieved the strongest test performance, but only by a small margin over the **BiLSTM + FastText** baseline.

This suggests that both approaches can perform effectively on this curated dataset.

The results also demonstrate the importance of building a strong baseline before assuming that a more complex transformer architecture will significantly outperform a simpler model.

Macro F1 is used as the main comparison metric because it gives every disease class equal importance despite differences in class frequency.

The project demonstrates effective **text classification within this specific dataset**.

It does **not** demonstrate general medical diagnosis capability.

Before considering real-world medical use, the system would require:

* External clinical datasets
* Source-separated evaluation
* More diverse patient descriptions
* Confidence calibration
* Safety-focused testing
* Fairness analysis
* Clinician review
* Prospective validation

---

## Technologies

* Python
* Jupyter Notebook
* Google Colab
* PyTorch
* Hugging Face Transformers
* NLTK
* Gensim
* scikit-learn
* pandas
* NumPy
* Matplotlib
* Seaborn
* WordCloud

---

## Repository Structure

```text
.
├── Phase_1_&_2.ipynb
├── Phase1.ipynb
├── data.csv
│
└── Datasource/
    ├── DataCollection.ipynb
    ├── symptom_to_diagnosis.csv
    ├── Symptom2Disease.csv
    └── ...
```

### Main Files

#### `Phase_1_&_2.ipynb`

Main notebook containing:

* Exploratory data analysis
* Text preprocessing
* Embedding experiments
* BiLSTM training
* DistilBERT fine-tuning
* Model evaluation
* Confusion matrices
* Learning curves
* Error analysis

#### `Datasource/DataCollection.ipynb`

Handles:

* Dataset loading
* Dataset consolidation
* Label normalization
* Duplicate removal

#### `data.csv`

Final cleaned and deduplicated dataset containing **1,988 symptom descriptions**.

#### `Phase1.ipynb`

Earlier exploratory work covering preprocessing and word-embedding experiments.


## Disclaimer

This project is intended for **educational and machine-learning research purposes only**.

The model should not be used to diagnose diseases, recommend treatment, or make medical decisions.
