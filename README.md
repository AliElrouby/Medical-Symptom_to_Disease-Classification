# Medical Symptom-to-Disease Classification

An NLP classification project that predicts one of 24 disease categories from a patient's written symptom description. The project covers data consolidation, exploratory analysis, text preprocessing, word-embedding experiments, deep-learning baselines, transformer fine-tuning, and quantitative and qualitative model evaluation.

> This is an educational machine-learning project, not a clinical diagnostic system. Its predictions should not be used for medical decisions.

Developed an end-to-end medical text-classification pipeline in Python and PyTorch that consolidated two symptom datasets into 1,988 unique descriptions across 24 disease classes. Trained and evaluated a BiLSTM/FastText baseline and three Hugging Face transformer models using stratified data splits, class-weighted loss, dropout, early stopping, confusion matrices, and per-class metrics. Bio_ClinicalBERT achieved the best held-out result with **99.00% accuracy and 99.04% macro F1 on 299 test examples**, reducing errors from 15 in the BiLSTM baseline to 3.

## Project workflow

1. Combined `symptom_to_diagnosis.csv` and `Symptom2Disease.csv`.
2. Normalized labels and removed 65 exact duplicate label-text pairs.
3. Cleaned text by lowercasing, removing unsupported characters, and normalizing whitespace.
4. Used NLTK tokenization and stop-word removal for the embedding and BiLSTM path.
5. Explored class frequency, text length, common terms, word clouds, and PCA projections of Word2Vec and FastText embeddings.
6. Created a reproducible, stratified 70/15/15 split with random seed 42.
7. Trained a BiLSTM/FastText baseline and fine-tuned DistilBERT, Bio_ClinicalBERT, and BiomedBERT.
8. Compared accuracy, weighted F1, macro F1, learning curves, confusion matrices, and misclassified examples.

## Dataset summary

| Property | Value |
|---|---:|
| Source datasets | 2 |
| Rows before deduplication | 2,053 |
| Exact duplicates removed | 65 |
| Final unique descriptions | 1,988 |
| Disease classes | 24 |
| Missing labels or descriptions | 0 |
| Examples per class | 41-90 |
| Mean examples per class | 82.83 |
| Mean description length | 29.86 words |
| Median description length | 30 words |
| Description length range | 8-55 words |
| 95th-percentile description length | 42 words |

The smallest classes are dimorphic hemorrhoids (41 examples) and acne (46), while diabetes, impetigo, and psoriasis each contain 90 examples. The largest class is therefore about 2.2 times the size of the smallest class.

### Data strengths

- The final dataset has no missing labels, missing descriptions, or exact duplicate rows.
- It includes several disease families and symptom-writing patterns instead of treating the task as binary classification.
- Most classes contain roughly 80-90 examples, and stratification preserves their proportions across the three splits.
- Natural-language descriptions support contextual models and realistic error analysis better than isolated symptom keywords.

### Data limitations

- The dataset is small for deep learning: only 1,988 descriptions and 41-90 examples per disease.
- The 299-example test set contains only 6-14 examples per class, so a few predictions can noticeably change the reported metrics.
- The merged data may contain source-specific or repeated writing templates even after exact duplicates are removed. A random text-level split can therefore produce more optimistic results than evaluation on a fully independent source or real clinical notes.
- The task covers only 24 labels and English symptom descriptions; it does not represent the full range of diseases, patient populations, languages, medical history, laboratory results, or symptom severity.
- Stop-word removal in the BiLSTM path can discard medically important negation words. The transformer path is less aggressively reduced and preserves more sentence context.
- There is no external clinical validation, probability calibration, fairness analysis, or prospective evaluation, so the high offline score should not be interpreted as clinical readiness.

## Experimental setup

| Split | Examples | Share |
|---|---:|---:|
| Training | 1,391 | 70.0% |
| Validation | 298 | 15.0% |
| Test | 299 | 15.0% |

The project uses stratified splitting and a fixed random seed. Class-weighted cross-entropy reduces the effect of the remaining imbalance. The BiLSTM uses trainable 150-dimensional FastText embeddings, a bidirectional hidden size of 128, dropout of 0.5, and validation-macro-F1 checkpointing. Transformer models use a maximum sequence length of 96 and are fine-tuned for four epochs with AdamW.

## Test results

The following values come from the saved output cells for the held-out 299-example test set, not from the training set or the plots alone.

| Model | Accuracy | Weighted F1 | Macro F1 | Test errors |
|---|---:|---:|---:|---:|
| **Bio_ClinicalBERT** | **99.00%** | **99.00%** | **99.04%** | **3** |
| BiomedBERT | 96.99% | 96.95% | 97.15% | 9 |
| DistilBERT | 95.32% | 95.25% | 95.53% | 14 |
| BiLSTM + FastText | 94.98% | 95.09% | 95.37% | 15 |

Bio_ClinicalBERT improved test accuracy by **4.02 percentage points** and macro F1 by **3.67 percentage points** over the BiLSTM/FastText baseline. Its three recorded errors were bronchial asthma predicted as common cold, psoriasis predicted as arthritis, and dengue predicted as chicken pox. These mistakes are plausible confusion pairs because their symptom descriptions can overlap.

## Validation results from the learning curves

| Model | Best/final validation accuracy | Best/final validation macro F1 |
|---|---:|---:|
| BiLSTM + FastText | 95.97% | 96.23% |
| DistilBERT | 95.64% | 95.79% |
| Bio_ClinicalBERT | 98.32% | 98.43% |
| BiomedBERT | 98.32% | 98.43% |

The validation curves show steady learning for all four models. The lower BiomedBERT test result despite matching Bio_ClinicalBERT on the validation split also shows why final model selection should use a held-out test set and why a larger external test set would strengthen the conclusion.

## Interpretation

The experiment supports domain-specific pretraining for this dataset: Bio_ClinicalBERT produced the strongest overall test performance, while BiomedBERT ranked second. DistilBERT remained a competitive lighter general-domain model, and the BiLSTM/FastText system provided a strong, simpler baseline. Macro F1 is the primary comparison metric because it gives every disease class equal importance despite the moderate class imbalance.

The result demonstrates effective classification within this particular curated dataset. It does not demonstrate general medical diagnosis ability. Before real-world use, the system would need source-separated and external validation, more diverse patient text, confidence calibration, safety-focused evaluation, and clinician review.

## Technologies

- Python, Jupyter, Google Colab
- PyTorch
- Hugging Face Transformers
- NLTK and Gensim
- scikit-learn
- pandas, NumPy, Matplotlib, Seaborn, and WordCloud

## Repository guide

- `Phase_1_&_2.ipynb`: complete exploratory analysis, model training, and comparison notebook with saved outputs.
- `Datasource/DataCollection.ipynb`: dataset merging, label normalization, and duplicate removal.
- `data.csv`: final deduplicated dataset.
- `Datasource/`: original input datasets and dependency list.
- `Phase1.ipynb`: earlier exploratory preprocessing and embedding work.

## Reproducibility note

The saved notebook outputs contain the reported experiments, but the current source of `Phase_1_&_2.ipynb` is not fully clean-run ready. The transformer split references a `transformer_text` column that is not created in the preceding saved cell; it should be assigned from the cleaned full text, for example `phase2_data['transformer_text'] = phase2_data['clean_input']`. The notebook also contains a stored `FileNotFoundError` from a Colab run in which `data.csv` was not in the runtime working directory. These issues do not change the recorded output metrics, but they should be fixed before claiming one-click reproducibility.
