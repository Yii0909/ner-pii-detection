# Personal Information Detection in Text (NER)

AiROST Internship 2026/27 · ML/AI track · Problem 2: Personal Information Detection in Text

A fine-tuned DistilBERT model that reads raw text, finds **people (PER)**, **organisations (ORG)** and **locations (LOC)**, and returns a **redacted version** of the text. It is trained on the public CoNLL-2003 dataset plus a small self-written Malaysian dataset (Malay, English and mixed sentences).

> Example (replace with a screenshot of your own app): `Nur Aisyah binti Ahmad studies at UTM in Johor Bahru.` → `[PER] studies at [ORG] in [LOC].`

![App screenshot](screenshot/AppExample.png)

## Features

- Trained model that detects PER, ORG and LOC (the model does the prediction itself; no external API or ready-made NER pipeline is used)
- Streamlit app: paste text, see highlighted entities with confidence scores, and copy the redacted text
- Bonus 1: redacted output (each entity replaced by its type label)
- Bonus 2: small labeled Malaysian dataset (242 training and 90 test sentences), used for training and evaluation
- Fine-tuning process documented step by step in the notebooks

## Results

The final model was continued from a CoNLL-2003 baseline and trained on 4,000 CoNLL sentences mixed with the Malaysian sentences (repeated 10 times). Scores are entity-level F1 (`seqeval`).

| Run | What changed | Malaysian train sentences | Malaysian test F1 | CoNLL-2003 test F1 |
|---|---|---|---|---|
| 1 (baseline) | DistilBERT fine-tuned on CoNLL-2003 only | 0 | 0.778 | 0.886 |
| 2 | Added Malaysian sentences | 203 | 0.959 | 0.888 |
| 3 (final) | Added Malaysian brand names and sentences with no entities | 242 | **0.965** | **0.888** |

Run 3: precision 0.959, recall 0.972 on the Malaysian test set.

Harder checks on the Malaysian test set (baseline → final model):

| Check | Baseline | Final |
|---|---|---|
| Entities never seen in training found exactly (113 entities) | 86 (76.1%) | 107 (94.7%) |
| Strict set: 27 sentences whose entities are all unseen, F1 | 0.650 | 0.885 |

The strict-set score is the most realistic estimate of how the model handles new Malaysian names. CoNLL-2003 F1 stayed the same, so the Malaysian training did not make the model forget international text.

### Before and after on example sentences

| Sentence | Baseline | Final model |
|---|---|---|
| `Giant dan Tesco menawarkan diskaun besar.` | Tesco → PER, Giant missed | Giant → ORG, Tesco → ORG |
| `Sila hantar borang sebelum hari Jumaat.` | Sila → PER, Jumaat → PER | no entities |

## Repository contents

```
.
├── 01_baseline.ipynb        # Fine-tune DistilBERT on CoNLL-2003 (Run 1)
├── 02_malaysian.ipynb       # Malaysian dataset, retraining, evaluation (Runs 2 and 3)
├── app.py                   # Streamlit app (detection + redaction)
├── requirements.txt         # Python dependencies
├── data/
│   ├── malaysian_train.txt  # 242 labeled training sentences
│   └── malaysian_test.txt   # 90 labeled test sentences
└── screenshots/             # App screenshots
```

## Getting the code (GitHub basics)

1. Install [Git](https://git-scm.com/downloads) and Python 3.10 or newer.
2. Clone the repository and enter the folder:
   ```bash
   git clone https://github.com/Yii0909/ner-pii-detection.git
   cd ner-pii-detection
   ```
3. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app

The trained model is hosted on Hugging Face at [`Yxian909/ner-malaysian`](https://huggingface.co/Yxian909/ner-malaysian), so you do not need to train anything to try it.

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. The first run downloads the model (about 260 MB). Paste text, click **Detect**, and read the highlighted entities and the redacted text.

To use a local model folder instead, change `MODEL_ID` at the top of `app.py`.

## Reproduce the training

Both notebooks were run on Google Colab with a free T4 GPU (Runtime → Change runtime type → T4 GPU).

1. Open `01_baseline.ipynb` and run all cells. It trains the baseline model, evaluates it, and saves it as `ner-model`. Copy that folder to your Google Drive (`MyDrive/ner-model`).
2. Open `02_malaysian.ipynb` and run the cells in order. It loads the baseline from `MyDrive/ner-model`, builds the Malaysian dataset, evaluates the baseline (before), retrains (after), and runs the evaluations above.
3. The last notebook cells that upload to Hugging Face are optional and need your own access token.

A seed (42) is set, so results should be very close to the numbers above on the same library versions (see `requirements.txt`).

### Method

- **Model:** `distilbert-base-cased` with a token-classification head (labels from CoNLL-2003; only PER, ORG and LOC are used in the app).
- **Baseline:** 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01 on CoNLL-2003. Labels are aligned to the first sub-word of each word.
- **Malaysian training:** the baseline is trained further for 3 epochs (same settings) on 4,000 random CoNLL sentences plus the Malaysian training sentences repeated 10 times. The CoNLL sentences stop the model forgetting international text.
- **Evaluation:** entity-level precision, recall and F1 with `seqeval`.
- **Redaction:** each detected entity is replaced with its type label, and neighbouring entities of the same type are merged.

## Malaysian dataset

- Sentences are written in the format `[entity text](TYPE)`, with TYPE as PER, ORG or LOC. Titles such as *Dr.*, *Encik* and *Puan* stay outside the brackets.
- They are written by me and by contributors, in English, Malay and mixed language, using made-up names or well-known public figures and places. Events, programmes and dates are not labeled.
- Some sentences have no entities on purpose, so the model learns not to tag every capitalised word.
- Train and test sentences are separate, and no sentence appears in both.
- Contributors: (add first names, with permission)

## Limitations

- The Malaysian dataset is small and self-written, and the test sentences are in a similar style to the training sentences, so real-world accuracy is likely lower than 0.965.
- Run 3 data was chosen after inspecting errors on the test set, so its test score is slightly optimistic. A fresh test set written by other people would give a fairer estimate.
- Remaining misses on unseen entities include long Malay organisation names (for example "Majlis Daerah Mukah", "Lembaga Hasil Dalam Negeri"), some titles and name formats ("a/p"), and some single-word names.
- Capitalised unfamiliar words can be tagged as names, and the confidence score is not reliable (wrong labels can show 0.99).
- Only people, organisations and locations are detected. Other personal information such as phone numbers, emails and ID numbers is not detected.
- Do not rely on this tool alone to remove personal data from real documents.

## Possible future work

- Test on sentences written by other people, and grow the Malaysian dataset
- Accept `.txt` and `.pdf` files in the app
- A from-scratch model (for example BiLSTM-CRF) to compare with the fine-tuned model

## Acknowledgements

- [CoNLL-2003](https://aclanthology.org/W03-0419/) (Tjong Kim Sang and De Meulder, 2003), loaded from the Hugging Face dataset `tomaarsen/conll2003`
- [DistilBERT](https://huggingface.co/distilbert-base-cased) (Sanh et al., 2019) via Hugging Face Transformers
- AiROST for the problem statement
