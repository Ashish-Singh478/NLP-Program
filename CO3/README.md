<div align="center">

# 🤖 NLP Lab &mdash; CO3

### Transformers, Pre-trained Language Models & Document Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-FFD21E?logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-ready-F9AB00?logo=googlecolab&logoColor=white)
![Programs](https://img.shields.io/badge/Programs-24_to_32-success)

*Nine self-contained, Colab-ready notebooks covering BERT, GPT-2, MarianMT, T5, BART, chatbots and document boundary detection.*

</div>

---

## 📚 Programs

| # | Program | Notebook | Libraries | Hardware | Run time |
|:-:|---------|----------|-----------|:--------:|:--------:|
| **24** | Fine-Tuning BERT for text classification (Hugging Face) | [`24_Fine_Tuning_BERT_Text_Classification.ipynb`](24_Fine_Tuning_BERT_Text_Classification.ipynb) | transformers, datasets | GPU recommended | ~10 min |
| **25** | Sentence Embeddings using BERT and GPT-2 | [`25_Sentence_Embeddings_BERT_GPT2.ipynb`](25_Sentence_Embeddings_BERT_GPT2.ipynb) | transformers, scikit-learn | CPU ok | ~3 min |
| **26** | Text Generation using GPT-2 on a custom dataset | [`26_Text_Generation_GPT2_Custom_Dataset.ipynb`](26_Text_Generation_GPT2_Custom_Dataset.ipynb) | transformers, datasets | GPU recommended | ~5 min |
| **27** | Named Entity Recognition using Transformers (BERT) | [`27_Named_Entity_Recognition_BERT.ipynb`](27_Named_Entity_Recognition_BERT.ipynb) | transformers | CPU ok | ~2 min |
| **28** | Machine Translation using MarianMT / T5 | [`28_Machine_Translation_MarianMT_T5.ipynb`](28_Machine_Translation_MarianMT_T5.ipynb) | transformers, sacrebleu | CPU ok | ~5 min |
| **29** | Text Summarization using BART / T5 | [`29_Text_Summarization_BART_T5.ipynb`](29_Text_Summarization_BART_T5.ipynb) | transformers, evaluate (ROUGE) | GPU recommended | ~5 min |
| **30** | Chatbot Development using pre-trained transformer | [`30_Chatbot_Pretrained_Transformer.ipynb`](30_Chatbot_Pretrained_Transformer.ipynb) | transformers | GPU recommended | ~5 min |
| **31** | Automatic Document Classification and Separation using ML | [`31_Automatic_Document_Classification_Separation_ML.ipynb`](31_Automatic_Document_Classification_Separation_ML.ipynb) | scikit-learn | CPU ok | < 1 min |
| **32** | Hybrid Probabilistic + FSM based Sequence Modeling for document boundary detection | [`32_Hybrid_Probabilistic_FSM_Document_Boundary.ipynb`](32_Hybrid_Probabilistic_FSM_Document_Boundary.ipynb) | scikit-learn, NumPy | CPU ok | < 1 min |

> **Run time** is approximate on a free Colab runtime (with GPU where recommended).

---

## 🗂️ Folder Structure

```text
CO3/
├── 24_Fine_Tuning_BERT_Text_Classification.ipynb
├── 25_Sentence_Embeddings_BERT_GPT2.ipynb
├── 26_Text_Generation_GPT2_Custom_Dataset.ipynb
├── 27_Named_Entity_Recognition_BERT.ipynb
├── 28_Machine_Translation_MarianMT_T5.ipynb
├── 29_Text_Summarization_BART_T5.ipynb
├── 30_Chatbot_Pretrained_Transformer.ipynb
├── 31_Automatic_Document_Classification_Separation_ML.ipynb
└── 32_Hybrid_Probabilistic_FSM_Document_Boundary.ipynb
```

---

## 🚀 How to Run

1. **Open a notebook in Google Colab**
   `colab.research.google.com` → **File → Open notebook → GitHub** → select this repository → choose a notebook from `CO3/`.
2. **(Programs 24, 26, 29, 30)** switch on the GPU: **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. The first cell of each notebook installs the required libraries.
4. Wait for the models to download on the first run (sizes below), then check the outputs below every cell.

---

## 🧠 Models Used

| Notebook | Pre-trained model | Approx. download |
|----------|-------------------|:----------------:|
| 24 | `bert-base-uncased` | 440 MB |
| 25 | `bert-base-uncased`, `gpt2` | 440 MB + 550 MB |
| 26 | `gpt2` | 550 MB |
| 27 | `dslim/bert-base-NER` | 430 MB |
| 28 | `Helsinki-NLP/opus-mt-en-hi`, `opus-mt-en-fr`, `t5-small` | 300 MB + 300 MB + 240 MB |
| 29 | `facebook/bart-large-cnn`, `t5-small` | 1.6 GB + 240 MB |
| 30 | `microsoft/DialoGPT-medium`, `Qwen/Qwen2.5-0.5B-Instruct` | 860 MB + 1 GB |
| 31, 32 | none (classical ML on generated pages) | - |

---

## 📝 Notes

- **Programs 31 and 32** generate a synthetic multi-page document stream (invoice, resume, research paper, legal, news), so they need no dataset download and run in seconds.
- **Program 32** compares three approaches: similarity threshold, a probabilistic model (logistic regression) and a **hybrid probabilistic + finite-state machine** decoder (Viterbi with FSM constraints).
- **Programs 30** includes an interactive chat cell: type your message and enter `quit` to stop.
- Outputs of training cells (24, 26) vary slightly from run to run.

---

<div align="center">

⭐ *CO3 &mdash; Natural Language Processing Lab* ⭐

</div>
