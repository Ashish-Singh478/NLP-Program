# NLP Lab - CO3

This folder contains programs 24 to 32 of the NLP lab. The programs are about transformer based models (BERT, GPT-2, BART, T5, MarianMT), a chatbot, and a small document processing task done with classical machine learning.

Every program is a separate Jupyter notebook (`.ipynb`) and is written to run on Google Colab. Nothing has to be installed on a local machine. The first cell of each notebook installs the libraries it needs, and the notebooks can be run from top to bottom without any changes.

## Contents

| No. | Program | Notebook |
|-----|---------|----------|
| 24 | Fine-tuning BERT for text classification using Hugging Face | [24_Fine_Tuning_BERT_Text_Classification.ipynb](24_Fine_Tuning_BERT_Text_Classification.ipynb) |
| 25 | Sentence embeddings using BERT and GPT-2 | [25_Sentence_Embeddings_BERT_GPT2.ipynb](25_Sentence_Embeddings_BERT_GPT2.ipynb) |
| 26 | Text generation using GPT-2 on a custom dataset | [26_Text_Generation_GPT2_Custom_Dataset.ipynb](26_Text_Generation_GPT2_Custom_Dataset.ipynb) |
| 27 | Named entity recognition using transformers (BERT) | [27_Named_Entity_Recognition_BERT.ipynb](27_Named_Entity_Recognition_BERT.ipynb) |
| 28 | Machine translation using MarianMT and T5 | [28_Machine_Translation_MarianMT_T5.ipynb](28_Machine_Translation_MarianMT_T5.ipynb) |
| 29 | Text summarization using BART and T5 | [29_Text_Summarization_BART_T5.ipynb](29_Text_Summarization_BART_T5.ipynb) |
| 30 | Chatbot development using pre-trained transformers | [30_Chatbot_Pretrained_Transformer.ipynb](30_Chatbot_Pretrained_Transformer.ipynb) |
| 31 | Automatic document classification and separation using ML | [31_Automatic_Document_Classification_Separation_ML.ipynb](31_Automatic_Document_Classification_Separation_ML.ipynb) |
| 32 | Hybrid probabilistic + FSM based sequence modeling for document boundary detection | [32_Hybrid_Probabilistic_FSM_Document_Boundary.ipynb](32_Hybrid_Probabilistic_FSM_Document_Boundary.ipynb) |

## What each program does

**24. Fine-tuning BERT.** Takes the pre-trained `bert-base-uncased` model and fine-tunes it for sentiment classification on the IMDB movie review dataset. To keep the training short, a subset of 2000 reviews is used for training and 500 for testing. The notebook trains for 2 epochs with the Hugging Face `Trainer`, then prints accuracy and F1, a classification report, a confusion matrix and the training loss curve. At the end the model predicts the sentiment of a few new sentences and the model is saved.

**25. Sentence embeddings.** Converts sentences into fixed size vectors using BERT (mean pooling and the [CLS] token) and GPT-2 (mean pooling). Cosine similarity between the sentences is shown as heatmaps for all three embeddings, so that paraphrase pairs can be compared with unrelated pairs. The embeddings are also plotted in 2-D using PCA, and a small semantic search example finds the closest sentence to a query.

**26. Text generation with GPT-2.** Fine-tunes GPT-2 on a small custom text collection (a few lines about college, programming and machine learning). The same prompts are given to the model before and after fine-tuning to show how the style of the generated text changes. The last cell compares greedy decoding with beam search. Any other text can be used by replacing the `lines` list in the notebook.

**27. Named entity recognition.** Uses the `dslim/bert-base-NER` model to find persons, organisations, locations and miscellaneous entities in a paragraph. The output is shown as a table, as highlighted text, as a count per entity type, and as raw token level BIO tags for a short sentence.

**28. Machine translation.** Translates English sentences to Hindi and French using MarianMT models (`opus-mt-en-hi`, `opus-mt-en-fr`) and to German and French using T5-small with task prefixes. The French outputs of both models are compared using the BLEU score against reference translations. There is also a cell to translate any sentence you type.

**29. Text summarization.** Summarizes a news style paragraph with `facebook/bart-large-cnn` and `t5-small`. The two summaries are compared using ROUGE-1, ROUGE-2 and ROUGE-L against a reference summary, and a last cell shows how changing the maximum summary length changes the output.

**30. Chatbot.** Two chatbots are built. The first uses DialoGPT-medium, which keeps the conversation history and replies turn by turn. The second uses the instruction tuned `Qwen2.5-0.5B-Instruct` model with a system prompt and a chat template, which behaves more like a ChatGPT style assistant. Each one has a short scripted demo and an interactive cell where you can type your own messages.

**31. Document classification and separation.** Generates a set of fake multi-page documents of five types (invoice, resume, research paper, legal, news) and trains Naive Bayes and Linear SVM classifiers with TF-IDF features to classify each page. After that, a mixed stream of pages is separated back into individual documents by detecting where one document ends and the next begins, using a change in predicted class, a header on the page and the similarity with the previous page. The separated documents are written as text files into a folder called `separated_documents`.

**32. Hybrid probabilistic + FSM boundary detection.** Works on the same kind of page stream as program 31, but treats boundary detection as sequence labelling where every page is either `B` (first page of a document) or `I` (continuation page). A logistic regression model gives the probability of a boundary for each page using the similarity with the previous page, the change in predicted class distribution and a header flag. A finite state machine then adds rules (the stream must start with `B`, and a document must have at least 2 pages so `B` cannot follow `B`), and the final tags are decoded with the Viterbi algorithm. The notebook compares a plain similarity threshold, the probabilistic model alone and the hybrid model using precision, recall and F1.

## How to run

1. Open https://colab.research.google.com and go to File > Open notebook > GitHub.
2. Enter the link or username of this repository and choose a notebook from the `CO3` folder.
3. For programs 24, 26, 29 and 30 switch on the GPU before running: Runtime > Change runtime type > T4 GPU.
4. Click Runtime > Run all, or run the cells one by one from the top.
5. The first cell installs the libraries. Wait for it to finish before running the next cells.

## Libraries and models

| Program | Main libraries | Pre-trained model(s) |
|---------|----------------|----------------------|
| 24 | transformers, datasets, scikit-learn | google-bert/bert-base-uncased |
| 25 | transformers, scikit-learn | google-bert/bert-base-uncased, openai-community/gpt2 |
| 26 | transformers, datasets | openai-community/gpt2 |
| 27 | transformers | dslim/bert-base-NER |
| 28 | transformers, sentencepiece, sacrebleu | Helsinki-NLP/opus-mt-en-hi, opus-mt-en-fr, google-t5/t5-small |
| 29 | transformers, evaluate, rouge_score | facebook/bart-large-cnn, google-t5/t5-small |
| 30 | transformers | microsoft/DialoGPT-medium, Qwen/Qwen2.5-0.5B-Instruct |
| 31 | scikit-learn, NumPy, pandas | none |
| 32 | scikit-learn, NumPy, pandas | none |

## Datasets

- Programs 24 uses the IMDB movie reviews dataset (`stanfordnlp/imdb`), which is downloaded automatically through the `datasets` library.
- Program 26 uses a small custom text written inside the notebook.
- Programs 28 and 29 use short example sentences and a sample paragraph written inside the notebook.
- Programs 31 and 32 generate their own data, so no download is needed.

## Notes

- The pre-trained models are downloaded from Hugging Face on the first run, so the first run takes a few minutes. The largest one is BART in program 29 (about 1.6 GB).
- Programs 24 and 26 train on a small amount of data so that they finish quickly on the free Colab GPU. Because of random shuffling and sampling, results can change a little from one run to another.
- Programs 31 and 32 run in a few seconds on CPU. Since their data is generated, the scores in these two notebooks show how the methods compare with each other, not how they would perform on real scanned documents.
- Program 30 has cells that wait for user input using `input()`. When "Run all" reaches such a cell it will stop and wait for a message. Type `quit` to leave the chat and continue.

## Common problems

- **"unauthenticated requests to the HF Hub" warning:** this is only a warning. Downloads still work, they are just a bit slower.
- **Import error right after the first cell:** go to Runtime > Restart session and run the notebook again from the top.
- **Out of memory on the GPU:** restart the session, run only one notebook at a time, or reduce `per_device_train_batch_size` in programs 24 and 26.
- **Training is very slow:** the runtime is probably on CPU. Check Runtime > Change runtime type and select T4 GPU.
