# CO2 - NLP Lab Programs (7 to 23)

Each program is a separate, self-contained, Google Colab ready notebook (code only).

## Programs

| # | Program | Notebook |
|---|---------|----------|
| 7 | Coreference Resolution using neuralcoref (or similar) | `07_Coreference_Resolution.ipynb` |
| 8 | Bag-of-Words (BoW) vectorization and representation | `08_Bag_of_Words.ipynb` |
| 9 | TF-IDF Implementation and comparison with BoW | `09_TFIDF_vs_BoW.ipynb` |
| 10 | N-Gram Model (uni-, bi-, tri-gram) generation from corpus | `10_NGram_Model.ipynb` |
| 11 | Cosine Similarity computation between text documents | `11_Cosine_Similarity.ipynb` |
| 12 | Word2Vec word embeddings using gensim on a custom corpus | `12_Word2Vec_Gensim.ipynb` |
| 13 | GloVe Embeddings loading and vector representation | `13_GloVe_Embeddings.ipynb` |
| 14 | Text Similarity using Word Mover's Distance (WMD) | `14_Word_Movers_Distance.ipynb` |
| 15 | Text Classification using Naive Bayes / SVM with TF-IDF | `15_Text_Classification_NB_SVM.ipynb` |
| 16 | Sentiment Analysis using TextBlob and VADER | `16_Sentiment_TextBlob_VADER.ipynb` |
| 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | `17_Topic_Modeling_LDA.ipynb` |
| 18 | Topic Modeling using Latent Semantic Analysis (LSA) | `18_Topic_Modeling_LSA.ipynb` |
| 19 | Opinion Mining on product/service reviews dataset | `19_Opinion_Mining_Reviews.ipynb` |
| 20 | Information Extraction (IE) from structured/unstructured documents | `20_Information_Extraction.ipynb` |
| 21 | Information Retrieval system with ranking using TF-IDF | `21_Information_Retrieval_TFIDF.ipynb` |
| 22 | Sequence Classification using RNN/LSTM (Keras/TensorFlow) | `22_Sequence_Classification_RNN_LSTM.ipynb` |
| 23 | Implementing Attention Mechanism (Basic custom model) | `23_Attention_Mechanism.ipynb` |

## How to run in Google Colab
1. Go to https://colab.research.google.com -> **File -> Upload notebook** and pick any `.ipynb` (or push this folder to GitHub and use *File -> Open notebook -> GitHub*).
2. Run cells top to bottom (**Runtime -> Run all**). The first code cell installs whatever the notebook needs.
3. Programs 22 and 23 (LSTM / Attention) train neural networks: use **Runtime -> Change runtime type -> T4 GPU** for speed.
4. Programs 13 and 14 download GloVe (~65 MB) and 15/17/18 download 20 Newsgroups - an internet connection in Colab is needed (it is by default).

## Notes
- Program 7 uses `fastcoref` because `neuralcoref` no longer installs on current Python/spaCy; a rule-based fallback is included.
- Program 12 trains on a small custom corpus, so similarities are rough but the pipeline is the same for large corpora.