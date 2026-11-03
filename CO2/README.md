# NLP Lab - CO2

This folder contains programs 7 to 23 of the NLP lab. The programs cover text representation (Bag-of-Words, TF-IDF, n-grams), word embeddings (Word2Vec, GloVe), text similarity, classification, sentiment analysis, topic modeling, information extraction and retrieval, and a first look at sequence models (RNN, LSTM) and the attention mechanism.

Every program is a separate Jupyter notebook (`.ipynb`) and is written to run on Google Colab. Nothing has to be installed on a local machine. The first cell of each notebook installs the libraries it needs, and the notebooks can be run from top to bottom without any changes.

## Contents

| No. | Program | Notebook |
|-----|---------|----------|
| 7 | Coreference resolution using neuralcoref (or similar) | [07_Coreference_Resolution.ipynb](07_Coreference_Resolution.ipynb) |
| 8 | Bag-of-Words (BoW) vectorization and representation | [08_Bag_of_Words.ipynb](08_Bag_of_Words.ipynb) |
| 9 | TF-IDF implementation and comparison with BoW | [09_TFIDF_vs_BoW.ipynb](09_TFIDF_vs_BoW.ipynb) |
| 10 | N-gram model (uni-, bi-, tri-gram) generation from corpus | [10_NGram_Model.ipynb](10_NGram_Model.ipynb) |
| 11 | Cosine similarity computation between text documents | [11_Cosine_Similarity.ipynb](11_Cosine_Similarity.ipynb) |
| 12 | Word2Vec word embeddings using gensim on a custom corpus | [12_Word2Vec_Gensim.ipynb](12_Word2Vec_Gensim.ipynb) |
| 13 | GloVe embeddings loading and vector representation | [13_GloVe_Embeddings.ipynb](13_GloVe_Embeddings.ipynb) |
| 14 | Text similarity using Word Mover's Distance (WMD) | [14_Word_Movers_Distance.ipynb](14_Word_Movers_Distance.ipynb) |
| 15 | Text classification using Naive Bayes / SVM with TF-IDF | [15_Text_Classification_NB_SVM.ipynb](15_Text_Classification_NB_SVM.ipynb) |
| 16 | Sentiment analysis using TextBlob and VADER | [16_Sentiment_TextBlob_VADER.ipynb](16_Sentiment_TextBlob_VADER.ipynb) |
| 17 | Topic modeling using Latent Dirichlet Allocation (LDA) | [17_Topic_Modeling_LDA.ipynb](17_Topic_Modeling_LDA.ipynb) |
| 18 | Topic modeling using Latent Semantic Analysis (LSA) | [18_Topic_Modeling_LSA.ipynb](18_Topic_Modeling_LSA.ipynb) |
| 19 | Opinion mining on product/service reviews dataset | [19_Opinion_Mining_Reviews.ipynb](19_Opinion_Mining_Reviews.ipynb) |
| 20 | Information extraction (IE) from structured/unstructured documents | [20_Information_Extraction.ipynb](20_Information_Extraction.ipynb) |
| 21 | Information retrieval system with ranking using TF-IDF | [21_Information_Retrieval_TFIDF.ipynb](21_Information_Retrieval_TFIDF.ipynb) |
| 22 | Sequence classification using RNN/LSTM (Keras/TensorFlow) | [22_Sequence_Classification_RNN_LSTM.ipynb](22_Sequence_Classification_RNN_LSTM.ipynb) |
| 23 | Implementing attention mechanism (basic custom model) | [23_Attention_Mechanism.ipynb](23_Attention_Mechanism.ipynb) |

## What each program does

**7. Coreference resolution.** Finds which words in a text refer to the same entity, for example linking "He" back to the person named earlier. `neuralcoref` no longer installs on the current Python and spaCy versions, so the notebook uses `fastcoref`, a newer neural model that does the same job, and prints the clusters of mentions. If `fastcoref` cannot run, a simple rule based resolver written with spaCy takes over and replaces he, she and it with the last matching entity.

**8. Bag-of-Words.** Builds a document-term matrix from scratch (tokenizing, building the vocabulary and counting) and checks that it matches scikit-learn's `CountVectorizer`. It also shows binary BoW, stop word removal and unigram plus bigram features, vectorizes a new sentence and plots the matrix as a heatmap.

**9. TF-IDF vs BoW.** Implements TF-IDF from scratch with the smoothed IDF formula and L2 normalisation, and verifies that the result is the same as `TfidfVectorizer`. The weights of one document are then compared with its BoW counts to show how common words like "the" are pushed down.

**10. N-gram model.** Counts unigrams, bigrams and trigrams in a corpus, calculates conditional probabilities such as P(language | natural), and generates new text by sampling from the bigram and trigram models. The top bigrams are shown as a bar chart. The corpus can be replaced with any text file.

**11. Cosine similarity.** Computes cosine similarity between documents manually with NumPy and with scikit-learn on TF-IDF vectors. A similarity matrix is plotted as a heatmap, the most similar pair of documents is found, and a query is ranked against the documents.

**12. Word2Vec.** Trains a skip-gram Word2Vec model with gensim on a small custom corpus. It shows the vector of a word, similarity between words, the most similar words, the odd one out in a list, a 2-D PCA plot of the vectors, and saves and reloads the model. Because the corpus is tiny, the similarities are only rough.

**13. GloVe embeddings.** Loads the pre-trained `glove-wiki-gigaword-50` vectors through the gensim downloader. It shows word vectors, similarity, nearest words, analogies such as king - man + woman, sentence vectors made by averaging word vectors, and a PCA plot of word groups (countries, animals, technology).

**14. Word Mover's Distance.** Uses GloVe vectors and gensim's `wmdistance` to measure how far apart two sentences are in meaning, even when they share no words. A pairwise distance matrix is plotted, the result is compared with TF-IDF cosine similarity, and a query is matched to the closest sentence.

**15. Text classification.** Classifies posts from four categories of the 20 Newsgroups dataset using TF-IDF features with Multinomial Naive Bayes and Linear SVM. Accuracy, classification report and confusion matrix are shown for both, and the models predict the category of some new sentences.

**16. Sentiment analysis.** Runs TextBlob (polarity and subjectivity) and VADER (positive, neutral, negative and compound score) on a set of sentences that include capital letters, exclamation marks, a smiley, negation and mixed opinions. The two tools are compared in a table and a bar chart.

**17. Topic modeling with LDA.** Trains an LDA model with gensim on 20 Newsgroups documents, prints the top words of each topic, calculates the c_v coherence score, tries 2, 4, 6 and 8 topics to see which number works best, and shows the topic distribution of new text.

**18. Topic modeling with LSA.** Applies truncated SVD on the TF-IDF matrix of the same dataset. It prints the top terms of each latent topic, plots the documents in a 2-D latent space coloured by category, and does a semantic search for a query in the latent space.

**19. Opinion mining.** Works on 20 product reviews written inside the notebook. Overall sentiment is found with VADER, aspects such as battery, camera, screen and price are extracted with POS tagging and a keyword map, and the average sentiment of each aspect is plotted. The most positive and negative opinion words are also listed. To use another dataset, load a CSV that has a column named `review`.

**20. Information extraction.** Part A takes unstructured text and uses spaCy named entity recognition, regular expressions for emails, phone numbers, dates and money, and subject-verb-object triples from the dependency parse, and then builds a JSON record. Part B handles structured data and pulls fields from a CSV table and a JSON object using pandas.

**21. Information retrieval.** Builds a small search engine over ten documents. The text is preprocessed (stop words and Porter stemming), an inverted index is created, and documents are ranked for a query by TF-IDF cosine similarity inside a `SearchEngine` class. Precision@3, recall@3 and F1 are calculated for three example queries.

**22. RNN and LSTM.** Classifies IMDB movie reviews as positive or negative using Keras. A SimpleRNN model and an LSTM model are trained for 3 epochs each, their validation accuracy and loss are plotted together, and the LSTM predicts the sentiment of some new reviews.

**23. Attention mechanism.** Part A shows scaled dot-product attention with NumPy on a toy sentence and plots the attention weights. Part B builds a custom attention layer in Keras and uses it on top of a BiLSTM for IMDB sentiment classification. After training, the notebook prints which words of a review got the highest attention weights.

## How to run

1. Open https://colab.research.google.com and go to File > Open notebook > GitHub.
2. Enter the link or username of this repository and choose a notebook from the `CO2` folder.
3. For programs 22 and 23 switch on the GPU before running: Runtime > Change runtime type > T4 GPU. The other programs run fine on CPU.
4. Click Runtime > Run all, or run the cells one by one from the top.
5. The first cell installs the libraries. Wait for it to finish before running the next cells.

## Libraries used

| Program | Main libraries |
|---------|----------------|
| 7 | fastcoref, spaCy |
| 8, 9, 11 | NumPy, pandas, scikit-learn, matplotlib |
| 10 | NLTK, pandas, matplotlib |
| 12 | gensim, scikit-learn |
| 13, 14 | gensim, POT (for WMD), scikit-learn |
| 15, 18 | scikit-learn |
| 16 | TextBlob, vaderSentiment |
| 17 | gensim, scikit-learn |
| 19 | NLTK, vaderSentiment, pandas |
| 20 | spaCy, pandas |
| 21 | scikit-learn, NLTK |
| 22, 23 | Keras, TensorFlow |

## Datasets

- Programs 15, 17 and 18 use the 20 Newsgroups dataset, which scikit-learn downloads automatically.
- Programs 13 and 14 download the pre-trained GloVe vectors (about 65 MB) through gensim.
- Programs 22 and 23 use the IMDB reviews dataset that comes with Keras.
- Programs 8 to 12, 16, 19, 20 and 21 use small example text written inside the notebooks.

## Notes

- The first run of programs 13, 14, 15, 17, 18, 22 and 23 needs a few minutes extra because the data or vectors are downloaded.
- Program 7 uses `fastcoref` instead of `neuralcoref` because `neuralcoref` is old and does not install on current Colab. If the neural model fails, the rule based fallback in the same notebook still gives output.
- Program 12 is trained on a very small corpus, so the similarity scores are rough. The steps are the same as for a large corpus.
- Programs 22 and 23 train neural networks for only 3 epochs to save time, so the accuracy can be improved by training longer.
- The colourful highlighted entity output of program 20 may not show properly on GitHub. The text outputs are always visible.

## Common problems

- **Import error right after the first cell:** go to Runtime > Restart session and run the notebook again from the top.
- **spaCy model not found (programs 7 and 20):** run `!python -m spacy download en_core_web_sm`, restart the session and run again.
- **NLTK resource missing (programs 10, 19, 21):** run `import nltk; nltk.download("all")` in a new cell.
- **"unauthenticated requests" warning from Hugging Face or similar download warnings:** these are only warnings, the download still works.
- **Training is very slow (programs 22 and 23):** the runtime is probably on CPU. Check Runtime > Change runtime type and select T4 GPU.
