import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import nltk
nltk.data.path.append('./library/stopwords')
from nltk.corpus import stopwords
nltk.data.path.append('./library/averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import torch
import random
import numpy as np


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Define additional stopwords to add
additional_stopwords = {'ed', 'er', 'al', 'es', 'ha',
                        'also', 'et', 'al', 'hf', 'lo', 'york',
                        'http', 'copenhagen', 'denmark', 'fa', 'eg',
                        'doi'
                        }
stop_words = stop_words.union(additional_stopwords)


def fix_seeds():
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def load_data(data_dir):
    """Load text data from .txt files in the specified directory and return filenames with their content."""
    documents = []
    filenames = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                filenames.append(filename)  # Store the filename
    return documents, filenames


def train_ctm_model(document_list, num_topics):
    sp = WhiteSpacePreprocessingStopwords(document_list, stopwords_list=list(stop_words))
    preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()
    tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")
    training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

    fix_seeds()
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=num_topics, num_epochs=50)
    ctm.fit(training_dataset)  # run the model
    return ctm


def get_top_documents_per_topic(ctm, documents, top_n=5):
    """Get the top N documents for each topic."""
    topic_distributions = ctm.get_document_topics(documents)

    top_documents = {}
    for topic_id in range(ctm.num_topics):
        topic_docs = [(doc, topic[1]) for doc, topic in zip(documents, topic_distributions) if topic[0] == topic_id]
        sorted_docs = sorted(topic_docs, key=lambda x: x[1], reverse=True)[:top_n]
        top_documents[topic_id] = sorted_docs

    return top_documents

if __name__ == '__main__':
    documents, filenames = load_data('./data_txt/')
    # processed_texts = [preprocess(document) for document in documents]
    train_ctm_model(document_list=documents, num_topics=10)
    ctm_model = train_ctm_model(document_list=documents, num_topics=10)

