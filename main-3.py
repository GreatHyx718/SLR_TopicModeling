import os
import PyPDF2
import pandas as pd
import string
import nltk
nltk.data.path.append('./library/stopwords')
from nltk.corpus import stopwords
nltk.data.path.append('./library/averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import Phrases
from gensim.models import CoherenceModel

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


def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)  # Get POS tags
    lemmatized_tokens = []

    for word, tag in tagged_tokens:
        if tag.startswith('NN'):  # Noun
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('JJ'):  # Adjective
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='a'))
        elif tag.startswith('VB'):  # Verb
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('RB'):  # Adverb
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='r'))

    return [word for word in lemmatized_tokens if word.isalnum() and word not in stop_words and len(word) > 1]


"""Load text data from .txt files in the specified directory and return filenames with their content."""
def load_data(data_dir):
    documents = []
    filenames = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                filenames.append(filename)  # Store the filename
    return documents, filenames


def get_top_documents_per_topic(lda_model, corpus, filenames):
    all_topics = lda_model.print_topics()
    docs_per_topic = [[] for _ in all_topics]

    # Iterate through each document to get its topic distribution
    for doc_id, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        for topic_id, prob in doc_topics:
            # add the doc_id & its probability to the topic's doc list
            docs_per_topic[topic_id].append((filenames[doc_id], prob))

    for doc_list in docs_per_topic:
        doc_list.sort(key=lambda id_and_prob: id_and_prob[1], reverse=True)

    return docs_per_topic


if __name__ == '__main__':
    texts, filenames = load_data('./data_txt/')
    processed_texts = [preprocess(text) for text in texts]

    # Create Bigram and Trigram models
    bigram = Phrases(processed_texts, min_count=2, threshold=2)
    # trigram = Phrases(bigram[processed_texts], threshold=2)

    # Apply the models
    bigram_texts = [bigram[text] for text in processed_texts]
    # trigram_texts = [trigram[bigram[text]] for text in processed_texts]

    # Choose either bigram_texts or trigram_texts
    final_texts = bigram_texts  # or bigram_texts, or processed_texts

    # Create a dictionary
    dictionary = corpora.Dictionary(final_texts)
    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in final_texts]

    highest_coherence = 0
    ideal_lda_model = None
    ideal_num_topics = 1

    # Set parameters
    num_topics = 8  # Adjust based on your needs
    # Build LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)


    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic {idx}: {topic}')

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}\n')


    # Get the top 5 documents for each topic
    top_documents = get_top_documents_per_topic(lda_model, corpus, filenames)
    # top_documents = get_top_documents_per_topic(lda_model, corpus, filenames, top_n=5)
    # Print the top 5 documents for each topic
    for topic_id in range(num_topics):
        # print(top_documents[topic_id][:5])
        print(f'\nTop documents for Topic {topic_id}:')
        for doc_name, prob in top_documents[topic_id][:5]:
            print(f'  {doc_name} (Probability: {prob:.4f})')
