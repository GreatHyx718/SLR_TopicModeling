import os
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

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Define additional stopwords to add
additional_stopwords = {'ed', 'er', 'es', 'ha', 'hf', 'lo',
                        'also', 'et', 'al', 'acm', 'dis',
                        'http', 'fa', 'eg', 'doi', 'ny', 'ca',
                        'york'
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
    docs_per_topic = [[] for _ in range(lda_model.num_topics)]

    # Iterate through each document to get its topic distribution
    for doc_id, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)

        for topic_id, prob in doc_topics:
            # add the doc_id & its probability to the topic's doc list
            docs_per_topic[topic_id].append((filenames[doc_id], prob))

    for doc_list in docs_per_topic:
        doc_list.sort(key=lambda id_and_prob: id_and_prob[1], reverse=True)

    return docs_per_topic


def plot_word_clouds(lda_model, num_topics):
    for topic_id in range(num_topics):
        # Extract the words and their probabilities for the topic
        topic_words = lda_model.get_topic_terms(topic_id, topn=20)  # Adjust topn as needed
        word_freq = {lda_model.id2word[id]: prob for id, prob in topic_words}

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Plotting the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {topic_id}')
        plt.axis('off')
        plt.show()

def build_lda(corpus, dictionary):

    num_topics = 16  # Adjust based on your needs

    # Build LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    lda_model.save('lda_model_16.gensim')


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

    # build_lda(corpus, dictionary)

    lda_model = LdaModel.load('lda_model_16.gensim')

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}\n')

    # Print the topics
    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic {idx}: {topic}')

        # Get the top 5 documents for each topic
    top_documents = get_top_documents_per_topic(lda_model, corpus, filenames)
    # Print the top 5 documents for each topic
    for topic_id in range(lda_model.num_topics):
        # print(top_documents[topic_id][:5])
        print(f'\nTop documents for Topic {topic_id}:')
        for doc_name, prob in top_documents[topic_id][:5]:
            print(f'  {doc_name} (Probability: {prob:.4f})')

    # Generate and plot word clouds for each topic
    plot_word_clouds(lda_model, lda_model.num_topics)