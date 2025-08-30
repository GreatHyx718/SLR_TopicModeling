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
import pyLDAvis.gensim_models
import pyLDAvis


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

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocess(text):
    tokens = word_tokenize(remove_punctuation(text.lower()))
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


# Add this function to determine the predominant topic for each document
def get_dominant_topic(lda_model, corpus):
    dominant_topics = []
    for doc in corpus:
        topic_distribution = lda_model.get_document_topics(doc)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])  # Get the topic with the highest probability
        dominant_topics.append(dominant_topic[0])  # Append the topic index
    return dominant_topics


def get_top_documents_per_topic(lda_model, corpus, filenames):
    all_topics = lda_model.print_topics()
    docs_per_topic = [[] for _ in all_topics]

    # Iterate through each document to get its topic distribution
    for doc_id, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        for topic_id, score in doc_topics:
            # ...add the doc_id & its score to the topic's doc list
            docs_per_topic[topic_id].append((filenames[doc_id], score))

    for doc_list in docs_per_topic:
        doc_list.sort(key=lambda id_and_score: id_and_score[1], reverse=True)

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

    # print(bigram_texts)

    # Choose either bigram_texts or trigram_texts
    final_texts = bigram_texts  # or bigram_texts, or processed_texts

    # Create a dictionary
    dictionary = corpora.Dictionary(final_texts)
    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in final_texts]

    highest_coherence = 0
    ideal_lda_model = None
    ideal_num_topics = 1

    # for num_topics in range(1, 31):
    #     lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    #
    #     # Compute Coherence Score
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=final_texts, dictionary=dictionary, coherence='c_v')
    #     coherence_lda = coherence_model_lda.get_coherence()
    #     print(f'Topic Num: {num_topics}; Coherence Score: {coherence_lda}')
    #
    #     if coherence_lda > highest_coherence:
    #         highest_coherence = coherence_lda
    #         ideal_lda_model = lda_model
    #         ideal_num_topics = num_topics
    #
    #     if coherence_lda > 0.3:
    #         vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    #         pyLDAvis.save_html(vis, './results/lda_visualization_' + str(num_topics) + '_topics.html')
    #
    # for idx, topic in ideal_lda_model.print_topics(-1):
    #     print(f'Topic {idx}: {topic}')
    # print(f'Ideal topic Num: {ideal_num_topics}; Highest coherence Score: {highest_coherence}\n')
    #
    # # Get the dominant topic for each document
    # dominant_topics = get_dominant_topic(ideal_lda_model, corpus)
    # # Print the dominant topic for each document
    # for i, doc in enumerate(dominant_topics):
    #     print(f'Document "{filenames[i]}": Dominant Topic {doc}')
    #
    # # Get the top 5 documents for each topic
    # top_documents = get_top_documents_per_topic(ideal_lda_model, corpus, filenames, top_n=5)
    # # Print the top documents for each topic
    # for topic_id, docs in top_documents.items():
    #     print(f'\nTop documents for Topic {topic_id}:')
    #     for doc_name, prob in docs:
    #         print(f'  {doc_name} (Probability: {prob:.4f})')


    # Set parameters
    num_topics = 16  # Adjust based on your needs
    # Build LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # display_topics(lda_model, num_topics=num_topics, num_words=15)

    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic {idx}: {topic}')

    # # Compute Perplexity
    # perplexity = lda_model.log_perplexity(corpus)
    # print(f'Perplexity: {perplexity}')

    # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=final_texts, dictionary=dictionary, coherence='u_mass')
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=final_texts, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print(f'Coherence Score: {coherence_lda}\n')

    # # Get the dominant topic for each document
    # dominant_topics = get_dominant_topic(lda_model, corpus)
    # # Print the dominant topic for each document
    # for i, doc in enumerate(dominant_topics):
    #     print(f'Document "{filenames[i]}": Dominant Topic {doc}')

    # Get the top 5 documents for each topic
    top_documents = get_top_documents_per_topic(lda_model, corpus, filenames)
    # top_documents = get_top_documents_per_topic(lda_model, corpus, filenames, top_n=5)
    # Print the top 5 documents for each topic
    for topic_id in range(num_topics):
        # print(top_documents[topic_id][:5])
        print(f'\nTop documents for Topic {topic_id}:')
        for doc_name, prob in top_documents[topic_id][:5]:
            print(f'  {doc_name} (Probability: {prob:.4f})')



    # # Prepare for visualization
    # vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    #
    # # Save it to an HTML file
    # pyLDAvis.save_html(vis, './results/lda_visualization_' + str(num_topics) + '_topics.html')
    # # pyLDAvis.save_html(vis, 'lda_visualization_test.html')

