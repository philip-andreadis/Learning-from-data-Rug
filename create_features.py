import os
import csv
import nltk
import demoji
import pickle
import numpy as np
import json
import re
import pandas as pd
from sklearn.utils import resample
from sklearn.utils import class_weight
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from better_profanity import profanity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer
from tensorflow.keras.layers import TextVectorization
from argparser import create_arg_parser
import tensorflow as tf
from multiprocessing import Pool
import wordsegment

"""This script is used to read the train, dev and test files with raw data and transform them to 
features depending on the model provided. Refer to argparser.py for the possible arguments."""

nltk.download('punkt')


def read_tsv(filename):
    """Reads the features and labels from a tsv file specified by the filename"""
    X = []
    Y = []
    with open(filename, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            X.append(row[0])
            Y.append(row[1])
    return X, Y


def read_data(args):
    """Read training, validation and testing features and labels"""
    X_train, Y_train = read_tsv(args.train_file)
    X_dev, Y_dev = read_tsv(args.dev_file)
    X_test, Y_test = read_tsv(args.test_file)
    return [X_train, Y_train, X_dev, Y_dev, X_test, Y_test]


def pos_tag(X):
    """Apply pos tagging to words in a string. It returns a new string with where the words
    are replaced by: word_POSTAG"""
    X_new = []
    for idx, feature in enumerate(X):
        new_feature = ''
        sents = sent_tokenize(feature)
        for sent in sents:
            tagged_sent = nltk.pos_tag(word_tokenize(sent))
            new_feature += ' '.join(['_'.join(word) for word in tagged_sent])
            new_feature += ' '
        X_new.append(new_feature)
    return X_new


def stem(X):
    """Applies stemming on the words in a string"""
    X_new = []
    stemmer = PorterStemmer()
    for feature in X:
        X_new.append(' '.join([stemmer.stem(word) for word in word_tokenize(feature)]))
    return X_new


def mark_offensive_words(x):
    """Masks offfensive words with the OFF token"""
    return profanity.censor(x, 'OFF')


def remove_offensive_words(x):
    """Removes offensive words"""
    return profanity.censor(x, '')


def segment_hashtag(x):
    """Splits a hashtag into separate terms"""
    words = x.split()
    new_words = []
    for word in words:
        if word.startswith('#'):
            word = word[1:]
            segments = wordsegment.segment(word)
            new_words.extend(segments)
        else:
            new_words.append(word)
    return ' '.join(new_words)



def preprocess(data, args):
    """Applies all preprocessing based on what the user specified"""
    [X_train, X_dev, X_test] = data

    # Remove Usernames
    X_train = np.vectorize(lambda x: x.replace('@USER', ''))(X_train)
    X_dev = np.vectorize(lambda x: x.replace('@USER', ''))(X_dev)
    X_test = np.vectorize(lambda x: x.replace('@USER', ''))(X_test)

    # Replace URLs
    X_train = np.vectorize(lambda x: x.replace('URL', 'http'))(X_train)
    X_dev = np.vectorize(lambda x: x.replace('URL', 'http'))(X_dev)
    X_test = np.vectorize(lambda x: x.replace('URL', 'http'))(X_test)

    # Segment hashtags
    wordsegment.load()
    X_train = np.vectorize(lambda x: segment_hashtag(x))(X_train)
    X_dev = np.vectorize(lambda x: segment_hashtag(x))(X_dev)
    X_test = np.vectorize(lambda x: segment_hashtag(x))(X_test)

    # Lowercase characters
    X_train = np.vectorize(lambda x: x.lower())(X_train)
    X_dev = np.vectorize(lambda x: x.lower())(X_dev)
    X_test = np.vectorize(lambda x: x.lower())(X_test)

    # Replace emojis
    X_train = np.vectorize(lambda x: demoji.replace_with_desc(x, sep=''))(X_train)
    X_dev = np.vectorize(lambda x: demoji.replace_with_desc(x, sep=''))(X_dev)
    X_test = np.vectorize(lambda x: demoji.replace_with_desc(x, sep=''))(X_test)

    if args.pos_tag:
        # POS tag
        X_train = pos_tag(X_train)
        X_dev = pos_tag(X_dev)
        X_test = pos_tag(X_test)

    # Filter offensive words
    if args.offensive_replacement == 'mark':
        pool = Pool(6)
        X_train = pool.map(mark_offensive_words, X_train)
        X_dev = pool.map(mark_offensive_words, X_dev)
        X_test = pool.map(mark_offensive_words, X_test)
        pool.close()
        pool.join()
    elif args.offensive_replacement == 'remove':
        pool = Pool(6)
        X_train = pool.map(remove_offensive_words, X_train)
        X_dev = pool.map(remove_offensive_words, X_dev)
        X_test = pool.map(remove_offensive_words, X_test)
        pool.close()
        pool.join()

    if args.stem:
        # Stem data
        X_train = stem(X_train)
        X_dev = stem(X_dev)
        X_test = stem(X_test)

    return [list(X_train), list(X_dev), list(X_test)] # CHECK THIS


def preprocess_labels(data,args):
    """Converts the string labels to one-hot-encoded labels or 0-1 labels"""
    [Y_train, Y_dev, Y_test] = data
    if args.model_type == 'ML':
        Y_train = np.array([np.array(1) if x == 'NOT' else np.array(0) for x in Y_train])
        Y_dev = np.array([np.array(1) if x == 'NOT' else np.array(0) for x in Y_dev])
        Y_test = np.array([np.array(1) if x == 'NOT' else np.array(0) for x in Y_test])
    if args.model_type in ['LM','LSTM']:
        Y_train = np.array([np.array([1, 0]) if x == 'NOT' else np.array([0, 1]) for x in Y_train])
        Y_dev = np.array([np.array([1, 0]) if x == 'NOT' else np.array([0, 1]) for x in Y_dev])
        Y_test = np.array([np.array([1, 0]) if x == 'NOT' else np.array([0, 1]) for x in Y_test])
    return [Y_train, Y_dev, Y_test]


def ml_vectorize(data, args):
    [X_train, X_dev, X_test] = data

    # Choose between word level and character level ngrams
    if args.char_ngram:
        analyzer = 'char_wb'
    else:
        analyzer = 'word'

    if args.vectorizer == 'BOW':
        # BOW
        vectorizer = CountVectorizer(
            analyzer=analyzer,
            ngram_range=(args.ngram_min, args.ngram_max), min_df=3)
    elif args.vectorizer == 'TFIDF':
        # TFIDF
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(args.ngram_min, args.ngram_max), min_df=3)
    else:
        raise 'Unknown vectorizer'

    X_train = vectorizer.fit_transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    X_test = vectorizer.transform(X_test)

    if args.most_important_features:
        vocab_inverted = vectorizer.vocabulary_
        vocab = dict()
        for key, value in vocab_inverted.items():
            vocab[value] = key
        os.makedirs(os.path.dirname('features/vocab.pkl'), exist_ok=True)
        with open('features/vocab.pkl', 'wb+') as file:
            pickle.dump(vocab,
                        file, pickle.HIGHEST_PROTOCOL)



    return [X_train, X_dev, X_test]

def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    flag = re.search("txt$", embeddings_file) #check if the input embedding file is txt or not(if not its considered a json file)
    if flag:
      emb = {}
      with open(embeddings_file,'r') as f:
          for line in f:
              values = line.split()
              word = values[0]
              vector = np.asarray(values[1:],'float32')
              emb[word]=vector
    else:
      embeddings = json.load(open(embeddings_file, 'r'))
      emb = {word: np.array(embeddings[word]) for word in embeddings}
    return emb

def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix

def upsample_minor_class(X, Y):
    """Upsample minority class to match the number of the majority. Train and dev sets are treated separately"""
    # Calling DataFrame constructor after zipping
    # X_train and Y_train, with columns specified
    df = pd.DataFrame(list(zip(X, Y)),
                      columns=['tweets', 'tag'])

    # Separate majority and minority classes
    data_majority = df[df['tag'] == 'NOT']
    data_minority = df[df['tag'] == 'OFF']
    print("majority class before upsample:", data_majority.shape)
    print("minority class before upsample:", data_minority.shape)

    # Upsample minority class
    data_minority_upsampled = resample(data_minority,
                                       replace=True,  # sample with replacement
                                       n_samples=data_majority.shape[0])  # to match majority class

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([data_majority, data_minority_upsampled])

    # Display new class counts
    print("Train set after upsampling\n", df_upsampled.tag.value_counts(), sep="")

    # Shuffle upsampled df
    df_upsampled_shuffled = df_upsampled.sample(frac=1).reset_index(drop=True)
    df_upsampled_shuffled.head()

    # Get df columns as lists, to reconstruct the upsampled x and y train
    X_upsampled = df_upsampled_shuffled['tweets'].tolist()
    Y_upsampled = df_upsampled_shuffled['tag'].tolist()

    return [X_upsampled, Y_upsampled]


def lstm_vectorize(data, args):
    """Vectorize using TextVectorization from keras and create embedding matrix"""
    [X_train, X_dev, X_test, embeddings] = data

    # TextVectorization
    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train)# + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings) # for embeddings, check later

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()

    return [X_train_vect, X_dev_vect, X_test_vect, emb_matrix]

def lm_vectorize(data, args):
    """Vectorize using the Huggingface vectorizer"""
    [X_train, X_dev, X_test] = data

    # Bert Vectorizer
    lm = args.language_model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    max_len = args.max_len
    tokens_train = tokenizer(X_train, padding=True, max_length=max_len,
                             truncation=True, return_tensors='np').data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_len,
                           truncation=True, return_tensors='np').data
    tokens_test = tokenizer(X_test, padding=True, max_length=max_len,
                            truncation=True, return_tensors='np').data
    return [tokens_train, tokens_dev, tokens_test]


def vectorize(data, args):
    """Vectorize the data based on the vectorizer"""
    if args.vectorizer == 'LM':
        return lm_vectorize(data, args)
    if args.vectorizer == 'LSTM':
        return lstm_vectorize(data, args)
    if args.vectorizer in ['BOW', 'TFIDF']:
        return ml_vectorize(data, args)
    raise 'Unknown vectorizer'


def count_offensive_words(X, Y, description):
    """Outputs the number of offensive words in offensive and non-offensive tweets respectively"""
    count_off_off = 0
    count_not_off = 0
    count_off = 0
    count_not = 0
    for feature, label in zip(X, Y):
        if label == 'NOT':
            count_not += 1
        else:
            count_off += 1
        if 'OFFOFFOFFOFF' in feature:
            if label == 'NOT':
                count_not_off += 1
            else:
                count_off_off += 1
    print(f'{description} set: {count_not_off} out of {count_not} non-offensive tweets contain offensive words')
    print(f'{description} set: {count_off_off} out of {count_off} offensive tweets contain offensive words')


def main():
    args = create_arg_parser()

    # Read data
    [X_train, Y_train, X_dev, Y_dev, X_test, Y_test] = read_data(args)

    # Save original labels
    Y_train_org = Y_train
    Y_dev_org = Y_dev
    Y_test_org = Y_test

    # Upsample minority class
    if args.upsampling:
        # Upsample training set
        print('Upsample train set:')
        [X_train, Y_train] = upsample_minor_class(X_train, Y_train)
        # Upsample dev set
        print('Upsample dev set:')
        [X_dev, Y_dev] = upsample_minor_class(X_dev, Y_dev)

    # NLP preprocessing
    [X_train, X_dev, X_test] = preprocess([X_train, X_dev, X_test], args)
    # Count offensive words
    if args.offensive_word_count:
        count_offensive_words(X_train, Y_train, 'Training')
        count_offensive_words(X_dev, Y_dev, 'Development')
        count_offensive_words(X_test, Y_test, 'Test')

    # If LSTM is chosen vectorize needs to return an extra arguments which are the embedding matrix and the class weights
    if args.vectorizer == 'LSTM':
        # Read in pre-trained embeddings
        embeddings = read_embeddings(args.embeddings)
        # Vectorize
        [X_train, X_dev, X_test, emb_matrix] = vectorize([X_train, X_dev, X_test, embeddings], args)
        # Compute class weights
        if args.class_weights:
            weights = "balanced"
        else:
            weights = None
        class_weights = class_weight.compute_class_weight(
            class_weight=weights,
            classes=np.unique(Y_train),
            y=Y_train + Y_dev
        )
        class_weight_dict = dict(enumerate(class_weights))
    else:
        [X_train, X_dev, X_test] = vectorize([X_train, X_dev, X_test], args)

    # One-hot encode labels
    [Y_train, Y_dev, Y_test] = preprocess_labels([Y_train, Y_dev, Y_test],args)

    # If LSTM is chosen we need to add the embedding matrix and the class weights dict to pickle file
    if args.model_type == 'LSTM':
        os.makedirs(os.path.dirname(f'features/{args.features_file}.pkl'), exist_ok=True)
        with open(f'features/{args.features_file}.pkl', 'wb+') as file:
            pickle.dump([X_train, Y_train, X_dev, Y_dev, X_test, Y_test, emb_matrix, class_weight_dict],
                        file, pickle.HIGHEST_PROTOCOL)
    else:
        os.makedirs(os.path.dirname(f'features/{args.features_file}.pkl'), exist_ok=True)
        with open(f'features/{args.features_file}.pkl', 'wb+') as file:
            pickle.dump([X_train, Y_train, X_dev, Y_dev, X_test, Y_test],
                        file, pickle.HIGHEST_PROTOCOL)

    # EDA
    # Compute class bias
    majority_class = sum(map(lambda x: x == 'NOT', Y_train_org)) + sum(map(lambda x: x == 'NOT', Y_dev_org))  # num_of majority class instances in train and dev set
    minority_class = sum(map(lambda x: x == 'OFF', Y_train_org)) + sum(map(lambda x: x == 'OFF', Y_dev_org))  # num_of minority class instances in train and dev set
    # Compute class bias
    bias = minority_class / majority_class
    print('Number of instances in the majority class on train set:', sum(map(lambda x: x == 'NOT', Y_train_org)))
    print('Number of instances in the majority class on dev set:', sum(map(lambda x: x == 'NOT', Y_dev_org)))
    print('Number of instances in the minority class on train set:', sum(map(lambda x: x == 'OFF', Y_train_org)))
    print('Number of instances in the minority class on dev set:', sum(map(lambda x: x == 'OFF', Y_dev_org)))
    print('Class bias:',bias)

if __name__ == '__main__':
    main()
