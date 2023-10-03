import os
import pickle
import numpy as np
import random as python_random
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from argparser import create_arg_parser
from transformers import TFAutoModelForSequenceClassification

"""This script is used for evaluating a model using the features created with create_features.py 
and the model created and trained using train_model.py. Refer to argparser.py for the possible 
arguments."""


def print_performance_scores(Y_true, Y_pred):
    """Print f-score, precision, recall and accuracy for the predicted labels"""
    print('f1-score & precision & recall & accuracy \\\\ \\hline')
    print(f'{round(f1_score(Y_true, Y_pred, average="macro"), 3)} & '
          f'{round(precision_score(Y_true, Y_pred, average="macro"), 3)} & '
          f'{round(recall_score(Y_true, Y_pred, average="macro"), 3)} & '
          f'{round(accuracy_score(Y_true, Y_pred), 3)} \\\\')


def predict_labels(model, X,args):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    if args.model_type == 'ML':
        Y_pred = model.predict(X)
    if args.model_type == 'LSTM':
        #may need logits r one-hot vector?
        Y_pred = model.predict(X)
    if args.model_type == 'LM':
        Y_pred = model.predict(X)["logits"]
        # Finally, convert to numerical labels to get scores with sklearn
        return np.argmax(Y_pred, axis=1)
    return Y_pred


def save_labels(Y_pred, filename):
    os.makedirs(os.path.dirname(f'predicted_labels/{filename}'), exist_ok=True)
    with open(f'predicted_labels/{filename}', 'w+') as file:
        for label in Y_pred:
            file.write(str(label) + '\n')


def evaluate_model(model, X_dev, Y_dev, X_test, Y_test, args):
    """"Evaluate the model based on the dev and test sets"""
    # Evaluate LM
    if args.vectorizer == 'LM':
        # Dev set predictions
        Y_pred = predict_labels(model, X_dev, args)
        save_labels(Y_pred, args.model_file + '_dev.txt')
        print_performance_scores(np.argmax(Y_dev, axis=1), Y_pred)
        # Test set predictions
        Y_pred = predict_labels(model, X_test, args)
        save_labels(Y_pred, args.model_file + '_test.txt')
        print_performance_scores(np.argmax(Y_test, axis=1), Y_pred)

    # Evaluate LSTM
    if args.model_type == 'LSTM':
        # Get dev predictions using the trained model
        Y_pred = model.predict(X_dev)
        # Convert to numerical labels to get scores with sklearn
        Y_pred = np.argmax(Y_pred, axis=1)
        # Print scores
        print("Dev scores:")
        print_performance_scores(np.argmax(Y_dev, axis=1), Y_pred)

        # Get test predictions using the trained model
        Y_pred = model.predict(X_test)
        # Convert to numerical labels to get scores with sklearn
        Y_pred = np.argmax(Y_pred, axis=1)
        # Print scores
        print("Test scores:")
        print_performance_scores(np.argmax(Y_test, axis=1), Y_pred)

    # Evaluate ML model
    if args.vectorizer in ['BOW', 'TFIDF']:
        Y_pred = predict_labels(model, X_dev, args)
        print('Validation scores')
        print_performance_scores(Y_dev, Y_pred)
        Y_pred = predict_labels(model, X_test, args)
        print('Test scores')
        print_performance_scores(Y_test, Y_pred)





def main():
    args = create_arg_parser()

    # Load dataset from pickle directory
    # If LSTM is chosen we need to load the embedding matrix as well
    if args.model_type == 'LSTM':
        with open(f'features/{args.features_file}.pkl', 'rb') as file:
            [X_train, Y_train, X_dev, Y_dev, X_test, Y_test, emb_matrix, dict_weights_class] = pickle.load(file)
    else:
        with open(f'features/{args.features_file}.pkl', 'rb') as file:
            [X_train, Y_train, X_dev, Y_dev, X_test, Y_test] = pickle.load(file)

    # Load trained model
    if args.model_type == 'ML':
        model = pickle.load(open(f'{args.model_file}.sav', 'rb'))
    if args.model_type == 'LM':
        model = TFAutoModelForSequenceClassification.from_pretrained(f'saved_model/{args.model_file}', num_labels=2)
    if args.model_type == 'LSTM':
        model = keras.models.load_model(f'saved_model/{args.model_file}')

    # Evaluate model
    evaluate_model(model, X_dev, Y_dev, X_test, Y_test, args)


if __name__ == '__main__':
    main()
