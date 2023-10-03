import pickle
import tensorflow as tf
from sklearn.utils import class_weight
import random as python_random
import numpy as np
from argparser import create_arg_parser
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam,SGD, Adadelta
from keras.layers import Embedding, LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.initializers import Constant
from transformers import TFAutoModelForSequenceClassification
from abc import ABC, abstractmethod
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit
import numpy as np
from scipy import sparse
import random as python_random

"""This scripts is used to create a model and train this model with the features created using 
create_feature.py. Refer to argparser.py for the possible arguments."""

# Make reproducible as much as possible
np.random.seed(42)
tf.random.set_seed(42)
python_random.seed(42)


class Algorithm(ABC):
    """Abstract base class for the machine learning algorithms"""

    def __init__(self):
        self.params = {}

    def set_params(self, params):
        """updates the parameters based on the received list"""
        for param in params:
            # parameters are in the form: key=value
            key = param.split('=')[0]
            value = param.split('=')[1]
            # Convert to None or booleans
            if value == 'None':
                value = None
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            else:
                if is_integer(value):
                    # value is an integer
                    value = int(value)
                else:
                    # Try convert to float
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            self.params[key] = value

    @abstractmethod
    def get_algorithm(self):
        """Gets the final algorithm"""
        pass


class NB(Algorithm):
    """Naive Bayes"""

    def __init__(self):
        super().__init__()
        self.params = {
            'alpha': 0.5,
            'fit_prior': True
        }

    def get_algorithm(self):
        return MultinomialNB(alpha=self.params.get('alpha'),
                             fit_prior=self.params.get('fit_prior'))


class KNN(Algorithm):
    """K Nearest Neighbor"""

    def __init__(self):
        super().__init__()
        self.params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski'
        }

    def get_algorithm(self):
        return KNeighborsClassifier(n_neighbors=self.params.get('n_neighbors'),
                                    weights=self.params.get('weights'),
                                    algorithm=self.params.get('algorithm'),
                                    leaf_size=self.params.get('leaf_size'),
                                    p=self.params.get('p'),
                                    metric=self.params.get('metric'), )


class LSVM(Algorithm):
    """Linear Support Vector Machine"""

    def __init__(self):
        super().__init__()
        self.params = {
            'loss': 'hinge',
            'dual': True,
            'tol': 0.01,
            'C': 0.75,
            'multi_class': 'ovr',
            'fit_intercept': True,
            'intercept_scaling': 1.0,
            'max_iter': 1000
        }

    def get_algorithm(self):
        return LinearSVC(loss=self.params.get('loss'),
                         dual=self.params.get('dual'),
                         tol=self.params.get('tol'),
                         C=self.params.get('C'),
                         multi_class=self.params.get('multi_class'),
                         fit_intercept=self.params.get('fit_intercept'),
                         intercept_scaling=self.params.get('intercept_scaling'),
                         max_iter=self.params.get('max_iter'))


def is_integer(value):
    """Checks if the value is an integer"""
    if value[0] == '-' or value[0] == '+':
        return value[1:].isdigit()
    else:
        return value.isdigit()



def parse_algorithm(algorithm_arg):
    """Converts string argument to algorithm"""
    algorithms = {
        'NB': NB,
        'KNN': KNN,
        'LSVM': LSVM
    }
    return algorithms.get(algorithm_arg, 'Classifier not found')()


def create_ml_model(args):
    """Creates the ml model"""
    algorithm = parse_algorithm(args.algorithm)
    algorithm.set_params(args.params)
    classifier = algorithm.get_algorithm()
    return classifier


def create_lstm_model(args, Y_train,emb_matrix):
    """Creates the lstm neural network"""
    # Define settings
    loss_function = 'categorical_crossentropy'

    # Get the user selected optimizer
    if args.optimizer == 'SGD':
        optim = SGD(learning_rate=args.learning_rate)
    elif args.optimizer == 'Adam':
        optim = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'Adadelta':
        optim = Adadelta(learning_rate=args.learning_rate)
    else:
        raise 'Optimizer not found!'
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = 2 #len(set(Y_train)) # throws error so we hardcoded 2
    # Now build the model
    model = Sequential()

    # Check whether to use a pretrained embedding
    if args.pretrained_embedding:
        embedding = Constant(emb_matrix)
    else:
        embedding = "uniform"

    # Create the embedding layer
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=embedding,
                        trainable=args.trainable_embedding))

    # Add an optional dense layer
    if args.dense != 0:
        model.add(Dense(args.dense))

    # Add (stacked) LSTM layer(s), possibly with LSTM dropout
    for _ in range(args.lstm_layers - 1):
        model.add(LSTM(units=args.lstm_units, return_sequences=True, dropout=args.lstm_dropout,
                       go_backwards=args.bidirectional_lstm))
        # Add normal dropout layer in between
        if args.dropout > 0.:
            model.add(Dropout(args.dropout))
    if args.lstm_layers > 0:
        model.add(LSTM(units=args.lstm_units, return_sequences=False, dropout=args.lstm_dropout,
                       go_backwards=args.bidirectional_lstm))

    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def create_lm_model(args):
    """Creates the language model"""
    # Set the learning schedule or learning rate
    if args.polynomial_decay:
        learning_rate = PolynomialDecay(
            args.pd_start,
            args.pd_steps,
            end_learning_rate=args.pd_end,
            power=args.pd_power,
            cycle=True,
            name=None
        )
    else:
        learning_rate = args.learning_rate

    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)

    lm = args.language_model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def create_model(args, Y_train, emb_matrix):
    # Create a model based on the model type
    if args.model_type == 'ML':
        return create_ml_model(args)
    if args.model_type == 'LSTM':
        return create_lstm_model(args, Y_train, emb_matrix)
    if args.model_type == 'LM':
        return create_lm_model(args)


def train_ml_model(model, X_train, Y_train, X_dev, Y_dev, args):
    if args.random_search or args.grid_search:
        # Perform a random or grid search of the hyperparameters
        # Define distributions hyperparameter sets for each algorithm
        if args.algorithm == 'NB':
            distributions = dict(
                alpha=[0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                fit_prior=[False, True]
            )
        elif args.algorithm == 'KNN':
            distributions = dict(
                n_neighbors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30],
                weights=['uniform', 'distance'],
                algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                leaf_size=[10,20,30,50],
                p=[1, 2],
                metric=['minkowski','mahalanobis','cosine']
            )
        elif args.algorithm == 'LSVM':
            distributions = dict(
                loss=['hinge', 'squared_hinge'],
                dual=[False, True],
                tol=[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                C=[0.5, 0.75, 1.0, 1.25, 1.5],
                fit_intercept=[False, True],
                intercept_scaling=[0.5, 0.75, 1.0, 1.25, 1.5],
                class_weight=['balanced', None],
                random_state=[42],
                max_iter=[500, 1000, 2000, 5000]
            )

        n_iter = 800

        #in order to train on training set, and validate on development set
        split_index = [-1] * Y_train.size + [0] * Y_dev.size
        X = sparse.vstack((X_train, X_dev))
        Y = np.concatenate((Y_train, Y_dev),axis=0)
        pds = PredefinedSplit(test_fold=split_index)

        if args.random_search:
            search = RandomizedSearchCV(model, distributions, random_state=42,
                                        n_iter=n_iter, scoring='f1_macro', verbose=1, n_jobs=-1, cv=pds)
        else:
            search = GridSearchCV(model, distributions, scoring='f1_macro',
                                  verbose=1, n_jobs=-1, cv=pds)


        # Run the search
        search_results = search.fit(X, Y)

        # Show the results
        print('Best hyperparameters:', search.best_params_)
        print('\n')
        print('f-1 score of best hyperparameters on dev set:', search.best_score_)
        print('\n')


        #add the best found parameters to the arguments
        for param in search.best_params_:
            args.params.append( str(param)+'='+str(search.best_params_[param]))

        #create model with the best parameters found from search
        model= create_ml_model(args)
        model.fit(X_train, Y_train)
    else:
        model.fit(X_train,Y_train)

    if args.most_important_features:
        # Get the best features per class
        with open('features/vocab.pkl', 'rb') as file:
            vocab = pickle.load(file)

        # Get the 10 most important features for this class
        indices = reversed([i[0] for i in sorted(enumerate(model.coef_[0]),
                                                 key=lambda x: x[1])][-10:])
        # Convert the indices to the words
        best_features = []
        for idx in indices:
            best_features.append(vocab[idx])

        # Provide Latex output of the most important features
        print('most important features:')
        for idx in range(10):
            line = ''
            line += f'{best_features[idx]}  '
            print(line)

    return model



def train_lstm_model(model, X_train, Y_train, X_dev, Y_dev, class_weight_dict, args):
    '''Train the lstm model here'''
    verbose = 1
    batch_size = args.batch_size
    epochs = args.epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
    # Finally fit the model to our data
    # Enable/disable class weights
    if args.class_weights:
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback],
                  batch_size=batch_size, class_weight=class_weight_dict, validation_data=(X_dev, Y_dev))
    else:
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback],
                  batch_size=batch_size, validation_data=(X_dev, Y_dev))
    return model

def train_lm_model(model, X_train, Y_train, X_dev, Y_dev, args):
    """Train a language model based on the specified settings by the user"""
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = args.batch_size
    epochs = args.epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback],
              batch_size=batch_size, validation_data=(X_dev, Y_dev))
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, class_weight_dict, args):
    """Train a model based on the model type"""
    if args.model_type == 'ML':
        return train_ml_model(model, X_train, Y_train, X_dev, Y_dev, args)
    if args.model_type == 'LSTM':
        return train_lstm_model(model, X_train, Y_train, X_dev, Y_dev, class_weight_dict, args)
    if args.model_type == 'LM':
        return train_lm_model(model, X_train, Y_train, X_dev, Y_dev, args)


def save_model(model, args):
    """Save the trained model in the corresponding file"""
    if args.model_type == 'ML':
        pickle.dump(model, open(f'{args.model_file}.sav', 'wb'))
    if args.model_type == 'LSTM':
        model.save(f'saved_model/{args.model_file}')
    if args.model_type == 'LM':
        model.save_pretrained(f'saved_model/{args.model_file}')


def main():
    args = create_arg_parser()

    # Load pickle directory with the serialized dataset
    # If LSTM is chosen we need to load the embedding matrix and class weights as well
    emb_matrix = None
    class_weight_dict = None
    if args.model_type == 'LSTM':
        with open(f'features/{args.features_file}.pkl', 'rb') as file:
            [X_train, Y_train, X_dev, Y_dev, X_test, Y_test, emb_matrix, class_weight_dict] = pickle.load(file)
    else:
        with open(f'features/{args.features_file}.pkl', 'rb') as file:
            [X_train, Y_train, X_dev, Y_dev, X_test, Y_test] = pickle.load(file)

    # Initialize model
    model = create_model(args, Y_train, emb_matrix)
    model = train_model(model, X_train, Y_train, X_dev, Y_dev, class_weight_dict, args)

    # Save model
    save_model(model, args)


if __name__ == '__main__':
    main()
