import argparse

"""This script handles the arguments that the user can provide. It returns an args object that 
can be used to obtain the provided arguments."""


def create_arg_parser():
    """Adds arguments to the argument parser and parses them in the end"""
    parser = argparse.ArgumentParser()

    # Input files
    parser.add_argument("--train_file", default='train.tsv', type=str,
                        help="File containing the training features and labels")
    parser.add_argument("--dev_file", default="dev.tsv", type=str,
                        help="File containing the dev features and labels")
    parser.add_argument("--test_file", default='test.tsv', type=str,
                        help="File containing the testing features and labels")
    parser.add_argument("--features_file", default="features", type=str,
                        help="Specify the filename to save the features")
    parser.add_argument("--model_file", default="model", type=str,
                        help="Specify the filename to save the model")
    parser.add_argument("--offensive_word_count", action="store_true",
                        help="Enable printing of the offensive word count. Note that"
                             "this only works when --offensive_replacement=mark")

    # Natural language preprocessing
    parser.add_argument("--remove_emojis", action="store_true",
                        help="Enable to remove emojis")
    parser.add_argument("--pos_tag", action="store_true",
                        help="Enable to apply POS tagging")
    parser.add_argument("--stem", action="store_true",
                        help="Enable to apply stemming")
    parser.add_argument("--offensive_replacement", default=None, type=str,
                        help="Select what to do with offensive words: None, mark, remove")
    parser.add_argument("--ngram_min", default=1, type=int,
                        help="Set the min size of the ngrams")
    parser.add_argument("--ngram_max", default=1, type=int,
                        help="Set the max size of the ngrams")
    parser.add_argument("--char_ngram", action="store_true",
                        help="Enable character level ngrams")
    parser.add_argument("--vectorizer", default="BOW", type=str,
                        help="Define which vectorizer to use: BOW, TFIDF")
    parser.add_argument("--upsampling", action="store_true",
                        help="Enable upsampling of minority class")

    # ML arguments
    parser.add_argument("-alg", "--algorithm", default="NB", type=str,
                        help="Select the classification algorithm to use (NB,KNN,LSVM)")
    parser.add_argument("--random_search", action="store_true",
                        help="Enable random search of hyperparameters")
    parser.add_argument("--grid_search", action="store_true",
                        help="Enable grid search of hyperparameters")
    parser.add_argument("--most_important_features", action="store_true",
                        help="Show the most important features per class")

    # LM arguments
    parser.add_argument("--max_len", default=64, type=int,
                        help="Specify the maximum length of the feature vectors")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Set the batch size used during training.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Set the number of epochs for training.")
    parser.add_argument("--patience", default=5, type=int,
                        help="Set the early stopping patience.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Set the learning rate during training.")
    parser.add_argument("-pd", "--polynomial_decay", default=True, type=bool,
                        help="Enable polynomial decay learning rate.")
    parser.add_argument("--pd_start", default=5e-5, type=float,
                        help="Set the starting learning rate for the polynomial decay.")
    parser.add_argument("--pd_end", default=5e-7, type=float,
                        help="Set the ending learning rate for the polynomial decay.")
    parser.add_argument("--pd_power", default=2., type=float,
                        help="Set the power for the polynomial decay.")
    parser.add_argument("--pd_steps", default=450, type=int,
                        help="Set the number of steps for the polynomial decay.")

    # LSTM arguments
    # batch size, epochs, patience and learning rate are taken from the LM arguments above
    parser.add_argument("-e", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    parser.add_argument("--lstm_units", default=16, type=int,
                        help="Set the number of units for the LSTM layer(s).")
    parser.add_argument("--dense", default=0, type=int,
                        help="Set the number of nodes for the dense layer. If 0, no dense layer "
                             "is used.")
    parser.add_argument("--trainable_embedding", default=False, type=bool,
                        help="Enable or disable a trainable embedding layer.")
    parser.add_argument("--pretrained_embedding", default=True, type=bool,
                        help="Enable or disable a pretrained embedding.")
    parser.add_argument("--lstm_layers", default=1, type=int,
                        help="Set the number of LSTM layers.")
    parser.add_argument("--lstm_dropout", default=0.1, type=float,
                        help="Set the amount of dropout for the LSTM layers.")
    parser.add_argument("--dropout", default=0., type=float,
                        help="Set the amount of normal dropout between the LSTM layers.")
    parser.add_argument("--bidirectional_lstm", default=True, type=bool,
                        help="Set to true for bidirectional LSTM layers.")
    parser.add_argument("--optimizer", default='SGD', type=str,
                        help="Set the optimizer to use.")
    parser.add_argument("--class_weights", default=False, type=bool,
                        help="Enable or disable class weights for training a keras model.")

    # General options
    parser.add_argument("--model_type", default="ML", type=str,
                        help="Select the type of the model")
    parser.add_argument("-lm", "--language_model", default="bert-base-uncased", type=str,
                        help="Select the language model to use from huggingface.co. Possible "
                             "options include bert-base-uncased, xlnet-base-cased, roberta-base.")
    parser.add_argument("-p", "--params", nargs='*', default=[], type=str,
                        help="Provide the parameters for the selected algorithm")
    args = parser.parse_args()
    return args
