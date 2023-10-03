This repository contains the code and files required to run the experiments we performed for our 
research. The three main important scripts are `create_features.py`, `train_model.py` and 
`evaluate_model.py`. For each of the files, refer to `argparser.py` to find the parameters you 
can specify for each of the models. The result files can be found in the folder `results`.

You can install the required dependencies using `pip install -r requirements.txt`.

Since the LSTM model uses a pretrained word embedding, download the following file
`https://nlp.stanford.edu/data/glove.6B.zip` and save it in the main folder of the repo.

`create_features.py` is used to create the features for a specified model using the files `train.
tsv`, `dev.tsv` and `test.tsv`. It stores the features file in a filename provided by the user 
in the folder `features/`.

`train_model.py` trains the specified model on the created features using the previous file. 
After training, this model is saved in the user specified filename in the folder `saved_model`.

`evaluate_model.py` evaluates a model on the dev and test set. The evaluation is printed in the 
console and the predicted labels are written to a file with a filename provided by the user in 
the folder `predicted_labels`.

Now follow the commands that are used to run the models.

LSTM keep
```
python create_features.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-keep --model_file=lstm-keep --lstm_units=16 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=40 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python train_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-keep --model_file=lstm-keep --lstm_units=16 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=40 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python evaluate_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-keep --model_file=lstm-keep --lstm_units=16 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=40 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
```

LSTM replace
```
python create_features.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-replace --model_file=lstm-replace --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python train_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-replace --model_file=lstm-replace --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python evaluate_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-replace --model_file=lstm-replace --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
```

LSTM remove
```
python create_features.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-remove --model_file=lstm-remove --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python train_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-remove --model_file=lstm-remove --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
python evaluate_model.py --vectorizer=LSTM --model_type=LSTM --features_file=lstm-remove --model_file=lstm-remove --lstm_units=128 -pd=False --learning_rate=0.01 --batch_size=16 --epochs=30 -e=glove.6B.200d.txt --class_weights=True --stem --patience=5 --bidirectional_lstm=False
```

Bert keep
```
python create_features.py --vectorizer=LM --model_type=LM --features_file=bert-keep --model_file=bert-keep --max_len=64 --batch_size=16 --epochs=1
python train_model.py --vectorizer=LM --model_type=LM --features_file=bert-keep --model_file=bert-keep --max_len=64 --batch_size=16 --epochs=1
python evaluate_model.py --vectorizer=LM --model_type=LM --features_file=bert-keep --model_file=bert-keep --max_len=64 --batch_size=16 --epochs=1
```

Bert replace
```
python create_features.py --vectorizer=LM --model_type=LM --features_file=bert-replace --model_file=bert-replace --offensive_replacement=mark --max_len=256 --batch_size=8
python train_model.py --vectorizer=LM --model_type=LM --features_file=bert-replace --model_file=bert-replace --offensive_replacement=mark --max_len=256 --batch_size=8
python evaluate_model.py --vectorizer=LM --model_type=LM --features_file=bert-replace --model_file=bert-replace --offensive_replacement=mark --max_len=256 --batch_size=8
```

Bert remove
```
python create_features.py --vectorizer=LM --model_type=LM --features_file=bert-remove --model_file=bert-remove --offensive_replacement=remove --max_len=256 --batch_size=8
python train_model.py --vectorizer=LM --model_type=LM --features_file=bert-remove --model_file=bert-remove --offensive_replacement=remove --max_len=256 --batch_size=8
python evaluate_model.py --vectorizer=LM --model_type=LM --features_file=bert-remove --model_file=bert-remove --offensive_replacement=remove --max_len=256 --batch_size=8
```

RoBERTa keep
```
python create_features.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-keep --model_file=roberta-keep --max_len=256 -pd=False --learning_rate=5e-5 --batch_size=8
python train_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-keep --model_file=roberta-keep --max_len=256 -pd=False --learning_rate=5e-5 --batch_size=8
python evaluate_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-keep --model_file=roberta-keep --max_len=256 -pd=False --learning_rate=5e-5 --batch_size=8
```

RoBERTa replace
```
python create_features.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-replace --model_file=roberta-replace --offensive_replacement=mark --max_len=64 --batch_size=8 --epochs=3
python train_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-replace --model_file=roberta-replace --offensive_replacement=mark --max_len=64 --batch_size=8 --epochs=3
python evaluate_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-replace --model_file=roberta-replace --offensive_replacement=mark --max_len=64 --batch_size=8 --epochs=3
```

RoBERTa remove
```
python create_features.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-remove --model_file=roberta-remove --offensive_replacement=remove --max_len=128 -pd=False --learning_rate=5e-6 --batch_size=8
python train_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-remove --model_file=roberta-remove --offensive_replacement=remove --max_len=128 -pd=False --learning_rate=5e-6 --batch_size=8
python evaluate_model.py --vectorizer=LM --model_type=LM -lm=roberta-base --features_file=roberta-remove --model_file=roberta-remove --offensive_replacement=remove --max_len=128 -pd=False --learning_rate=5e-6 --batch_size=8
```


NB keep
```
python create_features.py -alg "NB" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params alpha=2 fit_prior=True
python train_model.py -alg "NB" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params alpha=2 fit_prior=True
python evaluate_model.py -alg "NB" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params alpha=2 fit_prior=True
```

NB replace
```
python create_features.py -alg "NB" --vectorizer "BOW" --model_type "ML" --offensive_replacement "mark" --params fit_prior=False alpha=3
python train_model.py -alg "NB" --vectorizer "BOW" --model_type "ML" --offensive_replacement "mark" --params fit_prior=False alpha=3
python evaluate_model.py -alg "NB" --vectorizer "BOW" --model_type "ML" --offensive_replacement "mark" --params fit_prior=False alpha=3
```

NB remove
```
python create_features.py -alg "NB" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params alpha=2 fit_prior=False
python train_model.py -alg "NB" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params alpha=2 fit_prior=False
python evaluate_model.py -alg "NB" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params alpha=2 fit_prior=False
```

LSVM keep
```
python create_features.py -alg "LSVM" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params tol=1 max_iter=2000 loss="hinge" intercept_scaling=1.25 fit_intercept=True dual=True class_weights='balanced' C=0.5
python train_model.py -alg "LSVM" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params tol=1 max_iter=2000 loss="hinge" intercept_scaling=1.25 fit_intercept=True dual=True class_weights='balanced' C=0.5
python evaluate_model.py -alg "LSVM" --vectorizer "BOW" --stem --model_type "ML" --offensive_replacement None --params tol=1 max_iter=2000 loss="hinge" intercept_scaling=1.25 fit_intercept=True dual=True class_weights='balanced' C=0.5
```

LSVM replace
```
python create_features.py -alg "LSVM" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "mark" --params tol=1e-05 max_iter=500 loss="hinge" intercept_scaling=1.0 fit_intercept=True dual=True class_weights=None C=0.75
python train_model.py -alg "LSVM" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "mark" --params tol=1e-05 max_iter=500 loss="hinge" intercept_scaling=1.0 fit_intercept=True dual=True class_weights=None C=0.75
python evaluate_model.py -alg "LSVM" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "mark" --params tol=1e-05 max_iter=500 loss="hinge" intercept_scaling=1.0 fit_intercept=True dual=True class_weights=None C=0.75
```

LSVM remove
```
python create_features.py -alg "LSVM" --vectorizer "BOW" --pos_tag --stem --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement "remove" --params tol=0.1 max_iter=5000 loss="squared_hinge" intercept_scaling=1.25 fit_intercept=True dual=False class_weights=None C=1.0
python train_model.py -alg "LSVM" --vectorizer "BOW" --pos_tag --stem --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement "remove" --params tol=0.1 max_iter=5000 loss="squared_hinge" intercept_scaling=1.25 fit_intercept=True dual=False class_weights=None C=1.0
python evaluate_model.py -alg "LSVM" --vectorizer "BOW" --pos_tag --stem --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement "remove" --params tol=0.1 max_iter=5000 loss="squared_hinge" intercept_scaling=1.25 fit_intercept=True dual=False class_weights=None C=1.0
```

KNN keep
```
python create_features.py -alg "KNN" --vectorizer "BOW" --stem --pos_tag --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement None --params weights="distance" p=1 n_neighbors=1 metric="cosine" leaf_size=10 algorithm="brute"
python train_model.py -alg "KNN" --vectorizer "BOW" --stem --pos_tag --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement None --params weights="distance" p=1 n_neighbors=1 metric="cosine" leaf_size=10 algorithm="brute"
python evaluate_model.py -alg "KNN" --vectorizer "BOW" --stem --pos_tag --ngram_min 1 --ngram_max 3 --model_type "ML" --offensive_replacement None --params weights="distance" p=1 n_neighbors=1 metric="cosine" leaf_size=10 algorithm="brute"
```

KNN replace
```
python create_features.py -alg "KNN" --vectorizer "BOW" --stem --ngram_min 1 --ngram_max 2 --model_type "ML" --offensive_replacement "mark" --params weights="uniform" p=2 n_neighbors=6 metric"minkowski" leaf_size=30 algorithm="auto"
python train_model.py -alg "KNN" --vectorizer "BOW" --stem --ngram_min 1 --ngram_max 2 --model_type "ML" --offensive_replacement "mark" --params weights="uniform" p=2 n_neighbors=6 metric"minkowski" leaf_size=30 algorithm="auto"
python evaluate_model.py -alg "KNN" --vectorizer "BOW" --stem --ngram_min 1 --ngram_max 2 --model_type "ML" --offensive_replacement "mark" --params weights="uniform" p=2 n_neighbors=6 metric"minkowski" leaf_size=30 algorithm="auto"
```

KNN remove
```
python create_features.py -alg "KNN" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params weights="uniform" p=1 n_neighbors=8 metric="cosine" leaf_size=20 algorithm="brute"
python train_model.py -alg "KNN" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params weights="uniform" p=1 n_neighbors=8 metric="cosine" leaf_size=20 algorithm="brute"
python evaluate_model.py -alg "KNN" --vectorizer "BOW" --pos_tag --model_type "ML" --offensive_replacement "remove" --params weights="uniform" p=1 n_neighbors=8 metric="cosine" leaf_size=20 algorithm="brute"
```
