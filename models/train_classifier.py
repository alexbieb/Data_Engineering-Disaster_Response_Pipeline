# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages',con=engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = list(Y.columns.values)
    return X, Y, category_names

# tokenization function to process the text data
def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    # machine pipeline should take in the message column as input and output classification results on the other 36 categories in the dataset.
    pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])
    
    parameters = {
    #        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    #        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    #        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [5,10,30],
        'clf__estimator__min_samples_split': [2, 3, 4],
    #        'features__transformer_weights':
    #            {'text_pipeline': 1, 'starting_verb': 0.5},
    #            {'text_pipeline': 0.5, 'starting_verb': 1},
    #            {'text_pipeline': 0.8, 'starting_verb': 1},
            }
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for column in range(0,Y_test.shape[1]):     
        print(classification_report(Y_test.iloc[:,column],Y_pred[:,column]))
    
def save_model(model, model_filepath):
    import pickle
    outfile = open(model_filepath,'wb')
    pickle.dump(model,outfile)
    outfile.close()
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()