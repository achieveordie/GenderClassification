"""
Predictor code for old-GBM based classification, uses sources present in `v1` directory.
One can directly pass name(s) in the command line or wait for the program to prompt.
"""

import pickle
import argparse
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from timeit import default_timer as timer


def predict(X):
    print("Received: ", X)
    if isinstance(X, tuple):
        X = vectorizer.fit_transform(X)
    else:
        X = vectorizer.fit_transform([X])
    X = scaler.fit_transform(X)
    predictions = loaded_model.predict_proba(X)
    if isinstance(predictions, list):
        [print(i) for i in predictions]
    else:
        print(predictions)


if __name__ == '__main__':
    start = timer()
    parser = argparse.ArgumentParser(description="The central code to predict the gender(s) based on name(s)")
    parser.add_argument('names', metavar='-n', nargs='*', help='Enter name(s) if not then input when asked.')

    dict_saved_location = r'../v1/feature.pkl'
    model_location = r'../v1/genderClassificationV4.sav'
    loaded_model = pickle.load(open(model_location, 'rb'))

    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 7),
                                 vocabulary=pickle.load(open(dict_saved_location, 'rb')))
    scaler = MaxAbsScaler()
    args = parser.parse_args()
    if len(sys.argv) > 2:
        names = tuple(args.names)
    elif len(sys.argv) == 2:
        names = str(args.names[0])
    else:
        names = input("No command-line argument found for names, enter single name or space separated list of names\n")
        names = names.split(' ')
        if len(names) == 1:  # single element list should be converted back to a string
            names = names[0]

    predict(names)
    print("Total Time->", timer() - start)
