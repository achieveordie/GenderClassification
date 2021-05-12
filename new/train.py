"""
A lot of code is from `architecture.ipynb`, modularized for better readability.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from VotingClassifier import VotingClassifier
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
from timeit import default_timer as timer


def get_params():
    gbc_params = {
        "learning_rate": 0.85,
        "max_features": 0.05,
        "max_depth": 10,
        "min_samples_split": 520,
        "min_samples_leaf": 15,
        "n_estimators": 400
    }

    abc_params = {
        "algorithm": "SAMME.R",
        "learning_rate": 0.90,
        "n_estimators": 500
    }

    lsvc_params = {"C": 0.1}

    return gbc_params, abc_params, lsvc_params


def get_main_accuracy(clf, train_features, val_features, y, y_val):
    """
    Calling `score()` method of VotingClassifier.VotingClassifier() instance will only give an average
    of base-estimators, so instead call predict() for a better performance estimate.

    Not the most efficient, can further be improved by converting predictions and true data into pandas for analysis.
    :param clf: <VotingClassifer.VotingClassifier()> instance with all base estimators already fitted.
    :param train_features: <List<array-like>> Features to get prediction from trained data
    :param val_features: <List<array-like>> Features to get prediction from validation data
    :param y: True Labels for training
    :param y_val: True labels for validation
    :return: <Tuple(train_acc, val_acc)> where both elements are floats.
    """
    train_predict = clf.predict(train_features)
    val_predict = clf.predict(val_features)

    train_acc, val_acc = 0, 0
    for (true, train_pred) in zip(y, train_predict):
        if true == train_pred:
            train_acc += 1
    for (true, val_pred) in zip(y_val, val_predict):
        if true == val_pred:
            val_acc += 1

    return train_acc / len(y), val_acc / len(y_val)


if __name__ == "__main__":
    # get data from csv
    start = timer()
    print("Collecting data and cleaning data..")
    base_location = Path(__file__).resolve().parents[2]
    data_location = base_location / r"data/gender_final.csv"
    X = pd.read_csv(str(data_location))
    y = X["Gender"]
    X.drop(labels="Gender", inplace=True, axis=1)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.05, random_state=11)
    print("End of step 1, time taken: ", timer() - start, '\n')

    # preprocessing on train data and transform to test data
    start = timer()
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 10), decode_error='replace', binary=True)
    scaler = MaxAbsScaler()
    X = vectorizer.fit_transform(X['Name'])
    X = scaler.fit_transform(X)

    X_val = vectorizer.transform(X_val['Name'])
    X_val = scaler.transform(X_val)
    print("Done, training and validation shape is {}, {}".format(X.shape, X_val.shape))
    print("End of step 2, time taken: ", timer() - start, '\n')

    # get all parameters for base classifiers:
    start = timer()
    gbc_params, abc_params, lsvc_params = get_params()
    print("End of step 3, time taken: ", timer() - start, '\n')

    # Initializing estimators using above parameters:
    start = timer()
    gbc_clf = GradientBoostingClassifier(learning_rate=gbc_params["learning_rate"],
                                         max_features=gbc_params["max_features"],
                                         min_samples_split=gbc_params["min_samples_split"],
                                         n_estimators=gbc_params["n_estimators"],
                                         max_depth=gbc_params["max_depth"],
                                         min_samples_leaf=gbc_params["min_samples_leaf"],
                                         random_state=0, verbose=1)

    abc_clf = AdaBoostClassifier(algorithm=abc_params["algorithm"],
                                 n_estimators=abc_params["n_estimators"],
                                 learning_rate=abc_params["learning_rate"],
                                 random_state=0)

    lsvc_clf = LinearSVC(C=lsvc_params['C'], verbose=1)

    X_selected_main = []
    X_val_selected_main = []
    selector_params = []  # To store dicts from `get_params()` method of each selector below.
    print("End of step 4, time taken: ", timer() - start, '\n')

    # fit and select apt. features for each estimator then refit on them.
    # GBC
    start = timer()
    print("Fitting GradientBoost Classifier:")
    gbc_clf.fit(X, y)
    print("Score using all features:Training ", gbc_clf.score(X, y))
    print("Score using all features:Validation ", gbc_clf.score(X_val, y_val))

    selector = SelectFromModel(gbc_clf, prefit=True)
    selector_params.append(selector.get_params())
    X_selected = selector.transform(X)
    X_selected_main.append(X_selected)
    X_val_selected = selector.transform(X_val)
    X_val_selected_main.append(X_val_selected)
    print("Shaped reduced from {} to {}, difference is {}".format(X.shape[1],
                                                                  X_selected.shape[1],
                                                                  X.shape[1] - X_selected.shape[1]))
    print("Refitting using selected features.")
    gbc_clf.fit(X_selected, y)
    print("Score using selected features:Training ", gbc_clf.score(X_selected, y))
    print("Score using selected features:Validation ", gbc_clf.score(X_val_selected, y_val))

    # ABC
    print("Fitting AdaBoost Classifier: ")
    abc_clf.fit(X, y)
    print("Score using all features:Training ", abc_clf.score(X, y))
    print("Score using all features:Validation ", abc_clf.score(X_val, y_val))

    selector = SelectFromModel(abc_clf, prefit=True)
    selector_params.append(selector.get_params())
    X_selected = selector.transform(X)
    X_selected_main.append(X_selected)
    X_val_selected = selector.transform(X_val)
    X_val_selected_main.append(X_val_selected)
    print("Shaped reduced from {} to {}, difference is {}".format(X.shape[1],
                                                                  X_selected.shape[1],
                                                                  X.shape[1] - X_selected.shape[1]))
    print("Refitting using selected features.")
    abc_clf.fit(X_selected, y)
    print("Score using selected features:Training ", abc_clf.score(X_selected, y))
    print("Score using selected features:Validation ", abc_clf.score(X_val_selected, y_val))

    # LSVC
    print("Fitting LinearSV Classifier:")

    lsvc_clf.fit(X, y)
    print("Score using all features:Training ", lsvc_clf.score(X, y))
    print("Score using all features:Validation ", lsvc_clf.score(X_val, y_val))

    selector = SelectFromModel(lsvc_clf, prefit=True)
    selector_params.append(selector.get_params())
    X_selected = selector.transform(X)
    X_selected_main.append(X_selected)
    X_val_selected = selector.transform(X_val)
    X_val_selected_main.append(X_val_selected)
    print("Shaped reduced from {} to {}, difference is {}".format(X.shape[1],
                                                                  X_selected.shape[1],
                                                                  X.shape[1] - X_selected.shape[1]))
    print("Refitting using selected features.")
    lsvc_clf.fit(X_selected, y)
    print("Score using selected features:Training ", lsvc_clf.score(X_selected, y))
    print("Score using selected features:Validation ", lsvc_clf.score(X_val_selected, y_val))
    print("End of step 5, time taken: ", timer() - start, '\n')

    # Main classifier made of custom VotingClassifier, orders of clfs same as list of selected features above.
    start = timer()
    main_clf = VotingClassifier(clfs=[gbc_clf, abc_clf, lsvc_clf], voting='hard')
    main_clf.fit(X_selected_main, y)  # base estimators are already fitted so nothing happens, good measure to check

    train_acc, val_acc = get_main_accuracy(main_clf, X_selected_main, X_val_selected_main, y, y_val)
    print(f"Total training accuracy: {round(train_acc, 3)*100.0}")
    print(f"Total validation accuracy: {round(val_acc, 3)*100.0}")

    # Save vectorizer's vocab, selector params and individual base-estimators.
    print("Saving vocab, selector params & base-estimators..")
    vocab_location = str(base_location / r"model/vectorizer_vocab.pkl")
    gbc_location = str(base_location / r"model/gbc_model_v1.sav")
    abc_location = str(base_location / r"model/abc_model_v1.sav")
    lsvc_location = str(base_location / r"model/lsvc_model_v1.sav")
    selector_location = str(base_location / r"model/selector_v1.pkl")

    pickle.dump(vectorizer.vocabulary_, open(vocab_location, 'wb'))
    pickle.dump(gbc_clf, open(gbc_location, 'wb'))
    pickle.dump(abc_clf, open(abc_location, 'wb'))
    pickle.dump(lsvc_clf, open(lsvc_location, 'wb'))
    pickle.dump(selector_params, open(selector_location, 'wb'))
    print(f"All configurations/models are saved in {str(base_location/'model')}")
    print("End of step 6, time taken: ", timer() - start)
