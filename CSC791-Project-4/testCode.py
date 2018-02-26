import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])
method_names = ["traditional NLP", "Doc2Vec"]



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True, title="Normalized Confusion Matrix of Naive Bayes Classifier by "+method_names[method])
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True, title="Normalized Confusion Matrix of Logistic Regression Classifier by "+method_names[method])


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = []
    for index, words in enumerate(train_pos):
        sentence = LabeledSentence(words, ["TRAIN_POS_%s"%index])
        labeled_train_pos.append(sentence)
    labeled_train_neg = []
    for index, words in enumerate(train_neg):
        sentence = LabeledSentence(words, ["TRAIN_NEG_%s"%index])
        labeled_train_neg.append(sentence)
    labeled_test_pos = []
    for index, words in enumerate(test_pos):
        sentence = LabeledSentence(words, ["TEST_POS_%s"%index])
        labeled_test_pos.append(sentence)
    labeled_test_neg = []
    for index, words in enumerate(test_neg):
        sentence = LabeledSentence(words, ["TEST_NEG_%s"%index])
        labeled_test_neg.append(sentence)
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [], [], [], []
    for index in range(len(labeled_train_pos)):
        doc_vec = model.docvecs["TRAIN_POS_%s"%index]
        train_pos_vec.append(doc_vec)
    for index in range(len(labeled_train_neg)):
        doc_vec = model.docvecs["TRAIN_NEG_%s"%index]
        train_neg_vec.append(doc_vec)
    for index in range(len(labeled_test_pos)):
        doc_vec = model.docvecs["TEST_POS_%s"%index]
        test_pos_vec.append(doc_vec)
    for index in range(len(labeled_test_neg)):
        doc_vec = model.docvecs["TEST_NEG_%s"%index]
        test_neg_vec.append(doc_vec)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X, Y)
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    return nb_model, lr_model



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    posCounts = collections.Counter([word for term in train_pos for word in term if word not in stopwords]) # condition (1)
    negCounts = collections.Counter([word for term in train_neg for word in term if word not in stopwords]) # condition (1)
    posTexSize = len(posCounts.keys())
    negTexSize = len(negCounts.keys())
    feature_words = []
    for word in set(posCounts.keys()+negCounts.keys()):
        if (posCounts[word]!=0) and (negCounts[word]!=0):
            rate1 = posCounts[word] / float(negCounts[word]) 
            rate2 = negCounts[word] / float(posCounts[word])
            occupation1 = posCounts[word] / float(posTexSize)
            occupation2 = negCounts[word] / float(negTexSize)
            if (rate1>=2 or rate2>=2) and (occupation1>=0.01 or occupation2>=0.01): # condition (2)(3)
                feature_words.append(word)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = [[1 if word in term else 0 for word in feature_words] for term in train_pos]
    train_neg_vec = [[1 if word in term else 0 for word in feature_words] for term in train_neg]
    test_pos_vec = [[1 if word in term else 0 for word in feature_words] for term in test_pos]
    test_neg_vec = [[1 if word in term else 0 for word in feature_words] for term in test_neg]
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec + train_neg_vec # predictor
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X, Y)
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X, Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False, title="Feature Extraction Method"):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    test_pos_predict = model.predict(test_pos_vec) # predict pos
    test_neg_predict = model.predict(test_neg_vec) # predict neg
    test_pos_Y = ["pos"]*len(test_pos_vec) # actual pos
    test_neg_Y = ["neg"]*len(test_neg_vec) # actual neg
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(test_pos_predict)):
        if test_pos_predict[i] == test_pos_Y[i]:
            tp += 1
        else:
            fn += 1
    for i in range(len(test_neg_predict)):
        if test_neg_predict[i] == test_neg_Y[i]:
            tn += 1
        else:
            fp += 1
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
        y_test = test_pos_Y 
        y_test.extend(test_neg_Y)
        y_pred = test_pos_predict.tolist()
        y_pred.extend(test_neg_predict.tolist())
        cnf_matrix = confusion_matrix(y_test, y_pred)
        class_names = ['positive', 'negative']
        plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    print "accuracy: %f" % (accuracy)


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.png')


if __name__ == "__main__":
    main()