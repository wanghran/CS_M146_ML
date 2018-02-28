"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        n = sum(y == 1)
        N = len(y)
        self.probabilities_ = [1-n/float(N), n/float(N)]
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        y = np.random.choice([0, 1], len(X), p=self.probabilities_)
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0.0
    test_error = 0.0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
        train_error += 1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True)
        test_error += 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)
    train_error = train_error/100.0
    test_error = test_error/100.0
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    rdf = RandomClassifier()
    rdf.fit(X,y)
    rd_y_pred = rdf.predict(X)
    rd_train_error = 1 - metrics.accuracy_score(y, rd_y_pred, normalize=True)
    print('\t-- training error: %.3f' % rd_train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    dtf = DecisionTreeClassifier(criterion="entropy")
    dtf.fit(X,y)
    dt_y_pred = dtf.predict(X)
    dt_train_error = 1 - metrics.accuracy_score(y, dt_y_pred, normalize=True)
    print('\t-- training error: %.3f' % dt_train_error)
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    for i in range(3):
        knf = KNeighborsClassifier(n_neighbors=3+2*i)
        knf.fit(X,y)
        kn_y_pred = knf.predict(X)
        kn_error = 1 - metrics.accuracy_score(y, kn_y_pred, normalize=True)
        print('\t-- training error for : %.3f' % kn_error)

    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    M_Er = error(MajorityVoteClassifier(), X, y)
    print('train error for Majority Classifier is %.3f' % M_Er[0])
    print('test error for Majority Classifier is %.3f' % M_Er[1])
    R_Er = error(RandomClassifier(), X, y)
    print('train error for Random Classifier is %.3f' % R_Er[0])
    print('test error for Random Classifier is %.3f' % R_Er[1])
    DT_Er = error(DecisionTreeClassifier(criterion="entropy"), X, y)
    print('train error for DT Classifier is %.3f' % DT_Er[0])
    print('test error for DT Classifier is %.3f' % DT_Er[1])
    kN_Er = error(KNeighborsClassifier(n_neighbors=5), X, y)
    print('train error for 5N Classifier is %.3f' % kN_Er[0])
    print('test error for 5N Classifier is %.3f' % kN_Er[1])

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    k = []
    score = []
    for i in range(25):
        k.append(2*i+1)
    for j in range(25):
        avg = 0.0
        E = cross_val_score(KNeighborsClassifier(n_neighbors=k[j]), X, y, cv=10)
        for n in range(len(E)):
            avg += E[n]
        avg = avg / len(E)
        score.append(avg)
    plt.title('cross validation score for different k')
    plt.xlabel('k')
    plt.ylabel('score')
    tic = range(0, 51, 5)
    plt.xticks(tic)
    plt.plot(k, score, 'o')
    plt.savefig('cross validation score for different k.png', dpi=300)
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depth = []
    DT_t_error = []
    DT_T_error = []
    for i in range(20):
        depth.append(i+1)
        DT_t_error.append(error(DecisionTreeClassifier(criterion="entropy", max_depth=(i+1)), X, y)[0])
        DT_T_error.append(error(DecisionTreeClassifier(criterion="entropy", max_depth=(i+1)), X, y)[1])
    plt.title('training errors and test errors for different depth')
    plt.xlabel('depth')
    plt.ylabel('error')
    tick = range(1, 21)
    plt.xticks(tick)
    plt.plot(depth, DT_t_error, 'o', label="train error")
    plt.plot(depth, DT_T_error, 'x', label="test error")
    plt.legend()
    plt.savefig('training errors and test errors for different depth.png', dpi=300)
    plt.show()

    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    train_split = []
    DT_train_error = []
    DT_test_error = []
    KN_train_error = []
    KN_test_error = []
    DTF = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    KNF = KNeighborsClassifier(n_neighbors=7)
    for i in range(1, 11):
        train_split.append(i * 10)
        X_train_i = X_train[:int(len(X_train)*0.1*i)]
        y_train_i = y_train[:int(len(y_train)*0.1*i)]
        DTF.fit(X_train_i, y_train_i)
        KNF.fit(X_train_i, y_train_i)
        DTF_test_pred = DTF.predict(X_test)
        KNF_test_pred = KNF.predict(X_test)
        DTF_train_pred = DTF.predict(X_train_i)
        KNF_train_pred = KNF.predict(X_train_i)
        DT_train_error.append(1 - metrics.accuracy_score(y_train_i, DTF_train_pred, normalize=True))
        KN_train_error.append(1 - metrics.accuracy_score(y_train_i, KNF_train_pred, normalize=True))
        DT_test_error.append(1 - metrics.accuracy_score(y_test, DTF_test_pred, normalize=True))
        KN_test_error.append(1 - metrics.accuracy_score(y_test, KNF_test_pred, normalize=True))
    plt.title('error of different training sizes for Decision Tree and k-Neighbor')
    plt.xlabel('percentage of training data')
    plt.ylabel('error')
    plt.plot(train_split, DT_test_error, '-', label="Decision Tree test error")
    plt.plot(train_split, DT_train_error, '-', label="Decision Tree train error")
    plt.plot(train_split, KN_test_error, '-.', label="k Neighbor test error")
    plt.plot(train_split, KN_train_error, '-.', label="k Neighbor train error")
    plt.legend()
    plt.savefig('error of different training sizes for Decision Tree and k-Neighbor.png', dpi=300)
    plt.show()

    ### ========== TODO : END ========== ###
    
       
    print('Done')

if __name__ == "__main__":
    main()
