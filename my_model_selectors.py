import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        #Training complete for FISH with 5 states with time 0.25316671729524387 seconds
        #Training complete for BOOK with 6 states with time 2.7065385947844334 seconds
        #Training complete for VEGETABLE with 2 states with time 1.1540903888690082 seconds
        #Training complete for FUTURE with 2 states with time 2.724060741598805 seconds
        #Training complete for JOHN with 12 states with time 24.234671652046018 seconds
        best_score = float('-inf')
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #X = np.array(self.sequences)
        #y = np.array(self.lengths)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            #going into K-folds
            #1. define kfold
            #2. define model
            #3. fit model with train, score with testing, record scores
            inside_scores = []
            if len(self.lengths) <= 2:
                #GaussianHMM takes in a numpy array and a list
                #print('short')
                try:
                    #print(n_components, self.this_word, 'with length: ', len(self.lengths))
                    model = GaussianHMM(n_components = n_components, covariance_type = 'diag', n_iter = 1000,
                                        verbose = self.verbose, random_state = self.random_state).fit(self.X, self.lengths)
                    inside_scores.append(model.score(self.X, self.lengths))
                except:
                    print('some error with ', self.this_word)
                
            else:
                #print('long, kfold')
                kf = KFold()
                for train, test in kf.split(self.sequences):
                    x_train, length_train = combine_sequences(train, self.sequences)
                    x_test, length_test = combine_sequences(test, self.sequences)
                    #GaussianHMM takes in a numpy array and a list
                    try:
                        #print(n_components, self.this_word, 'with length: ', len(self.lengths))
                        model = GaussianHMM(n_components = n_components, covariance_type = 'diag', n_iter = 1000,
                                            verbose = self.verbose, random_state = self.random_state).fit(x_train, length_train)
                        inside_scores.append(model.score(x_test, length_test))
                    except:
                        print('some error with ', self.this_word)
                    
            average_score = np.mean(inside_scores)
            if average_score > best_score:
                best_model = model
                best_score = average_score
                
        return best_model
