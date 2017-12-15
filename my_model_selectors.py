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


class SelectorBIC(ModelSelector): #bayesian information criterion
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    L: likelihood
    p: # params
    N: # datapoints
    """
    #features_ground
    #Training complete for FISH with 2 states with time 0.27888145027827704 seconds, -246.546743082
    #
    #
    #
    #
    #features_norm
    #
    #features_polar
    #
    #features_delta
    #
    #features_custom
    #
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('-inf')
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            #GaussianHMM takes in a numpy array and a list
            try:
                #print(n_components, self.this_word, 'with length: ', len(self.lengths))
                model = GaussianHMM(n_components = n_components, covariance_type = 'diag', n_iter = 1000,
                                    verbose = self.verbose, random_state = self.random_state).fit(self.X, self.lengths)
                p = n_components * n_components + 2 * n_components * self.X.shape[1] - 1
                score = -2 * model.score(self.X, self.lengths) + p * np.log(self.X.shape[0])
            except:
                print('some error with ', self.this_word)

            if score > best_score:
                best_model = model
                best_score = score
        print('best score: ', score)
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    #features_ground
    #
    #features_norm
    #
    #features_polar
    #
    #features_delta
    #
    #features_custom
    #
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    #features_ground
    #Training complete for FISH with 5 states with time 0.25316671729524387 seconds, 129.629479202
    #Training complete for BOOK with 6 states with time 2.7065385947844334 seconds, -872.181584526
    #Training complete for VEGETABLE with 2 states with time 1.1540903888690082 seconds, -733.102980665
    #Training complete for FUTURE with 2 states with time 2.724060741598805 seconds, -823.622325721
    #Training complete for JOHN with 12 states with time 24.234671652046018 seconds, -6466.75908837
    #features_norm
    #Training complete for FISH with 7 states with time 0.2778986163702939 seconds, 198.327634515
    #Training complete for BOOK with 5 states with time 2.953319445658053 seconds,  -164.310862245
    #Training complete for VEGETABLE with 2 states with time 1.2421704554308235 seconds, -543.54843802
    #Training complete for FUTURE with 2 states with time 2.9048829508210474 seconds, -186.157748889
    #Training complete for JOHN with 7 states with time 30.14987841524271 seconds, -989.344976839
    #features_polar
    #Training complete for FISH with 8 states with time 0.3030782141686359 seconds, 177.228941261
    #Training complete for BOOK with 5 states with time 3.074277725245338 seconds, -388.011624949
    #Training complete for VEGETABLE with 4 states with time 1.3913127217656438 seconds, -523.096316927
    #Training complete for FUTURE with 2 states with time 2.7460244203139155 seconds, -380.284159976
    #Training complete for JOHN with 14 states with time 26.484363145560565 seconds, -2737.76371059
    #features_delta
    #Training complete for FISH with 4 states with time 0.276614712582159 seconds, 163.83318006
    #Training complete for BOOK with 9 states with time 3.9357891861836833 seconds, -243.090832208
    #Training complete for VEGETABLE with 5 states with time 1.2812572582024586 seconds, 150.331275284
    #Training complete for FUTURE with 15 states with time 3.559952718207569 seconds, -102.934264146
    #Training complete for JOHN with 14 states with time 31.8506579218174 seconds, 1491.34573686
    #features_custom
    #Training complete for FISH with 8 states with time 0.3064113161399291 seconds, 213.85722355
    #Training complete for BOOK with 8 states with time 4.687662835387528 seconds, -6.0624130653
    #Training complete for VEGETABLE with 2 states with time 4.112795439843467 seconds, 80.004348799
    #Training complete for FUTURE with 7 states with time 3.984837135316411 seconds, -21.0385378184
    #Training complete for JOHN with 9 states with time 42.12399872481001 seconds, 2152.96757772
    def select(self):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('-inf')

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
        print('best score: ',best_score)
        return best_model
