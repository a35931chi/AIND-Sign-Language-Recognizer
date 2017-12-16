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
    #Training complete for FISH with 2 states with time 0.24785232311842265 seconds, -55.8635132609
    #Training complete for BOOK with 15 states with time 1.4070293507029419 seconds, 5326.0728508
    #Training complete for VEGETABLE with 2 states with time 0.48193655838258564 seconds, 1684.36952095
    #Training complete for FUTURE with 15 states with time 1.51898497986258 seconds, 4518.86250704
    #Training complete for JOHN with 2 states with time 12.88493632011523 seconds, 33155.4870739
    #features_norm
    #Training complete for FISH with 15 states with time 0.25418741028988734 seconds, 765.927165712
    #Training complete for BOOK with 15 states with time 1.2978782986610895 seconds, 938.213287424
    #Training complete for VEGETABLE with 15 states with time 0.5216864961112151 seconds, 284.261432606
    #Training complete for FUTURE with 15 states with time 1.6133087910275208 seconds, 1038.9508218
    #Training complete for JOHN with 2 states with time 11.968922797808773 seconds, 1106.52640794
    #features_polar
    #Training complete for FISH with 15 states with time 0.2570193300707615 seconds, 806.932910796
    #Training complete for BOOK with 15 states with time 1.691023280036461 seconds, 2648.15933662
    #Training complete for VEGETABLE with 15 states with time 0.622443046864646 seconds, 1118.87242499
    #Training complete for FUTURE with 15 states with time 1.2972343937144615 seconds, 2521.4625648
    #Training complete for JOHN with 2 states with time 16.218004940310493 seconds, 11896.2320128
    #features_delta
    #Training complete for FISH with 3 states with time 0.25097179167642025 seconds, -178.297830093
    #Training complete for BOOK with 15 states with time 2.0354490252939286 seconds, 2311.51935148
    #Training complete for VEGETABLE with 15 states with time 0.6006347105940222 seconds, 139.696994457
    #Training complete for FUTURE with 6 states with time 1.4571328472375171 seconds, 1219.95618631
    #Training complete for JOHN with 2 states with time 14.384262818988645 seconds, -6429.55352983
    #features_custom
    #Training complete for FISH with 15 states with time 0.2826078660946223 seconds, 749.318147575
    #Training complete for BOOK with 15 states with time 2.517547836905578 seconds, 985.993736056
    #Training complete for VEGETABLE with 15 states with time 0.8043138926950633 seconds, 432.14275924
    #Training complete for FUTURE with 2 states with time 2.092334105844202 seconds, 253.839358089
    #Training complete for JOHN with 2 states with time 87.57936691794748 seconds, -13756.2029148
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

class SelectorDIC(ModelSelector): #Deviance/Discriminative Information Criterion 
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    #features_ground
    #Training complete for FISH with 3 states with time 0.6466124680737266 seconds, 230180.61414
    #Training complete for BOOK with 15 states with time 2.6932305248774355 seconds, 3567.62159984
    #Training complete for VEGETABLE with 15 states with time 1.9102868847403442 seconds, 61692.8182094
    #Training complete for FUTURE with 15 states with time 2.9005303577287123 seconds, 2872.43351904
    #Training complete for JOHN with 15 states with time 14.206606651001493 seconds, -14012.5834179
    #features_norm
    #Training complete for FISH with 2 states with time 1.6661934280855348 seconds, 5921.76222584
    #Training complete for BOOK with 15 states with time 2.7481078719574725 seconds, 6701.439946
    #Training complete for VEGETABLE with 5 states with time 1.9491248992271721 seconds, 14453.6937081
    #Training complete for FUTURE with 15 states with time 2.9451022210269002 seconds, 2272.96080201
    #Training complete for JOHN with 15 states with time 13.126323453019722 seconds, 1494.16136341
    #features_polar
    #Training complete for FISH with 2 states with time 1.7060124736599391 seconds, 8299.81021031
    #Training complete for BOOK with 14 states with time 3.064462005844689 seconds, 6309.42716149
    #Training complete for VEGETABLE with 12 states with time 2.0801630104106152 seconds, 11653.9121093
    #Training complete for FUTURE with 13 states with time 2.721184953348711 seconds, 2952.05062051
    #Training complete for JOHN with 15 states with time 17.59380264779611 seconds, -4156.55721606
    #features_delta
    #Training complete for FISH with 4 states with time 0.5097239261231152 seconds, 6773.7416203
    #Training complete for BOOK with 15 states with time 3.3638772031554254 seconds, 414.828030913
    #Training complete for VEGETABLE with 2 states with time 2.0508573731058277 seconds, 5506.91592539
    #Training complete for FUTURE with 5 states with time 1.7761915020819288 seconds, -31.9504419956
    #Training complete for JOHN with 15 states with time 15.24973446057993 seconds, 4503.2623067
    #features_custom
    #Training complete for FISH with 3 states with time 1.737266652067774 seconds, 7639.05028789
    #Training complete for BOOK with 15 states with time 3.907765649855719 seconds, 886.218551151
    #Training complete for VEGETABLE with 15 states with time 2.2559553695464274 seconds, 3217.63147386
    #Training complete for FUTURE with 7 states with time 2.6171499859774485 seconds, 237.470563057
    #Training complete for JOHN with 10 states with time 89.40519108172157 seconds, 7986.9030521
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('-inf')
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            #GaussianHMM takes in a numpy array and a list
            try:
                #print(n_components, self.this_word, 'with length: ', len(self.lengths))
                model = GaussianHMM(n_components = n_components, covariance_type = 'diag', n_iter = 1000,
                                    verbose = self.verbose, random_state = self.random_state).fit(self.X, self.lengths)
                LogL_i = model.score(self.X, self.lengths)
                
                avg_LogL_but_i = np.mean([model.score(self.hwords[key][0], self.hwords[key][1]) for key in self.hwords if key != self.this_word])

                score = LogL_i - avg_LogL_but_i
            except:
                print('some error with ', self.this_word)

            if score > best_score:
                best_model = model
                best_score = score
        print('best score: ', score)
        return best_model

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
    #Training complete for BOOK with 5 states with time 2.953319445658053 seconds, -164.310862245
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
