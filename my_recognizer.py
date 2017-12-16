import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    crazy_dict = test_set.get_all_sequences()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for key in crazy_dict:
        temp_dict = {}
        for model_key in models:
            #print(model_key, models[model_key])
            #print(np.array(crazy_dict[key][0]))
            try:
                temp_dict[model_key] = models[model_key].score(np.array(crazy_dict[key][0]))
                #print('no error', model_key)
            except:
                pass
                #print('some error', model_key)
        probabilities.append(temp_dict)

    #print(len(probabilities))

    for probability in probabilities:
        best_value = float('-inf')
        for guess in probability:
            if probability[guess] > best_value:
                best_value = probability[guess]
                best_guess = guess
        guesses.append(best_guess)
        
    #try these get_all_sequences, get_all_Xlengths, get_item_sequences and get_item_Xlengths
    #self.sequences = all_word_sequences[this_word]
    #self.X, self.lengths = all_word_Xlengths[this_word]
        
    # TODO implement the recognizer
    # return probabilities, guesses
    
    return probabilities, guesses
