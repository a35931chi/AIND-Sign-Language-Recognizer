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
    test_dict = test_set.get_all_Xlengths()
    #print(test_set.get_all_sequences())
    #print(crazy_dict[0][0], crazy_dict[0][1])
    #print(type(crazy_dict[0][0]), type(crazy_dict[0][1]))
    #print(test_set.get_item_sequences())
    #print(test_set.get_item_Xlengths())
    #what = input('wait..')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for key in test_dict:
        temp_dict = {}
        for model_key in models:
            #print(model_key, models[model_key])
            #print(np.array(crazy_dict[key][0]))
            try:
                temp_dict[model_key] = models[model_key].score(test_dict[key][0], test_dict[key][1])
                #print('no error', model_key)
            except:
                pass
                #print('some error', model_key)
        probabilities.append(temp_dict)

    #{'TOY1': -4320663.6128668981, 'GO1': -3837917.7311831936, 'CHICKEN': -8451142.8208826762, 'FISH': -1349.4175554706248, 'WHAT': -533898.25418711454, 'NEXT-WEEK': -10807822.470332241, 'BORROW': -5554728.0818963582, 'FRANK': -4013563.9203068851,
    #{'TOY1': -1405110.6824430912, 'GO1': -1651001.4209484737, 'CHICKEN': -3931316.7022426762, 'FISH': -1279.6217095246971, 'WHAT': -164567.78525418928, 'NEXT-WEEK': -3845482.2700254954, 'BORROW': -2036350.7054552771, 'FRANK': -1109028.5241383566,
    #{'TOY1': -3686966.7350153867, 'GO1': -3815007.0439755265, 'CHICKEN': -9238259.2992473301, 'FISH': -2363.2392820951004, 'WHAT': -448230.90822472738, 'NEXT-WEEK': -9669279.3506101649, 'BORROW': -5052967.9670644403, 'FRANK': -3148184.2374639804,
    
    for probability in probabilities:
        best_value = float('-inf')
        for guess in probability:
            if probability[guess] > best_value:
                best_value = probability[guess]
                best_guess = guess
                #print(best_guess)
        guesses.append(best_guess)
        

    #try these get_all_sequences, get_all_Xlengths, get_item_sequences and get_item_Xlengths
    #self.sequences = all_word_sequences[this_word]
    #self.X, self.lengths = all_word_Xlengths[this_word]
        
    # TODO implement the recognizer
    # return probabilities, guesses
    
    return probabilities, guesses
