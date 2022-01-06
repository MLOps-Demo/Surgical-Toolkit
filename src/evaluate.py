# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:30:00 2022

@author: AMIT CHAKRABORTY
"""
import sklearn.metrics as metrics
import pickle
import json
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import json

def evaluate_model():

    json_file = open('model_in_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('saved-models/weights.h5')
    test_data = np.load('data/test_data.npy')
    test_label = np.load('data/test_label.npy')
    
    test_score = loaded_model.evaluate(test_data, test_label)
    print('loss',test_score[0])
    print('accuracy',test_score[1])
    with open("scores.json", "w") as fd:
        json.dump({"loss": test_score[0], "accuracy": test_score[1]}, fd, indent=4)
        
if __name__ == '__main__':
    evaluate_model()