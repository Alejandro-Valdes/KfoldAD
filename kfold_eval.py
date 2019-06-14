
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from shutil import copyfile
import pprint

import keras

from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.models import model_from_json

from  keras.applications import vgg16

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import timeit

def getModel(Ri, Ki):
    json_file_path = "./models/vgg16_model_R"+str(Ri)+"_K"+str(Ki)+".json"
    json_file = open(json_file_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    weigth_file_path = "./weights/vgg16_R"+str(Ri)+"_K"+str(Ki)+".h5"
    loaded_model.load_weights(weigth_file_path)
    print("Loaded keras model from disk")

    loaded_model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
          loss='categorical_crossentropy',
          metrics=['acc'])

    return loaded_model

def clean_folders():
    paths = ['./K-Fold/Axial/T2_1/train/AD/',
        './K-Fold/Axial/T2_1/train/NC/',
        './K-Fold/Axial/T2_1/train/sMCI/',
        './K-Fold/Axial/T2_1/train/pMCI/',
        './K-Fold/Axial/T2_1/validate/AD/',
        './K-Fold/Axial/T2_1/validate/NC/',
        './K-Fold/Axial/T2_1/validate/sMCI/',
        './K-Fold/Axial/T2_1/validate/pMCI/',]

    for path in paths:
        files = os.listdir(path)
        for f in files:
            os.remove(path + f)

ensemble_predictions = {}

all_names = os.listdir('./K-Fold/Axial/T2_1/all')

for f in all_names:
    start = f.rindex('_') + 1
    end = f.index('.')
    ensemble_predictions[f[:start-1]] = {'id':f[:start-1], 'truth': f[start:end], 'final_pred': '', 'predictions': [''] * 10, 'K': [''] * 10, 'probs': [''] * 10}

results = []

all_path = './K-Fold/Axial/T2_1/all/'
validate_path = './K-Fold/Axial/T2_1/validate/'
validation_dir = validate_path[:-1]
nClasses = 4
batch_size = 20

from lists import *

for run in range(10):
    for currK in range(5):
        clean_folders()

        key = "R" + str(run) + "K" + str(currK)
        validated_on = lists[key]

        nVal = len(validated_on)

        for fname in validated_on:
            start = fname.rindex('_') + 1
            end = fname.index('.')
            s_class = fname[start:end]
            print(s_class)
            if(s_class == 'AD'):
                copyfile(all_path + fname, validate_path + 'AD/' + fname)
            elif(s_class == 'NC'):
                copyfile(all_path + fname, validate_path + 'NC/' + fname)
            elif(s_class == 'pMCI'):
                copyfile(all_path + fname, validate_path + 'pMCI/' + fname)
            elif(s_class == 'sMCI'):
                copyfile(all_path + fname, validate_path + 'sMCI/' + fname)

        model = getModel(run, currK)

        test_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (224, 224),
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle=False)

        fnames =  validation_generator.filenames
        ground_truth = validation_generator.classes

        label2index = validation_generator.class_indices
        idx2label = dict((v,k) for k,v in label2index.items())

        # Get the predictions from the model using the generator
        nb_samples = len(fnames)

        predictions = model.predict_generator(validation_generator,
            steps=validation_generator.samples/validation_generator.batch_size,
            verbose=1)

        predicted_classes = np.argmax(predictions,axis=1)

        errors = np.where(predicted_classes != ground_truth)[0]
        print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
        print("acc " + str(1 - (len(errors)) / validation_generator.samples))
        print(confusion_matrix(ground_truth, predicted_classes))

        for i in range(len(predicted_classes)):
            pred_class = predicted_classes[i]
            pred_label = idx2label[pred_class]

            curr = fnames[i].split('/')[-1]
            cutoff = f.rindex('_')
            curr_name = curr[:cutoff]
            ensemble_predictions[curr_name]['predictions'][run] = pred_label
            ensemble_predictions[curr_name]['probs'][run] = [predictions[i][0],
                    predictions[i][1],
                    predictions[i][2],
                    predictions[i][3]]
            ensemble_predictions[curr_name]['K'][run] = currK

def getFinalPred(AD, NC, pMCI, sMCI):
    preds = {'AD': AD, 'NC': NC, 'pMCI': pMCI, 'sMCI': sMCI}

    return max(preds, key=preds.get)


correct = 0

for img in ensemble_predictions:
    AD = 0
    NC = 0
    pMCI = 0
    sMCI = 0
    for pred in ensemble_predictions[img]['predictions']:
        if pred == 'AD':
            AD += 1
        elif pred == 'NC':
            NC += 1
        elif pred == 'pMCI':
            pMCI += 1
        elif pred == 'sMCI':
            sMCI += 1

    ensemble_predictions[img]['final_pred'] = getFinalPred(AD, NC, pMCI, sMCI)

    if ensemble_predictions[img]['final_pred'] == ensemble_predictions[img]['truth']:
        correct += 1

import csv

with open('results.csv', 'w') as f:  # Just use 'w' mode in 3.x

    for p in ensemble_predictions:
        w = csv.DictWriter(f, ensemble_predictions[p].keys())
        w.writeheader()
        break

    for p in ensemble_predictions:
        w.writerow(ensemble_predictions[p])

print('ACC' + str(correct/len(ensemble_predictions)))
