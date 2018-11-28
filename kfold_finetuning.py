import tensorflow as tf

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

from  keras.applications import vgg16

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import timeit

def clean_folders():
    paths = ['./K-Fold/Axial/T2_1/train/AD/', './K-Fold/Axial/T2_1/train/NC/', './K-Fold/Axial/T2_1/validate/AD/', './K-Fold/Axial/T2_1/validate/NC/']
    for path in paths:
        files = os.listdir(path)
        for f in files:
            os.remove(path + f)

clean_folders()
ensemble_predictions = {}

fnames = os.listdir('./K-Fold/Axial/T2_1/all')

for f in fnames:
    start = f.rindex('_') + 1
    end = f.index('.')
    ensemble_predictions[f[:start-1]] = {'id':f[:start-1], 'truth': f[start:end], 'final_pred': '', 'predictions': [''] * 10, 'K': [''] * 10, 'probs': [''] * 10}

kf = KFold(n_splits=5)
all_path = './K-Fold/Axial/T2_1/all/'
train_path = './K-Fold/Axial/T2_1/train/'
validate_path = './K-Fold/Axial/T2_1/validate/'

results = []

for run in range(10):

    fnames = os.listdir(all_path[:-1])
    random.shuffle(fnames)
    X = fnames
    Y = []

    for f in X:
        start = f.rindex('_') + 1
        end = f.index('.')
        Y.append(f[start:end])

    currK = 0

    for train_index, test_index in kf.split(X):
        clean_folders()

        for i in train_index:
            if(Y[i] == 'AD'):
                copyfile(all_path + X[i], train_path + 'AD/' + X[i])
            elif(Y[i] == 'NC'):
                copyfile(all_path + X[i], train_path + 'NC/' + X[i])

        for i in test_index:
            if(Y[i] == 'AD'):
                copyfile(all_path + X[i], validate_path + 'AD/' + X[i])
            elif(Y[i] == 'NC'):
                copyfile(all_path + X[i], validate_path + 'NC/' + X[i])

        nTrain = len(train_index)
        nVal = len(test_index)

        print(nTrain)
        print(nVal)

        K.clear_session()

        train_dir = train_path[:-1]
        validation_dir = validate_path[:-1]

        vgg_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        nClasses = 2
        batch_size = 20

        for layer in vgg_model.layers[:-4]:
            layer.trainable = False


        x = vgg_model.output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(nClasses, activation="softmax")(x)

        model = Model(inputs = vgg_model.input, outputs = predictions)

        model.compile(loss = "categorical_crossentropy", 
            optimizer = optimizers.RMSprop(lr=2e-4),
            metrics=["acc"])

        # Initiate the train and test generators with data Augumentation 
        train_datagen = ImageDataGenerator(rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (224, 224),
        batch_size = batch_size, 
        class_mode = "categorical",
        shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (224, 224),
        batch_size = batch_size, 
        class_mode = "categorical",
        shuffle=True)

        model_save_name = "vgg16_R"+str(run)+"_K"+ str(currK) +".h5"

        checkpoint = ModelCheckpoint(model_save_name, 
            monitor='val_acc', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='auto', 
            period=1)

        early = EarlyStopping(monitor='val_acc', 
            min_delta=0.001, 
            patience=8, 
            verbose=1, 
            mode='auto')

        #save model
        model_json = model.to_json()
        with open("vgg16_model_R"+str(run)+"_K"+ str(currK) +".json", "w") as json_file:
            json_file.write(model_json)

        history = model.fit_generator(
            train_generator,
            epochs = 100,
            validation_data = validation_generator,
            callbacks = [checkpoint, early],
            verbose=1)

        fnames =  validation_generator.filenames

        ground_truth = validation_generator.classes

        label2index = validation_generator.class_indices

        # Getting the mapping from class index to class label
        idx2label = dict((v,k) for k,v in label2index.items())


        # Get the predictions from the model using the generator
        predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
        predicted_classes = np.argmax(predictions,axis=1)
         
        errors = np.where(predicted_classes != ground_truth)[0]
        print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

        print(confusion_matrix(ground_truth, predicted_classes))

        nb_samples = len(fnames)
        prob = model.predict_generator(validation_generator,steps = nb_samples)

        errors = np.where(predictions != ground_truth)[0]

        [test_loss, test_acc] = model.evaluate_generator(validation_generator, nb_samples/batch_size)

        print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(historytest_loss, test_acc))

        results.append({'loss': test_loss, 'acc': test_acc})

        for i in range(len(predictions)):
            pred_class = np.argmax(prob[i])
            pred_label = idx2label[pred_class]

            curr = fnames[i].split('/')[-1]
            cutoff = f.rindex('_')
            curr_name = curr[:cutoff]
            ensemble_predictions[curr_name]['predictions'][run] = pred_label
            ensemble_predictions[curr_name]['probs'][run] = [prob[i][0],prob[i][1]]
            ensemble_predictions[curr_name]['K'][run] = currK

        K.clear_session()
        tf.reset_default_graph()

        currK += 1

correct = 0
             
for img in ensemble_predictions:
    AD = 0
    NC = 0
    for pred in ensemble_predictions[img]['predictions']:
        if pred == 'AD':
            AD += 1
        elif pred == 'NC':
            NC += 1

    ensemble_predictions[img]['final_pred'] = 'AD' if AD >= NC else 'NC'

    if ensemble_predictions[img]['final_pred'] == ensemble_predictions[img]['truth']:
        correct += 1


import csv

with open('sag.csv', 'w') as f:  # Just use 'w' mode in 3.x

    for p in ensemble_predictions:
        w = csv.DictWriter(f, ensemble_predictions[p].keys())
        w.writeheader()
        break    

    for p in ensemble_predictions:
        w.writerow(ensemble_predictions[p])

print('ACC' + str(correct/len(ensemble_predictions)))
