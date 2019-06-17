# -*- coding: utf-8 -*-
# actives Lernnen mit "Pool-based-Sampling"
import sys
sys.path.append("/home/kamgo/activ_lerning _object_dection")
import train_frcnn as train
import test_frcnn as test
import utils
from keras import backend as K
import time
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import errno
import ntpath
from shutil import copyfile,move
import xml.etree.ElementTree as ET
#Zum predict 
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_frcnn import config, data_generators
import argparse
import pickle
import cv2
from operator import itemgetter

base_path='/home/kamgo/activ_lerning _object_dection'
test_path='/home/kamgo/test_image'
pathToPermformance = os.path.join(base_path, 'performance/performance.csv')
pathToDataSet= '/home/kamgo/VOCdevkit'
pathToDataSet = '/media/kamgo/15663FCC59194FED/Activ Leaning/dataset/VOCtrainval_11-May-2012/VOCdevkit'
#pathToSeed = '/home/kamgo/activ_lerning _object_dection/keras-frcnn/train_images' # pfad zum Seed: labellierte Datein, die zum training benutzen werden
#uncertainty sampling method
unsischerheit_methode = "entropie" # kann auch "least_confident oder margin"
batch_size = 300 # batch size for pool based simple
loos_not_change = 3
iteration_zyklus = 5

seed_imgs =[]
seed_classes_count={}
seed_classes_mapping={}
all_imgs =[]
classes_count ={}
class_mapping ={}

# Augmentation flag
horizontal_flips = True # Augment with horizontal flips in training. 
vertical_flips = True   # Augment with vertical flips in training. 
rot_90 = True           # Augment with 90 degree rotations in training. 
output_weight_path = os.path.join(base_path, 'models/model_frcnn.hdf5')
#record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)
base_weight_path = os.path.join(base_path, 'models/model_frcnn.hdf5') #Input path for weights. If not specified, will try to load default weights provided by keras. 
config_output_filename = os.path.join(base_path, 'models/model_frcnn.pickle') #Location to store all the metadata related to the training (to be used when testing).
num_epochs = 2
parser = 'simple' # kann pascal_voc oder Simple(für andere Dataset)
num_rois = 32 # Number of RoIs to process at once.
network = 'resnet50'# Base network to use. Supports vgg or resnet50

def train_vorbereitung ():
    con = config.Config()

    con.use_horizontal_flips = bool(horizontal_flips)
    con.use_vertical_flips = bool(vertical_flips)
    con.rot_90 = bool(rot_90)

    con.model_path = output_weight_path
    #con.model_path='/home/kamgo/Downloads/vgg16_weights.h5'
    con.num_rois = int(num_rois)
    con.network = network
    con.num_epochs= num_epochs
    # loard weight 
    con.base_net_weights = base_weight_path
    #con.base_net_weights ='/home/kamgo/Downloads/resnet50_coco_best_v2.0.1.h5'

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(con, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
  
    return con

def make_prediction(unsischerheit_methode,config):
    """es wurde gespeichert: bilder,vorhergesagtete Klasse mit entsprechende 
    Wahrscheinlichkeiten, und unsischerheitswert"""
    print("############# Anfang der Vorhersage #########")
    list_predicttion_bild_uncert =[]    
    list_pfad_imgs=[]
    for img in all_imgs:
        list_pfad_imgs.append(img['filepath'])
    #list_pfad_imgs = list_pfad_imgs[:500]
    for filepath in list_pfad_imgs:      
        preds = test.make_predicton(filepath,config)
        # unsischerheit rechnen 
        unsischerheit = utils.berechnung_unsischerheit(preds,unsischerheit_methode)
        list_predicttion_bild_uncert.append((filepath,preds,unsischerheit))
        pred_class ={}
        for pre in preds:
            pred_class[pre[0]]=pre[1]
        
        list_predicttion_bild_uncert.append((filepath,pred_class,unsischerheit))

    print("Ende prediction")
    list_predicttion_bild_uncert = utils.sort_list_sampling(list_predicttion_bild_uncert,unsischerheit_methode)
    return list_predicttion_bild_uncert

def pool_based_sampling ():
    """ es wird hier N-Element aus Unlabellierte Datenmenge(all_imgs) nach Seed entfernen
        die N-größer uncertainty"""
    print("Erstellung anotation, class_maping und class_count für neue Seed")
    for seed in seed_imgs:
        for bb in seed['bboxes']:
            if bb['class'] not in seed_classes_count:
                seed_classes_count[bb['class']] = 1
            else:
                seed_classes_count[bb['class']] += 1
            if bb['class'] not in seed_classes_mapping:
                seed_classes_mapping[bb['class']] = len(seed_classes_mapping)
    print("Erstellung anotation, class_maping und class_count vom unlabellierte Daten")     
    for im in all_imgs:
        for bb in im['bboxes']:
            if bb['class'] not in classes_count:
                classes_count[bb['class']] = 1
            else:
                classes_count[bb['class']] += 1
            if bb['class'] not in class_mapping:
                class_mapping[bb['class']] = len(class_mapping)    
    return all_imgs, seed_imgs, class_mapping,classes_count,seed_classes_mapping,seed_classes_count

def oracle(prediction_list):
    # N-Elements to query to Oracle or User
    print("Query oracle")
    to_find = len(prediction_list)
    # Einlesen von N-Elementen, die zur Oracle gefragt werden
    prediction_list= prediction_list[:batch_size]    
    for el in all_imgs:
        for pred in prediction_list:
            if ntpath.basename(el['filepath']) == ntpath.basename(pred[0]):
                to_find=to_find-1
                print("________________Vorhergesagtete Klassen für das Bild : {}".format(ntpath.basename(el['filepath'])))
                for keys,values in pred[1].items():
                    print('{} {} '.format(keys,values))
                print('__________Anwort des Oracle____________')
                for cl in el['bboxes']:
                    print(cl['class'])
                # addieren zum Seedsdataset
                seed_imgs.append(el)
                all_imgs.remove(el)
    print(to_find)
    
if __name__ == "__main__":
    #Erstellung von Seed und unlabellierte Datenmege
    all_imgs,seed_imgs,class_mapping,classes_count,seed_classes_mapping,seed_classes_count=utils.createSeedPlascal_Voc(pathToDataSet,batch_size)
    best_loose = 0
    cur_loos = 0
    iteration = 0
    not_change = 0
    """ # bg 
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    con.class_mapping = class_mapping """
    """ #test
    test = test.test_model(test_path,con)
    predict_list = make_prediction(unsischerheit_methode,con)
    print(predict_list) """
    while (iteration<iteration_zyklus):
        # train
        con = train_vorbereitung()
        start_time = time.time() 
        cur_loos,con = train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping,con)
        #test
        #test = test.test_model(test_path,con)
        # Anwendung des Models
        predict_list = make_prediction(unsischerheit_methode,con)
        # Query to Oracle return anzahl des true positive
        oracle(predict_list)
        performamce ={'unsischerheit_methode':unsischerheit_methode,'Iteration':iteration,'Aktuelle Ungenaueheit':cur_loos,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(predict_list)}
        tils.appendDFToCSV_void(performamce,pathToPermformance,";")
        # Transfer von neuen labellierte Daten zu Seed zu trainieren
        if (len(all_imgs)==0):
            print("kein Datei mehr fürs Training")
            break
        all_imgs,seed_imgs,class_mapping,classes_count,seed_classes_mapping,seed_classes_count=pool_based_sampling()
        iteration = iteration + 1
        #Abbruch Krieterium
        if best_loose != cur_loos:
            if best_loose>=cur_loos:
                # Verbesserung des Models 
                print("das Model hat sich verbessert von: {} loos ist jetzt:{}".format(best_loose, cur_loos))
                not_change+=1
            else:
                best_loose= cur_loos        
        if loos_not_change <= not_change:
            print("nach {} Trainingsiteration hat das Modle keine Verbesserung gamacht. Trainingsphase wird aufgehört: {}".format(loos_not_change))
            break
        