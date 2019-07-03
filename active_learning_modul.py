# -*- coding: utf-8 -*-
# actives Lernnen mit "Pool-based-Sampling"
import sys
import os
sys.path.append(os.getcwd())
import train_frcnn as train
import test_frcnn as test
import utils
from keras import backend as K
import time
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

base_path=os.getcwd()
#test_path='/home/kamgo/test_image'
pathToPermformance = os.path.join(base_path, 'performance/performance.csv')
pathToDataSet= '/home/kamgo/VOCdevkit'
#pathToDataSet = '/media/kamgo/15663FCC59194FED/Activ Leaning/dataset/VOCtrainval_11-May-2012/VOCdevkit'
#pathToSeed = '/home/kamgo/activ_lerning _object_dection/keras-frcnn/train_images' # pfad zum Seed: labellierte Datein, die zum training benutzen werden

#uncertainty sampling method
unsischerheit_methode = "entropie" # kann auch "least_confident oder "margin"
batch_size = 200 # batch size for pool based simple
batch = 50 # Anzahl von daein zu senden to oracle
loos_not_change = 10 # wie oft soll das weiter trainiert werden, ohne eine Verbesserung von perfomance

seed_imgs =[]
seed_classes_count={}
seed_classes_mapping={}
all_imgs =[]
datatosendtoOracle=[]
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
num_epochs = 1000
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
    Wahrscheinlichkeiten, und unsischerheitswert: predict über batch-Element"""
    print("############# Anfang der Vorhersage #########")
    list_predicttion_bild_uncert =[]    
    list_pfad_imgs=[]
    for img in all_imgs:
        list_pfad_imgs.append(img['filepath'])
        
    print("Anzahl von bildern zu predict:{}".format(len(list_pfad_imgs)))
    for filepath in list_pfad_imgs:      
        preds = test.make_predicton(filepath,config)
        print(preds)
        # unsischerheit rechnen 
        unsischerheit = utils.berechnung_unsischerheit(preds,unsischerheit_methode)
        list_predicttion_bild_uncert.append((filepath,preds,unsischerheit))
      
    print("Ende der Vorhersage")
    return list_predicttion_bild_uncert

""" def pool_based_sampling (list_predict,list_image_send_for_predict):
    
    neue_seed =[]
    for el in list_predict:

    
    return neue_seed """

def oracle(prediction_list,batch_size,uncertainty_m,trainingsmenge):
    neue_seed =[]
    print("Query Oracle")
    # auf basis von Unsischerheit wird die Liste  sortiert
    prediction_list = utils.sort_list_sampling(prediction_list,uncertainty_m)
    # die  erste batch-size Element werden rausgenommen
    # 
    prediction_list = prediction_list[:batch_size]
    to_find = len(prediction_list)
    truePositiv = 0
    for el in all_imgs:
        for pred in prediction_list:
            if ntpath.basename(el['filepath']) == ntpath.basename(pred[0]):
                to_find=to_find-1
                neue_seed.append(el)
                print("________________Vorhergesagtete Klassen für das Bild : {}".format(ntpath.basename(el['filepath'])))
                for val in pred[1]:
                    print("___________________________________________________________")
                    print('{} {} '.format(val[0],val[1]))
                    print('__________Anwort der Oracle____________')
                    for cl in el['bboxes']:
                        print(cl['class'])
                        if cl['class']== val[0]:
                            print("das model gut predicted!")
                            truePositiv +=1
                            

    print("true positive:{}".format(truePositiv))
    trainingsmenge = trainingsmenge + neue_seed
    for el in neue_seed:
        all_imgs.remove(el)
    print(len(all_imgs))
    return truePositiv,trainingsmenge
           
if __name__ == "__main__":
    #Erstellung von Seed und unlabellierte Datenmege
    datatosendtoOracle,seed_imgs,class_mapping,classes_count,seed_classes_mapping,seed_classes_count=utils.createSeedPlascal_Voc(pathToDataSet,batch_size)
    best_loose = 0
    cur_loos = 0
    iteration = 0
    not_change = 0
    #test    
    while (len(datatosendtoOracle)!=0):
        # train
        iteration += 1
        con = train_vorbereitung()
        start_time = time.time()
        # Transfer von neuen labellierte Daten zu Seed zu trainieren
        if len(all_imgs)==0:
            all_imgs = datatosendtoOracle[:batch]
            datatosendtoOracle= datatosendtoOracle[batch:]
        else:
            all_imgs = all_imgs + datatosendtoOracle[:batch_size]
            datatosendtoOracle= datatosendtoOracle[batch_size:]

        print("size of train data: {}".format(len(seed_imgs)))
        print("size of data to predict {}".format(len(all_imgs)))
        print("size of data reste data {}".format(len(datatosendtoOracle)))
        cur_loos,con = train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping,con)
        #test
        #test = test.test_model(test_path,con)

        # Anwendung des Models
        predict_list = make_prediction(unsischerheit_methode,con)
        # Query to Oracle: zurückgegeben wird anzahl der rictige vorhergesahte Klasse und die neue Trainingsmenge 
        truepositiv, seed_imgs = oracle(predict_list,batch_size,unsischerheit_methode,seed_imgs)
        # die batch-size Element, die von der Oracle überprüfen wurden,werden in der trainingsmenge übertragen       
        #seed_imgs=pool_based_sampling(list_predict_sort,unsischerheit_methode)
        seed_classes_count,seed_classes_mapping=utils.create_mapping_cls(seed_imgs)      
        performamce ={'unsischerheit_methode':unsischerheit_methode,'Iteration':iteration,'Aktuelle Ungenaueheit':cur_loos,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(predict_list),'Gut predicted':truepositiv}
        utils.appendDFToCSV_void(performamce,pathToPermformance)
        
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