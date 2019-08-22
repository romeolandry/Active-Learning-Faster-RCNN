# -*- coding: utf-8 -*-
# actives Lernnen mit "Pool-based-Sampling"
import sys
import os
sys.path.append(os.getcwd())
from keras_frcnn import train_frcnn as train
from keras_frcnn import test_frcnn as test
import utils

import time
from keras.callbacks import TensorBoard
from keras_frcnn import config

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
from numba import jit

pathToDataSet = sys.argv[1]
#pathToDataSet= '/media/romeo/Volume/dataset/VOCtrainval_11-May-2012/VOCdevkit'
base_path=os.getcwd()
#test_path='/home/kamgo/test_image'
pathToPermformance = os.path.join(base_path, 'performance/'+ sys.argv[2]+'.csv')
#pathToDataSet = '/media/kamgo/15663FCC59194FED/Activ Leaning/dataset/VOCtrainval_11-May-2012/VOCdevkit'
#pathToSeed = '/home/kamgo/activ_lerning _object_dection/keras-frcnn/train_images' # pfad zum Seed: labellierte Datein, die zum training benutzen werden

#uncertainty sampling method
unsischerheit_methode = "entropie" # kann auch "least_confident oder "margin"
batch_size =30 # Prozenzahl von Daten  pro batch_lement
train_size_pro_batch = 50 # N-Prozen von batch-size element
to_Query = 2 # Anzahl von daten, die zu dem Oracle gesenden werden. auch batch for Pool-based sampling

loos_not_change = 20 # wie oft soll das weiter trainiert werden, ohne eine Verbesserung der Leistung

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
output_weight_path = os.path.join(base_path, 'models/' + sys.argv[3]+ '.hdf5')

#record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)
base_weight_path = os.path.join(base_path, 'models/model_frcnn.hdf5') #Input path for weights. If not specified, will try to load default weights provided by keras.'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' 
config_output_filename = os.path.join(base_path, 'models/' + sys.argv[3]+ '.pickle') #Location to store all the metadata related to the training (to be used when testing).
num_epochs = 1
Earlystopping_patience= None

parser = 'simple' # kann pascal_voc oder Simple(für andere Dataset)
num_rois = 32 # Number of RoIs to process at once default 32 I reduice it to 16.
network = 'resnet50'# Base network to use. Supports vgg or resnet50
print("save hyperparameter")
config_img = config.Config()

def train_vorbereitung ():
    con = config.Config()

    con.use_horizontal_flips = bool(horizontal_flips)
    con.use_vertical_flips = bool(vertical_flips)
    con.rot_90 = bool(rot_90)
    # zum weiter Training
    if os.path.exists(config_output_filename):
        print(" Weiter Training")
        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)
            con.model_path = C.model_path
            con.num_rois = int(C.num_rois)
            con.network = C.network
            con.num_epochs= C.num_epochs
            # loard weight
            con.base_net_weights = C.model_path
            con.class_mapping = C.class_mapping
            con.best_loss = C.best_loss                        
    else:
        print("new traininig")
        con.model_path = output_weight_path
        con.num_rois = int(num_rois)
        con.network = network
        con.num_epochs= num_epochs
        # loard weight
        con.base_net_weights = base_weight_path
        con.class_mapping = 0
        con.best_loss = np.Inf 


    #con.base_net_weights = output_weight_path #weiter training
    #con.base_net_weights ='/home/kamgo/Downloads/resnet50_coco_best_v2.0.1.h5'

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(con, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
  
    return con

def oracle(pool,all_imgs,trainingsmenge):

    neue_seed =[]
    to_find = len(pool)
    truePositiv = 0 # das Model hat gut predict
    trueNegativ = 0 # das Model hat ein Objekt predict aber war falsch
    not_predict =0 # das Model hat kein Objekt predict
    for pred in pool:
        for el in all_imgs:
            if ntpath.basename(el['filepath']) == ntpath.basename(pred[0]):
                to_find-=1
                neue_seed.append(el)
                all_imgs.remove(el)
                all_bg,list_not_bg = utils.check_predict(pred[1])
                if all_bg == False:
                    not_predict+=1
                    print ("Model hat nur bg anerkannt")
                    continue
                else:
                    for val in list_not_bg:
                        print("________________Vorhergesagtete Klassen für das Bild : {}".format(ntpath.basename(el['filepath'])))                      
                        for cl in el['bboxes']:
                            if cl['class']== val[0]:
                                print("das Model predict : {} mit {} % Sicherheit".format(val[0],val[1]))
                                print("oracle: die Liste Von Objekt auf dem Bild ".format(cl['class']))
                                print("gut predicted!")
                                truePositiv +=1
                            else:
                                print("falsch Predict")
                                trueNegativ += 1
                                print("das Model predict : {} mit {} % Sicherheit".format(val[0],val[1]))
                                print("oracle: die Liste Von Objekt auf dem Bild ".format(cl['class']))

    print("true positive:{}".format(truePositiv))
    trainingsmenge = trainingsmenge + neue_seed
    return truePositiv, trueNegativ,not_predict,trainingsmenge,all_imgs
           
def trian_simple():
    cur_loos = 0
    iteration = 0
    not_change = 0  
    con = train_vorbereitung()
    print("base net {} and losse {} ".format(con.base_net_weights ,con.best_loss))
    all_imgs,seed_imgs,seed_classes_mapping,seed_classes_count = utils.createSeedPascal_Voc(pathToDataSet,batch_size)
    con.class_mapping = class_mapping
    con = utils.update_config_file(config_output_filename,con)
    while (len(all_imgs)!=0):
        iteration += 1
        start_time = time.time()
        print("size of train data: {}".format(len(seed_imgs)))
        print("size of data reste data {}".format(len(all_imgs)))
        con = train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping,con,Earlystopping_patience,config_output_filename)
        print("size of data to predict {}".format(len(all_imgs)))
        predict_list=test.make_predicton_new(all_imgs,con)
        print("Anwendung des Pool_based sampling")
        pool = utils.Pool_based_sampling_test(predict_list,to_Query,unsischerheit_methode)
        print("Abfrage an der Oracle")
        print("size of data {}".format(to_Query))
        truePositiv, trueNegativ,not_predict,seed_imgs,all_imgs = oracle(pool,all_imgs,seed_imgs)  
        performamce ={'unsischerheit_methode':unsischerheit_methode, 'num_roi':num_rois, 'img_size':config_img.im_size, 'Iteration':iteration,'Aktuelle_verlust':cur_loos,'seed':len(seed_imgs),'batch_size':batch_size,'to_Query':to_Query, 'num_epochs':num_epochs ,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(pool),'Good predicted':truePositiv,'Falsh_predicted':trueNegativ,'not_prediction':not_predict,}
        utils.appendDFToCSV_void(performamce,pathToPermformance)            
        #Abbruch Krieterium
        if con.best_loss>cur_loos:
            # Verbesserung des Models 
            print("das Model hat sich verbessert von: {} loos ist jetzt :{}".format(con.best_loss, cur_loos))
            con.best_loss= cur_loos
            con.base_net_weights = con.model_path
            not_change = loos_not_change
            con = utils.update_config_file(config_output_filename,con)
            print("neue  base net weight: {}".format(con.base_net_weights))
            not_change = 0                  
        else:
            not_change +=1     
        if loos_not_change <= not_change:
            print("nach {} Trainingsiteration hat das Modle keine Verbesserung gamacht. Trainingsphase wird aufgehört: {}".format(not_change,loos_not_change))
            break

""" def train_batch():
    #Erstellung von Seed und unlabellierte Datenmege
    #batchtify,classes_count,class_mapping = utils.create_batchify_from_path(pathToDataSet,batch_size)
    #print(" Es gibt: ", len(batchtify), "Batch von je: ", len(batchtify[0]), " Bilder")
    #batch_numb = 0 
    con = train_vorbereitung()
    cur_loos = con.best_loss
    iteration = 0
    not_change = 0 
    all_imgs,seed_imgs,class_mapping,classes_count,seed_classes_mapping,seed_classes_count = utils.createSeedPascal_Voc(pathToDataSet,batch_size)
    # release gpu memory    
    #test
    #print("the next Batch: ", batch_numb)
    print("size of train data: {}".format(len(seed_imgs)))
    print("size of data reste data {}".format(len(all_imgs)))
    while (len(all_imgs)!=0):
        # train
        #utils.reset_keras()
        iteration += 1
        start_time = time.time()
        print("size of train data: {}".format(len(seed_imgs)))
        print("size of data reste data {}".format(len(all_imgs)))
        con = train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping,con,num_epochs,Earlystopping_patience)
        #test
        #utils.reset_keras()
        #utils.clear_keras()
        #test = test.test_model(test_path,con)
        # Anwendung des Models
        pool = all_imgs[:to_Query]
        all_imgs = all_imgs[to_Query:]
        print("size of data to predict {}".format(len(pool)))
        predict_list = make_prediction(unsischerheit_methode,pool,con)
        # Query to Oracle: zurückgegeben wird anzahl der rictige vorhergesahte Klasse und die neue Trainingsmenge
        print("Abfrage an der Oracle")
        print("size of data {}".format(len(predict_list)))
        truePositiv, trueNegativ,not_predict,seed_imgs = oracle(pool,predict_list,to_Query,unsischerheit_methode,seed_imgs)
        # die batch-size Element, die von der Oracle überprüfen wurden,werden in der trainingsmenge übertragen       
        #seed_imgs=pool_based_sampling(list_predict_sort,unsischerheit_methode)
        seed_classes_count,seed_classes_mapping=utils.create_mapping_cls(seed_imgs)      
        performamce ={'unsischerheit_methode':unsischerheit_methode, 'num_roi':num_rois, 'img_size':config_img.im_size, 'Iteration':iteration,'Aktuelle_verlust':cur_loos,'seed':len(seed_imgs),'batch_size':batch_size,'to_Query':to_Query, 'num_epochs':num_epochs ,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(predict_list),'Good predicted':truePositiv,'Falsh_predicted':trueNegativ,'not_prediction':not_predict,}
        utils.appendDFToCSV_void(performamce,pathToPermformance)            
        #Abbruch Krieterium
        if cur_loos>con.best_loss:
            # Verbesserung des Models 
            print("das Model hat sich verbessert von: {} loos ist jetzt :{}".format(cur_loos,con.best_loss))
            cur_loos = con.best_loss
            con.base_net_weights = con.model_path
            not_change = loos_not_change
            con = utils.update_config_file(config_output_filename,con)
            print("neue  base net weight: {}".format(con.base_net_weights))
            not_change = 0                  
        else:
            not_change +=1     
        if loos_not_change <= not_change:
            print("nach {} Trainingsiteration hat das Modle keine Verbesserung gamacht. Trainingsphase wird aufgehört: {}".format(not_change,loos_not_change))
            break """
if __name__ == "__main__":
    trian_simple()
    