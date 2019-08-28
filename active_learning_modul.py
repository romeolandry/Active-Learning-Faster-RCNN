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
from optparse import OptionParser

base_path=os.getcwd()
parser = OptionParser()

parser.add_option("-p", "--path", dest="pathToDataSet", help="Path to VOC2012")
parser.add_option("-m", "--mode", dest="trainingsmode", help="simple / batch", default="simple")
parser.add_option("-s", dest="sampling_method", help="cloud be [entropie, margin or least_confident]", default ="entropie")
parser.add_option("--pool", dest="pool_size", help="number of image to query to oracle",default=300)
parser.add_option("--seed", dest="seed_size", help="size of seed im percent",default=30)
parser.add_option("--stop", dest="stopping", help="stop the active learning proces after n-Iteration without amelioration",default=20)
parser.add_option("-e", "--epochs", dest="num_epochs", help="",default=1000)

parser.add_option("-c", "--config_filename", dest="config_filename", help=
				"Location to write the metadata related to the training (generated when training).",
				default=os.path.join(base_path, 'models/out_model.pickle'))
parser.add_option("--base_weight", dest="base_weight_path", help ="give the path of weight file",
                default = os.path.join(base_path, 'models/model_frcnn.hdf5'))
parser.add_option("--output_weight", dest="output_weight_path", help ="give the path of file save model",
                default = os.path.join(base_path, 'models/out_model.hdf5'))
parser.add_option("--perform", dest="pathToPermformance", help ="give the paht file to save performance",
                default = os.path.join(base_path, 'performance/out_model.csv'))
(options, args) = parser.parse_args()


pathToDataSet = options.pathToDataSet
base_weight_path = options.base_weight_path #Input path for weights. If not specified, will try to load default weights provided by keras.'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' 
output_weight_path = options.output_weight_path
config_output_filename = options.config_filename #Location to store all the metadata related to the training (to be used when testing).
pathToPermformance = options.pathToPermformance
train_mode = options.trainingsmode
num_epochs = options.num_epochs

#uncertainty sampling method
unsischerheit_methode = options.sampling_method
batch_size = options.seed_size 
to_Query =  options.pool_size  
loos_not_change =options.stopping



Earlystopping_patience= None
train_size_pro_batch = 50 # N-Prozen von batch-size element
num_rois = 32 # Number of RoIs to process at once default 32 I reduice it to 16.
network = 'resnet50'# Base network to use. Supports vgg or resnet50

seed_imgs =[]
seed_classes_count={}
seed_classes_mapping={}
all_imgs =[]
classes_count ={}
class_mapping ={}
print("save hyperparameter")
config_img = config.Config()

def train_vorbereitung ():
    # Augmentation flag
    horizontal_flips = True # Augment with horizontal flips in training. 
    vertical_flips = True   # Augment with vertical flips in training. 
    rot_90 = True           # Augment with 90 degree rotations in training. 
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
    print("size of data {}".format(len(pool)))
    neue_seed =[]
    to_find = len(pool)
    truePositiv = 0 # das Model hat gut predict
    trueNegativ = 0 # das Model hat ein Objekt predict aber war falsch
    not_predict =0 # das Model hat kein Objekt predict
    for pred in pool:
        for el in all_imgs:
            if ntpath.basename(el['filepath']) == ntpath.basename(pred[0]):
                to_find-=1
                print("----------> für das Bild {}".format(ntpath.basename(el['filepath'])))
                neue_seed.append(el)
                all_imgs.remove(el)
                all_bg,list_not_bg = utils.check_predict(pred[1])
                if all_bg == True:
                    not_predict+=1
                    print ("Model hat nur bg anerkannt {}".format(not_predict))
                    continue
                else:
                    for val in list_not_bg:
                        print(" print val ")
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

    print("list of not bg: {} true positive:{} true negativ: {} and not predict: {}".format(len(list_not_bg),truePositiv, trueNegativ, not_predict))
    trainingsmenge = trainingsmenge + neue_seed
    return truePositiv, trueNegativ,not_predict,trainingsmenge,all_imgs
           
def train_simple():
    print("##################### simple Train #################################")
    con = train_vorbereitung()
    cur_loos = con.best_loss
    iteration = 0
    not_change = 0
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
        sizetopredict = int(round((len(all_imgs)* train_size_pro_batch)/100))
        list_to_predict = all_imgs[:sizetopredict]
        print("size of data to predict {}".format(len(list_to_predict)))
        print("weight {}".format(con.best_loss))
        predict_list=test.make_predicton_new(list_to_predict,con)
        print("Anwendung des Pool_based sampling")
        pool = utils.Pool_based_sampling_test(predict_list,to_Query,method)
        print("Abfrage an der Oracle")
        truePositiv, trueNegativ,not_predict,seed_imgs,all_imgs = oracle(pool,all_imgs,seed_imgs)  
        performamce ={'unsischerheit_methode':method, 'num_roi':num_rois, 'img_size':config_img.im_size, 'Iteration':iteration,'Aktuelle_verlust':con.best_loss,'seed':len(seed_imgs),'batch_size':batch_size,'to_Query':to_Query, 'num_epochs':num_epochs ,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(pool),'Good predicted':truePositiv,'Falsh_predicted':trueNegativ,'not_prediction':not_predict,}
        utils.appendDFToCSV_void(performamce,pathToPermformance)            
        #Abbruch Krieterium
        if con.best_loss<cur_loos:
            # Verbesserung des Models 
            print("das Model hat sich verbessert von: {} loos ist jetzt :{}".format(cur_loos,con.best_loss))
            cur_loos=con.best_loss
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

def train_Batch():
    print("##################### Train on Batch ##############################")
    con = train_vorbereitung()
    cur_loos = con.best_loss
    iteration = 0
    not_change = 0
    batch_number = 0
    train_size_pro_batch = 40
    print("base net {} and losse {} ".format(con.base_net_weights ,con.best_loss))
    list_batch = utils.create_batch_from_path(pathToDataSet,batch_size)
    for batch in list_batch:
        all_imgs,seed_imgs,seed_classes_mapping,seed_classes_count = utils.createSeed_pro_batch(batch,train_size_pro_batch)
        con.class_mapping = class_mapping
        con = utils.update_config_file(config_output_filename,con)
        while (len(all_imgs)!=0):
            iteration += 1
            start_time = time.time()
            print("size of train data: {}".format(len(seed_imgs)))
            print("size of data reste data {}".format(len(all_imgs)))
            con = train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping,con,Earlystopping_patience,config_output_filename)
            predict_list=test.make_predicton_new(all_imgs,con)
            print("Anwendung des Pool_based sampling")
            pool = utils.Pool_based_sampling_test(predict_list,to_Query,unsischerheit_methode)
            print("Abfrage an der Oracle")
            print("size of data {}".format(to_Query))
            truePositiv, trueNegativ,not_predict,seed_imgs,all_imgs = oracle(pool,all_imgs,seed_imgs)  
            performamce ={'unsischerheit_methode':unsischerheit_methode, 'num_roi':num_rois, 'img_size':config_img.im_size, 'Batch':batch_number+1, 'Iteration':iteration,'Aktuelle_verlust':con.best_loss,'seed':len(seed_imgs),'batch_size':batch_size,'to_Query':to_Query, 'num_epochs':num_epochs ,'abgelaufene Zeit':time.time() - start_time,'Anzahl der vorhergesagteten Bildern':len(pool),'Good predicted':truePositiv,'Falsh_predicted':trueNegativ,'not_prediction':not_predict,}
            utils.appendDFToCSV_void(performamce,pathToPermformance)            
            #Abbruch Krieterium
            if con.best_loss<cur_loos:
                # Verbesserung des Models 
                print("das Model hat sich verbessert von: {} loos ist jetzt :{}".format(cur_loos,con.best_loss))
                cur_loos=con.best_loss
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
if __name__ == "__main__":
    if train_mode == 'batch':
        train_Batch()
    else:
        train_simple()