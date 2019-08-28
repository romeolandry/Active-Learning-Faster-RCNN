import sys
from operator import itemgetter
from keras_frcnn.pascal_voc_parser import get_data
import os
sys.path.append(os.getcwd())
import glob
import errno
import ntpath
from shutil import copyfile,move
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import operator
from operator import itemgetter
import random
import pickle
from numba import jit

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow


# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    #print(gc.collect()) # if it's done something you should see a number being outputted
    ###################################
    # TensorFlow wizardry
    config = tensorflow.ConfigProto()
 
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
 
    # Only allow a total of half the GPU memory to be allocated
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tensorflow.Session(config=config))
    print("available gpu divice: {}".format(tensorflow.test.gpu_device_name()))

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

# clear gpu
def clear_keras():
    sess = get_session()
    clear_session()
    sess.close()

    # config = tensorflow.ConfigProto(
    #     device_count = {'GPU': -1}
    # )
    # #sess = tensorflow.Session(config=config)
    # K.tensorflow_backend.set_session(tensorflow.Session(config=config))

def entropy_sampling(prediction):
    """ Diese Funktion rechnet die Entropie einer Prediction 
        für jedes Bilde wird einen list von Vorgesagtete class und ihre entsprechende Wahrscheinlichkeit
        [(class,prbability),....] 
    """
    argmax =0
    summe = 0
    for elt in prediction:
        pred= elt[1]/100
        if argmax<pred:
            argmax=pred
        summe = summe + np.log10(pred)*pred
    
    entropy = argmax-summe
    return entropy

def least_confident(prediction):
    """ Diese Funktion rechnet die Leas_confidence einer Prediction 
        für jedes Bilde wird einen list von Vorgesagtete class und ihre entsprechende Wahrscheinlichkeit
        [(class,prbability),....] 
    """
    argmax = 0
    for elt in prediction:
        pred= elt[1]/100
        if argmax<pred:
            argmax=pred
    
    lc = 1-argmax
    return lc


def margin_sampling (prediction):
    """ Diese Funktion rechnet die margin einer Prediction 
        für jedes Bilde wird einen list von Vorgesagtete class und ihre entsprechende Wahrscheinlichkeit
        [(class,prbability),....] 
    """
    prediction=sorted(prediction,key=itemgetter(1),reverse=True)
    argmax = 0
    mg=0
    mg_list = prediction[:2]
    for prob in mg_list:
        mg=prob[1]/100-mg

    return abs(mg)   

def berechnung_unsischerheit(prediction, methode):
    print("-----> berechnung_unsischerheit")
    unsischerheit = 0 
    if (methode=="entropie"):
        unsischerheit = entropy_sampling(prediction)
    elif (methode=="margin"):
        unsischerheit = margin_sampling(prediction)
    elif (methode=="least_confident"):
        unsischerheit = least_confident(prediction)
    else:
        print("kein gültiges Uncertainty sampling")
        unsischerheit=None
    
    return unsischerheit

def sort_list_sampling(list,sampling_methode):
    print("sortierung nach unsischerheit methode")
    if (sampling_methode=="entropie"):
        list = sorted(list,key=itemgetter(1),reverse=True)
    elif (sampling_methode=="margin"):
        list = sorted(list,key=itemgetter(1))
    elif (sampling_methode=="least_confident"):
        list = sorted(list,key=itemgetter(1),reverse=True)
    else:
        print("kein gültiges Uncertainty sampling")
    
    return list

def create_mapping_cls(list_traimingsmenge):
    """ Diese Funktion wird mapping für trainingsmenge erstellen"""
    cls_count={}
    cls_mapping={}
    for val in list_traimingsmenge:
        for bb in val['bboxes']:
            if bb['class'] not in cls_count:
                cls_count[bb['class']] = 1
            else:
                cls_count[bb['class']] += 1
            if bb['class'] not in cls_mapping:
                cls_mapping[bb['class']] = len(cls_mapping)    

    return cls_count,cls_mapping
 
def create_batchify_from_list(listdata,percentTotrain):
    sizetotrain = int(round((len(listdata)* percentTotrain)/100))
    number_of_batch = int(len(listdata)/sizetotrain)
    return [listdata[i::number_of_batch] for i in range(number_of_batch)]

def create_batch_from_path (pathToDataSet,percentTotrain):

    all_imgs, classes_count, class_mapping = get_data(pathToDataSet)

    sizetotrain = int(round((len(all_imgs)* percentTotrain)/100))
    number_of_batch = int(len(all_imgs)/sizetotrain)
   
    return [all_imgs[i::number_of_batch] for i in range(number_of_batch)]

def createSeed_pro_batch(batch_elt,train_size_pro_batch):
    """ Diese Funktion wird labellierte Data auswählen."""
    print("##### Erstellung von Datenmenge Seed und unlabellierte ####")
    seed_imgs=[]
    all_imgs =[]
    classes_count ={}
    class_mapping ={}
    #all_imgs, classes_count, class_mapping = get_data(pathToDataSet)
    sizetotrain = int(round((len(batch_elt)* train_size_pro_batch)/100))
    seed_imgs = batch_elt[:sizetotrain]
    all_imgs = batch_elt[sizetotrain:]

    print("Erstellung anotation, class_maping und class_count vom unlabellierte Daten")     
    for im in batch_elt:
        for bb in im['bboxes']:
            if bb['class'] not in classes_count:
                classes_count[bb['class']] = 1
            else:
                classes_count[bb['class']] += 1
            if bb['class'] not in class_mapping:
                class_mapping[bb['class']] = len(class_mapping)
    
    return all_imgs, seed_imgs, class_mapping,classes_count

@jit
def createSeedPascal_Voc(pathToDataSet,batch_size):
    """ Diese Funktion wird labellierte Data auswählen."""
    print("##### Erstellung von Datenmenge Seed und unlabellierte ####")
    seed_imgs=[]
    all_imgs =[]
    seed_classes_count={}
    seed_classes_mapping={}
    classes_count ={}
    class_mapping ={}
    all_imgs, classes_count, class_mapping = get_data(pathToDataSet)
    sizetotrain = int(round((len(all_imgs)* batch_size)/100))
    seed_imgs = all_imgs[:sizetotrain]
    all_imgs = all_imgs[sizetotrain:]
    print("Erstellung anotation, class_maping und class_count vom Seed")
    for seed in seed_imgs:
        for bb in seed['bboxes']:
            if bb['class'] not in seed_classes_count:
                seed_classes_count[bb['class']] = 1
            else:
                seed_classes_count[bb['class']] += 1
            if bb['class'] not in seed_classes_mapping:
                seed_classes_mapping[bb['class']] = len(seed_classes_mapping)
    return all_imgs, seed_imgs,seed_classes_mapping,seed_classes_count

def Datei_vorbereitung (seed,Dateitype):
    """ Erstellung von Matching file .txt zum training oder zum Testen
    """
    xml_list = []
    for xmlFile in seed:
        tree = ET.parse(xmlFile[1])
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,                     
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
		     int(member[4][3].text),
                     member[0].text
                     )
            xml_list.append(value)

    column_name = ['path', 'xmin', 'ymin', 'xmax', 'ymax','class']
    xml_df = pd.DataFrame(xml_list,columns=column_name)
    xml_df.to_csv((basePath +'/'+ Dateitype + '_labels.txt'), index=None, header=False)
    print('Successfully converted xml to txt.')

def Pool_based_sampling_test (predict_list, pool_size ,sampling_methode):
    """predict_list as list [(file_path,prediction)]
        pool_size integer
        sampling methode: entropie, marging or least confident
        Compute uncertainty for each prdiction
        return Pool to Query to Oracle
    """
    # computer
    Predict_uncertainty_list = []
    Predict_uncertainty_listsorted =[]
    for elt in predict_list:
        uncertainty = berechnung_unsischerheit(elt[1],sampling_methode)
        print("uncertainty of {} is {}".format(ntpath.basename(elt[0]),uncertainty))
        Predict_uncertainty_list.append((elt[0], elt[1],uncertainty))

    Predict_uncertainty_listsorted = sort_list_sampling(Predict_uncertainty_list,sampling_methode)
    if pool_size < len(Predict_uncertainty_listsorted):
        pool = Predict_uncertainty_listsorted[:pool_size]
    else:
        pool = Predict_uncertainty_listsorted
    return pool

def appendDFToCSV_void(dictPerformance, csvFilePath):
    df = pd.DataFrame(dictPerformance, index=[0])
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=";")
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=";").columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=";").columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=";").columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=";", header=False)

def writePerformanceModell(ModellParmeter, pathtofile):
        Perform = pd.DataFrame(ModellParmeter, index=[0])
        Perform.to_csv(pathtofile, sep=';', mode='a', header=False, index=False)

def check_predict(list_pred):
    all_bg = False
    list_found_bg=0
    list_not_bg =[]
    for el in list_pred:
        if el[0] == 'bg':
            list_found_bg+=1
        else:
            list_not_bg.append(el)

    if len(list_pred)== list_found_bg:
        all_bg = True

    return all_bg,list_not_bg

def update_config_file(pathtofile,con):
    print("update config file")
    with open(pathtofile, 'wb') as config_f:
        pickle.dump(con, config_f)  
    return con
if __name__ == "__main__":

    #base_path=os.getcwd()
    #test_path='/home/kamgo/test_image'
    #pathToPermformance = os.path.join(base_path, 'performance/performance.csv')
    #print(pathToPermformance)
    #performamce ={'unsischerheit_methode':5,'Iteration':3,'Aktuelle Ungenaueheit':3,'abgelaufene Zeit':62,'Anzahl der vorhergesagteten Bildern':6,'Gut predicted':8}
    #appendDFToCSV_void(performamce,pathToPermformance)
    d1=[]
    d2=[]
    list_predict1 =[('bg', 0.9), ('bg', 0.09), ('bg', 0.01)]
    en1 = entropy_sampling(list_predict)
    lc1 = least_confident(list_predict)
    m1 = margin_sampling(list_predict)
    d1.append(()) 
    list_predict =[('bg', 0.2), ('bg', 0.5), ('bg', 0.3)]
    en = entropy_sampling(list_predict)
    lc = least_confident(list_predict)
    m = margin_sampling(list_predict)

    print("entropie is :{} least confident is {} and margin is {}".format(en,lc,m))
    #print(len(lt))

