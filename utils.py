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
    print("len all_image after get_data in {}".format(len(all_imgs)))
    sizetotrain = int(round((len(all_imgs)* batch_size)/100))
    seed_imgs = all_imgs[:sizetotrain]
    all_imgs = all_imgs[sizetotrain:]
    print("len all_image after get train_menge {}".format(len(all_imgs)))
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

    return all_bg,list_not_bg,list_found_bg

def update_config_file(pathtofile,con):
    print("update config file")
    with open(pathtofile, 'wb') as config_f:
        pickle.dump(con, config_f)  
    return con

def count_class_in_dict_oracle (classname, dict_oracle, list_prediction, match_list):
    count_in_dict_orcal =0
    count_in_prediction_list =0
    truePositiv = 0
    trueNegativ = 0
    found = False
    for c in match_list:
        if c == classname:
            found = True
            break
    if found == False:
        match_list.append(classname)
        for cl in dict_oracle['bboxes']:
                if cl['class']== classname:
                    count_in_dict_orcal +=1
        for val in list_prediction:                   
            if classname == val[0]:
                count_in_prediction_list +=1
    if count_in_prediction_list != 0 :
        check = count_in_prediction_list - count_in_dict_orcal
        if check == 0:
            truePositiv = count_in_dict_orcal
            trueNegativ = 0
        else:
            trueNegativ = abs(check)
            if count_in_prediction_list < count_in_dict_orcal:
                truePositiv = count_in_prediction_list
            else:
                truePositiv = count_in_dict_orcal

    return truePositiv,trueNegativ,match_list

    



if __name__ == "__main__":

    prediction = [('tvmonitor', 100.0), ('tvmonitor', 100.0), ('person', 100.0), ('person', 100.0), ('tvmonitor', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0), ('bg', 100.0)] 
    oracle ={'filepath': '/mnt/0CCCB718CCB6FB52/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2010_005372.jpg', 'width': 375, 'height': 500, 'bboxes': [{'class': 'chair', 'x1': 66, 'x2': 225, 'y1': 311, 'y2': 500, 'difficult': False}, {'class': 'tvmonitor', 'x1': 157, 'x2': 242, 'y1': 236, 'y2': 318, 'difficult': False}], 'imageset': 'trainval'}  
    all_bg,list_not_bg,list_found_bg = check_predict(prediction)
    print(all_bg)
    print(list_not_bg)
    truePositiv_all =0
    trueNegativ_all = 0
    match_list =[]
    for val in list_not_bg:    
        for cl in oracle['bboxes']:                  
            print("oracle: die Liste Von Objekt auf dem Bild: {} ".format(cl['class']))
            truePositiv,trueNegativ,match_list= count_class_in_dict_oracle(val[0],oracle,list_not_bg,match_list)
            truePositiv_all = truePositiv_all+truePositiv
            trueNegativ_all = trueNegativ_all + trueNegativ
            print("{} : {} : {} ".format(truePositiv,trueNegativ,match_list))

            """ if cl['class']== val[0]:
                print("gut predicted!")
                print("das Model predict : {} mit {} % Sicherheit".format(val[0],val[1]))
                truePositiv +=1
            else:
                print("falsch Predict")
                trueNegativ += 1
                print("das Model predict : {} mit {} % Sicherheit".format(val[0],val[1])) """

    print("true positive:{} true negativ: {}  bd {}/{} ".format(truePositiv_all, trueNegativ_all, list_found_bg,len(prediction)))

