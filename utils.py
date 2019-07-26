import sys
sys.path.append("/home/kamgo/activ_lerning _object_dection/keras-frcnn")
from operator import itemgetter
from keras_frcnn.pascal_voc_parser import get_data
import os
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

def entropy_sampling(prediction):
    """ Diese Funktion rechnet die Entropie einer Prediction 
        für jedes Bilde wird einen list von Vorgesagtete class und ihre entsprechende Wahrscheinlichkeit
        [(class,prbability),....] 
    """
    argmax = 0
    summe = 0
    for elt in prediction:
        if argmax<elt[1]:
            argmax=elt[1]
        summe = summe + np.log10(elt[1])*elt[1]
    
    entropy = argmax-summe
    return entropy

def least_confident(prediction):
    """ Diese Funktion rechnet die Leas_confidence einer Prediction 
        für jedes Bilde wird einen list von Vorgesagtete class und ihre entsprechende Wahrscheinlichkeit
        [(class,prbability),....] 
    """
    argmax = 0
    for elt in prediction:
        if argmax<elt[1]:
            argmax=elt[1]
    
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
        mg=prob[2]-mg

    return abs(mg)   

def berechnung_unsischerheit(prediction, methode):
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

def create_batchify_from_path (pathToDataSet,percentTotrain):

    all_imgs, classes_count, class_mapping = get_data(pathToDataSet)

    sizetotrain = int(round((len(all_imgs)* percentTotrain)/100))
    nmber_of_batch = int(len(all_imgs)/sizetotrain)
   
    return [all_imgs[i::nmber_of_batch] for i in range(nmber_of_batch)],classes_count,class_mapping

def createSeed_pro_batch(batch_elt,classes_count,class_mapping,train_size_pro_batch):
    """ Diese Funktion wird labellierte Data auswählen."""
    print("##### Erstellung von Datenmenge Seed und unlabellierte ####")
    seed_imgs=[]
    all_imgs =[]
    seed_classes_count={}
    seed_classes_mapping={}
    classes_count ={}
    class_mapping ={}
    #all_imgs, classes_count, class_mapping = get_data(pathToDataSet)
    sizetotrain = int(round((len(batch_elt)* train_size_pro_batch)/100))
    seed_imgs = batch_elt[:sizetotrain]
    all_imgs = batch_elt[sizetotrain:]
    print("Erstellung anotation, class_maping und class_count vom Seed")
    for seed in seed_imgs:
        for bb in seed['bboxes']:
            if bb['class'] not in seed_classes_count:
                seed_classes_count[bb['class']] = 1
            else:
                seed_classes_count[bb['class']] += 1
            if bb['class'] not in seed_classes_mapping:
                seed_classes_mapping[bb['class']] = len(seed_classes_mapping)
    print("Erstellung anotation, class_maping und class_count vom unlabellierte Daten")     
    for im in batch_elt:
        for bb in im['bboxes']:
            if bb['class'] not in classes_count:
                classes_count[bb['class']] = 1
            else:
                classes_count[bb['class']] += 1
            if bb['class'] not in class_mapping:
                class_mapping[bb['class']] = len(class_mapping)
    
    return all_imgs, seed_imgs, class_mapping,classes_count,seed_classes_mapping,seed_classes_count


def createSeedPlascal_Voc(pathToDataSet,batch_size):
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

def Pool_based_sampling_test (listeImage,listscore):
    imagepostclassifier=[]
    column_name = ['path', 'boxes', 'labels', 'score']
    # pool_based sampling scenario and as query strategies I will implement Least confidence(Lc): 
    for img in listeImage:
        # load image
        image = read_image_bgr(img)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # save into list
        imagepostclassifier.append(img,boxes[0],labels[0],scores[0])

    df_imgcls = pd.DataFrame(imagepostclassifier,columns=column_name)
    df_imgcls.sort_values(by=['scores'])
    return df_imgcls

def appendDFToCSV_void(dictPerformance, csvFilePath):
    df = pd.DataFrame(dictPerformance, index=[0])
    if not os.path.isfile(csvFilePath):
        print("saved file")
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
        

if __name__ == "__main__":
    base_path=os.getcwd()
    #test_path='/home/kamgo/test_image'
    pathToPermformance = os.path.join(base_path, 'performance/performance.csv')
    print(pathToPermformance)
    performamce ={'unsischerheit_methode':5,'Iteration':3,'Aktuelle Ungenaueheit':3,'abgelaufene Zeit':62,'Anzahl der vorhergesagteten Bildern':6,'Gut predicted':8}
    appendDFToCSV_void(performamce,pathToPermformance)  
