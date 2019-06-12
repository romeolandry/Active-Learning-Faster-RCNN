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

def readSeed_Simple():
    """ Diese Funktion wird labellierte Data auswählen."""
        # Eingabe:
                # pathToDataSet Pfad zu den gesamten Bilddateien
                # pathToSeed: pfad zu den Bildverzeichnis, das für das training des Models benuzt wird
                # imageType: Bildsart zu trainieren
                # anotationType: Dateitipp zur Annotation 
        # Verarbeitung:Es wurde datei in train Verzeichnis des Models (Model/research/object_detection/images/train)
        # Ausgabe: cvs :datei für tensorflow api
    seed=[]
    Unlabelierte=[]
    if pathToSeed ==None:
        pathImg = pathToDataSet + str('/*.') + imageType
        pathAnotattion = pathToDataSet + str('/*.')+ anotationType 
        imgFiles = glob.glob(pathImg)
        anotationFiles = glob.glob(pathAnotattion)

        sorted(imgFiles,reverse=True)
        sorted(anotationFiles,reverse=True)
        count= round(len(anotationFiles)*10/100)
        i=0;
        for bild in zip(imgFiles,anotationFiles):
            if (i<count):
                seed.append(bild)
            else:
                Unlabelierte.append(bild)
            i=i+1

        """ for img in imgFiles:
            for anot in anotationFiles:
                if (ntpath.basename(os.path.splitext(anot)[0])== ntpath.basename(os.path.splitext(img)[0])):
                    seed.append(img)
                    seed.append(anot)
   
        for data in seed:
            filename= os.path.basename(data)
            copyfile(data,os.path.join(pathtotrain,filename))
     """
    else:
        pathImg = pathToDataSet + str('/*.') + imageType
        pathAnotattion = pathToDataSet + str('/*.')+ anotationType 
        imgFiles = glob.glob(pathImg)
        anotationFiles = glob.glob(pathAnotattion)   
        #imgFiles = imgFiles.sort()
        #anotationFiles=anotationFiles.sort()

        for bild in zip(imgFiles,anotationFiles):
            Unlabelierte.append(bild)        

        pathImgSeed = pathToSeed + str('/*.') + imageType
        pathAnotattionSeed = pathToSeed + str('/*.')+ anotationType 
        imgFilesseed = glob.glob(pathImgSeed)
        anotationFilesseed = glob.glob(pathAnotattionSeed)
        #imgFilesseed = imgFiles.sort()
        #anotationFilesseed=anotationFiles.sort()
        for bild in zip(imgFilesseed,anotationFilesseed):
            seed.append(bild)

    return seed,Unlabelierte

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
    seed_imgs = all_imgs[:batch_size] + seed_imgs
    all_imgs = all_imgs[batch_size:]
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

if __name__ == "__main__": 
    pass