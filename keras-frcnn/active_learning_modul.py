# -*- coding: utf-8 -*-
# actives Lernnen mit "Pool-based-Sampling"
import sys
sys.path.append("/home/kamgo/activ_lerning _object_dection/keras-frcnn")
import train_frcnn as train
import test_frcnn as test
from keras import backend as K
from keras_frcnn.pascal_voc_parser import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import errno
import ntpath
from shutil import copyfile,move
import xml.etree.ElementTree as ET
import subprocess
#Zum predict 
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_frcnn import config, data_generators
import argparse
import pickle
import cv2
from operator import itemgetter

base_path='/home/kamgo/activ_lerning _object_dection/keras-frcnn'
#pathToDataSet='/home/kamgo/midras/keras-frcnn/test_images'
pathToDataSet = '/media/kamgo/15663FCC59194FED/Activ Leaning/dataset/VOCtrainval_11-May-2012/VOCdevkit'
pathToSeed = '/home/kamgo/activ_lerning _object_dection/keras-frcnn/train_images' # pfad zum Seed: labellierte Datein, die zum training benutzen werden
#pathToSeed =None
imageType = 'jpg'
anotationType = 'xml'
batch_size = 50 # batch size for pool based simple
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
base_weight_path = os.path.join(base_path, 'models/resnet50_coco_best_v2.0.1.h5') #Input path for weights. If not specified, will try to load default weights provided by keras.
config_output_filename = os.path.join(base_path, 'model_resnet_config.pickle') #Location to store all the metadata related to the training (to be used when testing).
num_epochs = 1000
parser = 'simple' # kann pascal_voc oder Simple(für andere Dataset)
num_rois = 32 # Number of RoIs to process at once.
network = 'resnet50'# Base network to use. Supports vgg or resnet50

def entropy_sampling(img_path,prediction):
    list_entropy =[]
    argmax = 0
    summe = 0
    for elt in prediction:
        if argmax<elt[2]:
            argmax=elt[2]
        summe = summe + np.log10(elt[2])*elt[2]
    
    entropy = argmax-summe
    list_entropy.append((img_path,entropy))
    return list_entropy

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

def createSeedPlascal_Voc():
    """ Diese Funktion wird labellierte Data auswählen."""
    seed_imgs=[]
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

def train_vorbereitung ():
    con = config.Config()

    con.use_horizontal_flips = bool(horizontal_flips)
    con.use_vertical_flips = bool(vertical_flips)
    con.rot_90 = bool(rot_90)

    con.model_path = output_weight_path
    con.num_rois = int(num_rois)

    if network == 'vgg':
        con.network='vgg'
        from keras_frcnn import vgg as nn 
    elif network == 'resnet50': 
        from keras_frcnn import resnet as nn
        con.network = 'resnet50'
    elif network == 'xception':
        from keras_frcnn import xception as nn
        con.network = 'xception'
    elif network == 'inception_resnet_v2':
        from keras_frcnn import inception_resnet_v2 as nn
        con.network = 'inception_resnet_v2'
    else:
        print('Not a valid model')
        raise ValueError
    # loard weight 
    con.base_net_weights = base_weight_path

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(con, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
  
    return con

def Test_model():
    con = config.Config()
    config_output_filename = config_filename

    with open(config_output_filename, 'rb') as f_in:
        con = pickle.load(f_in)

    if con.network == 'resnet50':
        import keras_frcnn.resnet as nn
    elif con.network == 'vgg':
        import keras_frcnn.vgg as nn

    # turn off any data augmentation at test time
    con.use_horizontal_flips = False
    con.use_vertical_flips = False
    con.rot_90 = False
    img_path = test_path
    return test_path,con

def make_prediction():
    #load model
    con = train_vorbereitung()
    print
    #model = ResNet50(weights= con.model_path)
    model = ResNet50(weights='imagenet')
    # predicting images
    list_pfad_imgs=[]
    for img in all_imgs:
        #print(img)
        list_pfad_imgs.append(img['filepath'])
    
    for filepath in list_pfad_imgs:
        img = cv2.imread(filepath)
        X, ratio = test.format_img(img, con)
        
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        preds = model.predict(X)
        #gitprint('Predicted:', decode_predictions(preds, top=3)[0])
        prediction = decode_predictions(preds, top=3)[0]
        list_pool_sampling = entropy_sampling(filepath,prediction)
        print(list_pool_sampling)

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

def Pool_based_sampling (listeImage,listscore):
    """
        Es wude Entropy Sampling Strategie angewendet 
    """
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


def Oracle():
    pass

def addNewImageToSeed (seed,newselection):
    pass

if __name__ == "__main__":
    ##seed,unL =readSeed()
    #Datei_vorbereitung(seed,'seed')
    #train_model()
    
    #createSeedPlascal_Voc(seed_img,batch_size,all_imgs,classes_count,class_mapping)
    all_imgs,seed_imgs,class_mapping,classes_count,seed_classes_mapping,seed_classes_count=createSeedPlascal_Voc()
    #train.train_model(seed_imgs,seed_classes_count,seed_classes_mapping)
    #print(train_vorbereitung())
    make_prediction()
    
