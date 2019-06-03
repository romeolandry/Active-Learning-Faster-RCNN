# -*- coding: utf-8 -*-
# actives Lernnen mit "Pool-based-Sampling"

"""Main module."""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import errno
import ntpath
from shutil import copyfile

def readSeed(pathtoSeed,pathtotrain,imgtype,anotationType):
    """ Diese Funktion wird labellierte Data auswählen. es wurde dafür einen Batch eingestellt,
        damit nicht alle Data im Seed ausgewählt seien sonder einen Menge (Batch size)."""
        # Eingabe: Path to Seed, Batch size
        # Verarbeitung:Es wurde datei in train Verzeichnis des Models (Model/research/object_detection/images/train)
        # Ausgabe: cvs :datei für tensorflow api

    pathImg = pathtoSeed + str('/*.') + imgtype
    pathAnotattion = pathtoSeed + str('/*.')+ anotationType 
    imgFiles = glob.glob(pathImg)
    anotationFiles = glob.glob(pathAnotattion)
    seed=[]
    for img in imgFiles:
        for anot in anotationFiles:
            if (ntpath.basename(os.path.splitext(anot)[0])== ntpath.basename(os.path.splitext(img)[0])):
                seed.append(img)
                seed.append(anot)

    for data in seed:
        filename= os.path.basename(data)
        copyfile(data,os.path.join(pathtotrain,filename))
 
""" def cross_validaion (pathtotrain,prozentTes,imgtype,anotationType):

    pathImg = pathtotrain + str('/*.') + imgtype
    pathAnotattion = pathtotrain + str('/*.')+ anotationType 
    imgFiles = glob.glob(pathImg)
    anotationFiles = glob.glob(pathAnotattion)
    imgFiles.sort()
    anotationFiles.sort()
    
    for img, anot in zip(imgFiles,anotationFiles):
        pass """

def Pool_based_sampling (listeImage,listscore):
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