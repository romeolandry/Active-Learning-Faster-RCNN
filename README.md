# Activ Learning Obeject detection
Keras Implementierung vom aktiven Lernnen in Objekt Erkennung: Ziel  dieses projekts ist Aktives Lernenverfahren zur Detection des Objekts zur Anwendung zu bringen 
Es wurde ein Keras Faster-RCNN (keras-frcnn) benuzt,die wurde geclont von https://github.com/kbardool/keras-frcnn

USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.
- `active_learning_modul.py` can be used to train a model. To train on Pascal VOC data,
    - python active_learning_modul.py argv1 argv2 argv3 agrv4 
    - argv1 = /media/romeo/Volume/dataset/VOCtrainval_11-May-2012/VOCdevkit corresponding to path to VOC2012 directory
    - argv2 = name of perfomance csv file
    - argv3 = name of model to save
    - argv4 = train on batch/simple cloud be simple or batch default is simple   

# test 
 to  test:
    - python AL_test.py -p patho to directory file -c path to .pickle file

# Issue
- "tensorflow.python.framework.errors_impl.InvalidArgumentError: Shape must be rank 1 but is rank 0 for 'bn_conv1/Reshape_4' (op: 'Reshape') with input shapes:     [1,1,1,64], []."
    - cause
        keras version 2.2.4
    - solution
        (-1) to [-1] in file "tensorflow_backend.py" on line 1908,1910, 1914, 1918