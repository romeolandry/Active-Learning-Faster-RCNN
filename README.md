# Activ Learning Obeject detection
Keras Implementierung vom aktiven Lernnen in Objekt Erkennung: Ziel  dieses projekts ist Aktives Lernenverfahren zur Detection des Objekts zur Anwendung zu bringen 
Es wurde ein Keras Faster-RCNN (keras-frcnn) benuzt,die wurde geclont von https://github.com/kbardool/keras-frcnn

USAGE:
- Both theano and tensorflow backends are supported. However compile times are very high in theano, and tensorflow is highly recommended.
- `active_learning_modul.py` can be used to train a model. To train on Pascal VOC data,