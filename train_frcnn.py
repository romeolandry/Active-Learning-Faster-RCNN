from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import keras
import utils

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.set_random_seed(2000)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)

print("available gpu divice: {}".format(tf.test.gpu_device_name()))

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

# tensorboard
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def train_model(seed_data, classes_count, class_mapping, con,best_loss):
    sys.setrecursionlimit(40000)
    #utils.reset_keras()
    from keras_frcnn import losses as losses    
    
    if con.network == 'vgg':
        from keras_frcnn import vgg as nn 
    elif con.network == 'resnet50': 
        from keras_frcnn import resnet as nn
    elif con.network == 'xception':
        from keras_frcnn import xception as nn
    elif con.network == 'inception_resnet_v2':
        from keras_frcnn import inception_resnet_v2 as nn
    else:
        print('Not a valid model')
        raise ValueError

    # bg 
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    con.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}


    random.shuffle(seed_data)

    num_imgs = len(seed_data)
    #train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    train_imgs = [s for s in seed_data if s['imageset'] == 'trainval']
    val_imgs = [s for s in seed_data if s['imageset'] == 'val']
    test_imgs = [s for s in seed_data if s['imageset'] == 'test']
    
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    print('Num test samples {}'.format(len(test_imgs)))
   
    # groundtruth
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, con, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, con, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
    data_gen_test = data_generators.get_anchor_gt(test_imgs, classes_count, con, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    # input placeholder 
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # base network(feature extractor)(resnet, VGG, Inception, Inception Resnet V2, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    # RPN 
    num_anchors = len(con.anchor_box_scales) * len(con.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    # detection network
    classifier = nn.classifier(shared_layers, roi_input, con.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    try:
        # load_weights by name
        # some keras application model does not containing name
        # for this kinds of model, we need to re-construct model with naming
        print('loading weights from {}'.format(con.base_net_weights))
        model_rpn.load_weights(con.base_net_weights, by_name=True)
        model_classifier.load_weights(con.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # Tensorboard log
    log_path = './logs'
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    # Tensorboard log
    callback = TensorBoard(log_path)
    callback.set_model(model_all)

    epoch_length = 50
    num_epochs = int(con.num_epochs)
    iter_num = 0
    train_step = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    
    # early stopping 
    #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    patience = 50
    change = 0
    
    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    # vis = True

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)   # keras progress bar 
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            # try:
            # mean overlapping bboxes
            if len(rpn_accuracy_rpn_monitor) == epoch_length and con.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # data generator에서 X, Y, image 
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], con, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, con, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            # sampling positive/negative samples
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if con.num_rois > 1:
                if len(pos_samples) < con.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, con.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, con.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, con.num_rois - len(selected_pos_samples), replace=True).tolist()
                    except:
                        # The neg_samples is [[1 0 ]] only, therefore there's no negative sample
                        continue
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)
            train_step += 1

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                    ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if con.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                        ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                        'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                        [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                        loss_class_cls, loss_class_regr, class_acc, curr_loss],
                        epoch_num)

                if curr_loss < best_loss:
                    if con.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    con.best_loss = best_loss
                    model_all.save_weights(con.model_path)
                    change = patience
                else:
                    change += 1
                break
        if patience == change:
            print("training stopped by early stopping")
            break           
                

    print('Training complete, exiting.')
    return best_loss,con