# coding:utf-8
import tensorflow as tf
from keras import backend as K


##############################
###     LOSS FUNCTION    #####
##############################
def sigmoid(x):
    result = 1 / (1 + K.exp(-x))
    return result


def AUC(y_true, y_pred):
    auc, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        auc = tf.identity(auc)
    return auc


# Define IoU metric
def mean_iou(y_true, y_pred):
    th = 0.7
    y_pred_ = tf.to_int32(y_pred > th)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def mean_iou_softmax(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred_ = tf.to_int32(y_pred)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def dice_coefficient(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def dice_coef_metrics(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.to_float(y_pred_f > 0.7)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss


def focal_loss(y_true, y_pred):
    # Hyper parameters
    gamma = 1  # For hard segmentation
    alpha = 0.9  # For unbalanced samples

    # Original focal loss for segmentation
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    loss_0 = -K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())
    loss_1 = -K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)

    sum_0 = tf.reduce_sum(loss_0)
    sum_1 = tf.reduce_sum(loss_1)

    loss_0 = tf.Print(loss_0, [sum_0, sum_1], message='loss_0 loss_1:')
    loss = alpha * loss_1 + (1 - alpha) * loss_0

    # Original focal loss for classification
    # loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
    #     (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    # Self-defined focal loss
    # loss = -alpha * y_true * K.pow(1.0 - y_pred, gamma) * K.log(K.epsilon() + y_pred) - (1.0 - y_true) * (
    # 1 - alpha) * K.pow(y_pred, gamma) * K.log(1. - y_pred + K.epsilon())

    return K.mean(loss)


def self_balancing_focal_loss(y_true, y_pred):
    # Hyper parameters
    
    gamma = 1  # For hard segmentation

    # Original focal loss for segmentation
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    loss_0 = -K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())
    loss_1 = -K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)
    sum_0 = tf.reduce_sum(loss_0)
    sum_1 = tf.reduce_sum(loss_1)

    alpha = (sum_0 / (sum_0 + sum_1)) * 0.4 + 0.5
    b_loss_0 = (1 - alpha) * loss_0  # background loss after balance
    b_loss_1 = alpha * loss_1  # foreground loss after balance
    sum_2 = tf.reduce_sum(b_loss_0)
    sum_3 = tf.reduce_sum(b_loss_1)

    b_loss_1 = tf.Print(b_loss_1, [sum_0, sum_1, alpha, sum_2, sum_3],
                        message='loss_0 loss_1 alpha sum2 sum3:')

    loss = b_loss_0 + b_loss_1
    return loss
