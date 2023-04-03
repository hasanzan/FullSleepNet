import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def cat_crossentropy_cut(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
    mask = K.not_equal(tf.argmax(y_true, axis=-1), 5)
    loss = K.categorical_crossentropy(y_true[:,:-1], y_pred)
    loss = tf.boolean_mask(loss, mask)
    masked_loss = tf.reduce_mean(loss)
    return masked_loss

def binary_w(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
    mask = K.not_equal(tf.argmax(y_true, axis=-1), 5)
    loss = K.categorical_crossentropy(y_true[:,:-1], y_pred)
    loss = tf.boolean_mask(loss, mask)
    masked_loss = tf.reduce_mean(loss)
    return masked_loss

def AUPRC(y_true, y_pred):
    # flatten tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    # filter out non-arousals
    idx = tf.where(y_true != -1.)
    y_true = tf.gather(y_true, idx)
    y_pred = tf.gather(y_pred, idx)
    
    y_true = tf.concat([y_true, [[1.], [0.]]], axis=0)
    y_pred = tf.concat([y_pred, [[1.], [0.]]], axis=0)
    
    y_true = tf.cast(y_true, tf.int32)
    return tf.py_function(average_precision_score, inp=[y_true, y_pred], Tout=tf.float32)

def AUROC(y_true, y_pred):
    # flatten tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    # filter out non-arousals
    idx = tf.where(y_true != -1.)
    y_true = tf.gather(y_true, idx)
    y_pred = tf.gather(y_pred, idx)
    
    y_true = tf.concat([y_true, [[1.], [0.]]], axis=0)
    y_pred = tf.concat([y_pred, [[1.], [0.]]], axis=0)
    
    y_true = tf.cast(y_true, tf.int32)
    return tf.py_function(roc_auc_score, inp=[y_true, y_pred], Tout=tf.float32)

def cat_AUPRC(inputs, y_pred):
    label1 = inputs[:,:,0]
   
    return tf.py_function(AUPRC, inp=[label1, y_pred], Tout=tf.float32)

def cat_AUROC(inputs, y_pred):
    label1 = inputs[:,:,0]    
    return tf.py_function(AUROC, inp=[label1, y_pred], Tout=tf.float32)