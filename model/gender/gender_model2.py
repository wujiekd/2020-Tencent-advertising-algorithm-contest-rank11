#age_version2
#输入特征：w2v,tfidf,count
#网络结构：单层lstm

# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.
import os
import pandas as pd
from tqdm.autonotebook import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from gensim.models import FastText, Word2Vec
import re
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.optimizers import *
from keras.utils import to_categorical
import keras
import tensorflow as tf
import random as rn
import gc
import logging
import numpy as np
import gensim
from numpy import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

# 得到emb矩阵
emb1 = np.load('emb/embcreative_id_256.npy')
print(emb1.shape)
emb2 = np.load('emb/embad_id_256.npy')
print(emb2.shape)
emb3 = np.load('emb/embadvertiser_id_128.npy')
print(emb3.shape)
emb4 = np.load('emb/embproduct_id_128.npy')
print(emb4.shape)
emb5 = np.load('emb/embindustry_64.npy')
print(emb5.shape)
gc.collect()



# 需要用到的函数
class AdamW(Optimizer):


    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        #super(AdamW, self).__init__(**kwargs)
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras.engine.topology import Layer
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def model_conv(emb1, emb2,emb3,emb4,emb5,num_feature_input):
    '''
    注意这个inputs
    seq1、seq2分别是两个输入
    hin是feature层输入
    是否做emb可选可不选，
    这个就是我们之前训练已经得到的用于embedding的（embedding_matrix1， embedding_matrix2）
    '''
    K.clear_session()

    emb_layer_1 = Embedding(
        input_dim=emb1.shape[0],
        output_dim=emb1.shape[1],
        weights=[emb1],
        input_length=100,
        trainable=False
    )

    emb_layer_2 = Embedding(
        input_dim=emb2.shape[0],
        output_dim=emb2.shape[1],
        weights=[emb2],
        input_length=100,
        trainable=False
    )
    
    emb_layer_3 = Embedding(
        input_dim=emb3.shape[0],
        output_dim=emb3.shape[1],
        weights=[emb3],
        input_length=100,
        trainable=False
    )

    emb_layer_4 = Embedding(
        input_dim=emb4.shape[0],
        output_dim=emb4.shape[1],
        weights=[emb4],
        input_length=100,
        trainable=False
    )
    
    emb_layer_5 = Embedding(
        input_dim=emb5.shape[0],
        output_dim=emb5.shape[1],
        weights=[emb5],
        input_length=100,
        trainable=False
    )

    
    seq1 = Input(shape=(100,))
    seq2 = Input(shape=(100,))
    seq3 = Input(shape=(100,))
    seq4 = Input(shape=(100,))
    seq5 = Input(shape=(100,))

    x1 = emb_layer_1(seq1)
    x2 = emb_layer_2(seq2)
    x3 = emb_layer_3(seq3)
    x4 = emb_layer_4(seq4)
    x5 = emb_layer_5(seq5)
    
    
    sdrop = SpatialDropout1D(rate=0.2) #某个区域全部置0，就是某一列全部置0

    
    x11 = Concatenate()([x1,x2,x3,x5,x4])  
    
    x = Dropout(0.35)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x11))#双向LSTM
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    semantic = TimeDistributed(Dense(128, activation="relu"))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    input_mask = Input(shape=(1,))
    merged_1_avg = Lambda(lambda x: K.sum(x, axis=1)/input_mask, output_shape=(128,))(semantic)
    
   
    hin = Input(shape=(num_feature_input,))
    htime = Dense(128, activation='relu')(hin)
    htime = Dense(64, activation='relu')(htime)
    x = Concatenate()([merged_1,merged_1_avg,htime]) 
    
    x = Dropout(0.4)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    pred = Dense(2, activation='softmax')(x)
    model = Model(inputs=[seq1,seq2,seq3,seq4,seq5,input_mask,hin], outputs=pred)
    from keras.utils import multi_gpu_model
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(lr=3e-4, weight_decay=0.08, ), metrics=["accuracy"])# AdamW(lr=0.001, weight_decay=0.08, ), metrics=["accuracy"])
    return model


gc.collect()
def label_smoothing(inputs, epsilon=0.5):
    K = inputs.shape[1]    # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

def h5_generator(inputPath, tfidfinputPath,countinputPath,all_num , batch_size, mode="train"):
    f = h5py.File(inputPath, "r")
    f2 = h5py.File(tfidfinputPath, "r")
    f3 = h5py.File(countinputPath, "r")
    all_sum = all_num/batch_size
    flag = 0-batch_size
    beta = 0.2
    #augu =1
    while True:
        
        flag += batch_size
        if(flag==all_num):
            flag=0  #切换下一个epoch
            #augu+=1 #3代后进行数据增强
        labels = []
        input1 = f['cre'][flag:flag+batch_size]
        input2 = f['ad'][flag:flag+batch_size]
        input3 = f['adv'][flag:flag+batch_size]
        input4 = f['pro'][flag:flag+batch_size]
        input5 = f['ind'][flag:flag+batch_size]
        
        input6 = f2['tfidf'][flag:flag+batch_size]
        input7 = f3['count'][flag:flag+batch_size]
        
        input8 = np.hstack((input6,input7))
        
        input_mask  = np.zeros(batch_size)
        
        if mode=='train':
            #if augu>=3:
            for i in range(batch_size):
                input_mask[i]=lens = len(np.nonzero(input1[i])[0])
                num = np.random.rand()
                if num > beta:
                    state = np.random.get_state()
                    np.random.shuffle(input1[i][100-lens:100])
                    np.random.set_state(state)
                    np.random.shuffle(input2[i][100-lens:100])
                    np.random.set_state(state)
                    np.random.shuffle(input3[i][100-lens:100])
                    np.random.set_state(state)
                    np.random.shuffle(input4[i][100-lens:100])
                    np.random.set_state(state)
                    np.random.shuffle(input5[i][100-lens:100])
            
            indexs = f['gender'][flag:flag+batch_size]
            labels = to_categorical(indexs, num_classes=2) #one-hot
                    
            #labels = label_smoothing(labels) #smoothing
        if mode=='vail':
            for i in range(batch_size):
                input_mask[i]= len(np.nonzero(input1[i])[0])
            indexs = f['gender'][flag:flag+batch_size]
            labels = to_categorical(indexs, num_classes=2) #one-hot
            
        
        yield ([input1, input2,input3,input4,input5,input_mask,input8], labels)


train_vail = np.zeros((600000, 2))
sub = np.zeros((1000000, 2))
score = []
if not os.path.exists("./model/gender_model_2"):
    os.mkdir("./model/gender_model_2")


num =5
for i in range(num):
    print("FOLD | ", i+1)
    print("###"*35)
    gc.collect()
    filepath = f"./gender_model_2/nn_v1_{i}.h5"
    filepath_tr = f"./w2v_feature/nn_feature{i}/train{i}.h5"
    filepath_te = f"./w2v_feature/nn_feature{i}/vail{i}.h5"
    
    filepath_tr_tf = f"./tfidf/nn_feature{i}/train{i}.h5"
    filepath_te_tf = f"./tfidf/nn_feature{i}/vail{i}.h5"
    
    filepath_tr_co = f"./count/nn_feature{i}/train{i}.h5"
    filepath_te_co = f"./count/nn_feature{i}/vail{i}.h5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=2, min_lr=0.00001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.00001, patience=5, verbose=1, mode='max')
    callbacks = [checkpoint, reduce_lr, earlystopping]
    model_age = model_conv(emb1,emb2,emb3,emb4,emb5,252*2)
    #if count==0:
    model_age.summary()

    train_batch = 1000 #1024
    vail_batch = 5000
    test_batch = 5000
    trainGen = h5_generator(inputPath=filepath_tr,tfidfinputPath=filepath_tr_tf,countinputPath=filepath_tr_co ,all_num = 2400000, batch_size=train_batch, mode="train")
    vailGen = h5_generator(inputPath=filepath_te,tfidfinputPath=filepath_te_tf,countinputPath=filepath_te_co ,all_num=600000, batch_size=vail_batch, mode="vail")
    testGen = h5_generator(inputPath='w2v_feature/test.h5',tfidfinputPath='tfidf/test.h5', countinputPath='count/test.h5',all_num=1000000, batch_size=test_batch, mode="test")


    steps_epoch = 2400000//train_batch
    vail_steps_epoch = 600000//vail_batch
    test_epoch = 1000000//test_batch
    epochs =60
    hist = model_age.fit_generator(trainGen,steps_per_epoch=steps_epoch,epochs=epochs,
                               verbose=1, use_multiprocessing=True,callbacks=callbacks,validation_data=vailGen, validation_steps=vail_steps_epoch)
    sub += model_age.predict_generator(testGen, verbose=1,steps=test_epoch)
    train_vail = model_age.predict_generator(vailGen, verbose=1,steps=vail_steps_epoch)
    score.append(np.max(hist.history['val_acc']))
    print('acc:', np.mean(score))