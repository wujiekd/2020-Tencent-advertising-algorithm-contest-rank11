import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import keras
from sklearn.model_selection import StratifiedKFold
import random as rn
from tensorflow.keras.callbacks import EarlyStopping
import gc
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)


"""
指定输入路径
"""
age_input_dir = '../models/data/stacking/age/' #所有age模型的输出
gender_input_dir = '../models/data/stacking/gender/' #所有gender模型的输出
"""
指定输出路径
"""
output_dir = '../models/stacking_model/' #集成后结果的输出
sumbit_dir = '../'  #指定submission.csv输出路径
model_dir = '../models/stacking_model/' #stacking模型的输出
raw_data_path = '../models/wujie/rawdata/' #额外保存的一份label路径


def read_age_out(label):  
    """
    读取各个模型的输出，进行处理，为stacking作准备
    输入
    label:单独提取出来的user_id、age、gender的信息
    
    输出
    x:训练集数据
    y:one-hot处理的标签
    y_one:没有处理过的原标签
    x_test:测试集数据
    """
    train1 = np.load(age_input_dir + 'train_age_model1.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model1.py
    test1 = np.load(age_input_dir + 'test_age_model1.npy')
    
    train2 = np.load(age_input_dir + 'train_age_model2.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model2.py
    test2 = np.load(age_input_dir + 'test_age_model2.npy')
    
    train3 = np.load(age_input_dir + 'age_LSTM_train_version1.npy')#模型代码文件在./models/lyu/model/model.py
    test3 = np.load(age_input_dir + 'age_LSTM_test_version1.npy')
    
    train4 = np.load(age_input_dir + 'age_LSTM_train_version2.npy')#模型代码文件在./models/lyu/model/model.py
    test4 = np.load(age_input_dir + 'age_LSTM_test_version2.npy')
    
    train5 = np.load(age_input_dir + 'age_LSTM__train.npy')#模型代码文件在./models/lyu/model/model.py
    test5 = np.load(age_input_dir + 'age_LSTM__test.npy')

    train6 = np.load(age_input_dir + 'train_age_model3.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model3.py
    test6 = np.load(age_input_dir + 'test_age_model3.npy')

    train7 = np.load(age_input_dir + 'age_LSTM_a_train.npy')#模型代码文件在./models/lyu/model/model.py
    test7 = np.load(age_input_dir + 'age_LSTM_a_test.npy')

    train8 = np.load(age_input_dir + 'age_Res__train.npy')#模型代码文件在./models/lyu/model/model.py
    test8 = np.load(age_input_dir + 'age_Res__test.npy')

    train9 = np.load(age_input_dir + 'train_age_model4.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model4.py
    test9 = np.load(age_input_dir + 'test_age_model4.npy')

    train10 = np.load(age_input_dir + 'age_LSTM_dw_train.npy')#模型代码文件在./models/lyu/model/model.py
    test10 = np.load(age_input_dir + 'age_LSTM_dw_test.npy')

    train11 = np.load(age_input_dir + 'train_age_model5.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model5.py
    test11 = np.load(age_input_dir + 'test_age_model5.npy')

    train12 = np.load(age_input_dir + 'train_age_model6.npy')#模型代码文件在 ./src/wujie/model/age_model/age_model6.py
    test12 = np.load(age_input_dir + 'test_age_model6.npy')

    train13 = np.load(age_input_dir + 'age_LSTM_dwa_train.npy')#模型代码文件在./models/lyu/model/model.py
    test13 = np.load(age_input_dir + 'age_LSTM_dwa_test.npy')

    train_w1_1 =np.load(age_input_dir + 'wuzhe_age_model2_datatime_relr_0.513355.npy') #模型代码文件在./src/istar/age/nn_age3_model2_datatime_relr
    train_w1_2 =np.load(age_input_dir + 'wuzhe_age_model2_datatime_relr_dastr_0.5133853333333334.npy')

    train_w2_1 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_wuzhe_256wei_train4_7_atten_relr_0.5120206666666667.npy')#模型代码文件在./src/istar/age/nn_age3_wuzhe_256wei_train4_7_atten_relr
    train_w2_2 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_wuzhe_256wei_train4_7_atten_relr_dastr_0.5123973333333334.npy')

    train_w3_1 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_adid1_count3_datatime_relr_0.51354.npy')#模型代码文件在./src/istar/age/nn_age3_adid1_count3_datatime_relr
    train_w3_2 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_adid1_count3_datatime_relr_dastr_0.5136636666666666.npy')

    train_w4_1 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_wuzhe_targetcode_atten_5id_relr_0.5134846666666667.npy')#模型代码文件在./src/istar/age/nn_age3_wuzhe_targetcode_atten_5id_relr_
    train_w4_2 =np.load(age_input_dir + 'wuzhe_submission_nn_age3_wuzhe_targetcode_atten_5id_relr_dastr_0.5137136666666666.npy')

    train_w1_1 = (train_w1_1 + train_w1_2)/2
    train_w1_1 = np.delete(train_w1_1, [10], axis=1)
    train_w2_1 = (train_w2_1 + train_w2_2)/2
    train_w2_1 = np.delete(train_w2_1, [10], axis=1)
    train_w3_1 = (train_w3_1 + train_w3_2)/2
    train_w3_1 = np.delete(train_w3_1, [10], axis=1)
    train_w4_1 = (train_w4_1 + train_w4_2)/2
    train_w4_1 = np.delete(train_w4_1, [10], axis=1)

    #17model 13+dw + me2 + li
    train = np.hstack((train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12,train13,train_w1_1[:3000000],train_w2_1[:3000000],train_w3_1[:3000000],train_w4_1[:3000000]))
    test = np.hstack((test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,train_w1_1[3000000:],train_w2_1[3000000:],train_w3_1[3000000:],train_w4_1[3000000:]))

    feature = np.vstack((train,test)) #合并做归一化
    scaler = MinMaxScaler()
    scaler.fit(feature)
    df= scaler.transform(feature)  
    train = df[:3000000]
    test = df[3000000:]
    df = pd.DataFrame(df)

    x = train
    x_test = test
    y_one = np.array(label['age']-1)
    y = to_categorical(y_one, num_classes=10)
    
    return x,y,y_one,x_test 


def read_gender_out(label):
    """
    读取各个模型的输出，进行处理，为stacking作准备
    输入
    label:单独提取出来的user_id、age、gender的信息
    
    输出
    x:训练集数据
    y:one-hot处理的标签
    y_one:没有处理过的原标签
    x_test:测试集数据
    """
    train1 = np.load(gender_input_dir+'gender_Res_train.npy')#模型代码文件在./models/lyu/model/model.py
    test1 = np.load(gender_input_dir+'gender_Res_test.npy')
    
    train2 = np.load(gender_input_dir+'gender_LSTM_train.npy')#模型代码文件在./models/lyu/model/model.py
    test2 = np.load(gender_input_dir+'gender_LSTM_test.npy')

    train3 = np.load(gender_input_dir+'gender_LSTM__train.npy')#模型代码文件在./models/lyu/model/model.py
    test3 = np.load(gender_input_dir+'gender_LSTM__test.npy')

    train4 = np.load(gender_input_dir+'train_gender_model2.npy')#模型代码文件在 ./src/wujie/model/gender_model/gender_model2.py
    test4 = np.load(gender_input_dir+'test_gender_model2.npy')

    train5 = np.load(gender_input_dir+'train_gender_model3.npy')#模型代码文件在 ./src/wujie/model/gender_model/gender_model3.py
    test5 = np.load(gender_input_dir+'test_gender_model3.npy')

    train6 = np.load(gender_input_dir+'gender_Res__train.npy')#模型代码文件在./models/lyu/model/model.py
    test6 = np.load(gender_input_dir+'gender_Res__test.npy')

    train7 = np.load(gender_input_dir+'gender_LSTM_dw_train.npy')#模型代码文件在./models/lyu/model/model.py
    test7 = np.load(gender_input_dir+'gender_LSTM_dw_test.npy')

    train_w1_1 =np.load(gender_input_dir+'wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_0.9494853.npy')#模型代码文件在./src/istar/gender/nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr
    train_w1_2 =np.load(gender_input_dir+'wuzhe_submission_nn_gender3_1input_fc256_drop0.4_bn0.4_datatime_relr_dastr_0.949459.npy')
    
    train_w2_1 =np.load(gender_input_dir+'wuzhe_submission_nn_gender3_atten_relr_0.949219.npy')#模型代码文件在./src/istar/gender/nn_gender3_atten_relr
    train_w2_2 =np.load(gender_input_dir+'wuzhe_submission_nn_gender4_atten_relr_dastr_0.9492086666666667.npy')
    
    train_w3_1 =np.load(gender_input_dir+'wuzhe_submission_nn_gender3_count3_tarcode_relr_0.949336.npy')#模型代码文件在./src/istar/gender/nn_gender3_atten_targetcode_datatime_relr
    train_w3_2 =np.load(gender_input_dir+'wuzhe_submission_nn_gender4_count3_tarcode_relr_dastr_0.949347.npy')

    train_w1_1 = np.delete(train_w1_1, [2], axis=1)#0.9494853333333333
    train_w1_2 = np.delete(train_w1_2, [2], axis=1)#0.9494596666666667
    train_w1_1 = (train_w1_1+train_w1_2)/2#0.949901

    train_w2_1 = np.delete(train_w2_1, [2], axis=1)#0.949219
    train_w2_2 = np.delete(train_w2_2, [2], axis=1)#0.9492086666666667
    train_w2_1 = (train_w2_1+train_w2_2)/2#0.9496683333333333

    train_w3_1 = np.delete(train_w3_1, [2], axis=1)#
    train_w3_2 = np.delete(train_w3_2, [2], axis=1)#
    train_w3_1 = (train_w3_1+train_w3_2)/2#0.949776


    train = np.hstack((train1,train2,train3,train4,train5,train6,train7,train_w1_1[:3000000],train_w2_1[:3000000],train_w3_1[:3000000]))
    test = np.hstack((test1,test2,test3,test4,test5,test6,test7,train_w1_1[3000000:],train_w2_1[3000000:],train_w3_1[3000000:]))


    feature = np.vstack((train,test))
    scaler = MinMaxScaler()
    scaler.fit(feature)
    df= scaler.transform(feature)
    train = df[:3000000]
    test = df[3000000:]
    df = pd.DataFrame(df)

    x = train
    y_one = np.array(label['gender']-1)
    x_test = test
    y = to_categorical(y_one, num_classes=2)

    return x,y,y_one,x_test


def age_model_conv():
    """
    age的stacking模型
    """
    K.clear_session()
    
    input1 = Input(shape=(170, ))
    x = Dense(600, activation='relu')(input1)
    pred = Dense(10, activation='softmax')(x)
    model = Model(inputs=[input1], outputs=pred)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),metrics=["accuracy"])
    return model

def gender_model_conv():
    """
    gender的stacking模型
    """
    K.clear_session()
    
    input1 = Input(shape=(20, ))
    x = Dense(64, activation='relu')(input1)
    pred = Dense(2, activation='softmax')(x)
    model = Model(inputs=[input1], outputs=pred)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=3e-4),metrics=["accuracy"])
    return model

if __name__ == "__main__":
    label = pd.read_pickle(raw_data_path+'label.pkl')

    """
    age执行stacking，使用了这一个月来的17个模型，从低版本到高版本，使用NN保证了不会缺失有用的信息。
    """
    x,y,y_one,x_test = read_age_out(label)
    kfolder = KFold(n_splits=5, shuffle=True, random_state=2020)
    oof_nn = np.zeros((3000000, 10))
    predictions_nn = np.zeros((1000000, 10))
    kfold = kfolder.split(x, y_one)
    fold_ = 0
    for train_index, vali_index in kfold:
        fold_+=1
        k_x_train = x[train_index]
        k_y_train = y[train_index]
        k_x_vali = x[vali_index]
        k_y_vali = y[vali_index]


        model = age_model_conv()
        model.summary()

        checkpoint = ModelCheckpoint(
            model_dir+f"./age_model{fold_}.h5", monitor='val_acc', verbose=0, save_best_only=True, mode='max',save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=3, min_lr=0.000001, verbose=0)
        earlystopping = EarlyStopping(
            monitor='val_acc', min_delta=0.000001, patience=10, verbose=0, mode='max')
        callbacks = [reduce_lr, earlystopping,checkpoint]
        model.fit(k_x_train,k_y_train,batch_size =1024,epochs=100,validation_data=(k_x_vali, k_y_vali), callbacks=callbacks)#callbacks=callbacks,
        model.load_weights(model_dir+f"./age_model{fold_}.h5")
        oof_nn[vali_index] = model.predict(k_x_vali)
        predictions_nn += model.predict(x_test) / kfolder.n_splits

    print("NN score: {:<8.8f}".format(accuracy_score(np.argmax(oof_nn,axis=1), y_one)))

    #保存结果
    answer = pd.DataFrame()
    answer['test_age']=np.argmax(predictions_nn, axis=1)
    answer.to_csv(output_dir+'nn_test_age_result.csv', index=False)
    
    """
    gender执行stacking，使用了这一个月来的10个模型，从低版本到高版本，使用NN保证了不会缺失有用的信息。
    """
    x,y,y_one,x_test = read_gender_out(label)
    kfolder = KFold(n_splits=5, shuffle=True, random_state=2020)
    oof_nn = np.zeros((3000000, 2))
    predictions_nn = np.zeros((1000000, 2))
    kfold = kfolder.split(x, y_one)
    fold_ = 0
    for train_index, vali_index in kfold:
        fold_ +=1
        k_x_train = x[train_index]
        k_y_train = y[train_index]
        k_x_vali = x[vali_index]
        k_y_vali = y[vali_index]


        model = gender_model_conv()
        model.summary()

        checkpoint = ModelCheckpoint(
            model_dir+f"./gender_model{fold_}.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        earlystopping = EarlyStopping(
            monitor='val_acc', min_delta=0.00001, patience=10, verbose=1, mode='max')
        callbacks = [reduce_lr, earlystopping,checkpoint]
        model.fit(k_x_train,k_y_train,batch_size =512,epochs=100,validation_data=(k_x_vali, k_y_vali), callbacks=callbacks)
        model.load_weights(model_dir+f"./gender_model{fold_}.h5")
        oof_nn[vali_index] = model.predict(k_x_vali)
        predictions_nn += model.predict(x_test) / kfolder.n_splits

    print("NN score: {:<8.8f}".format(accuracy_score(np.argmax(oof_nn,axis=1), y_one)))

    #保存结果
    answer = pd.DataFrame()
    answer['test_gender']=np.argmax(predictions_nn, axis=1)
    answer.to_csv(output_dir+'nn_test_gender_result.csv', index=False)

    """
    生成submission文件
    """
    sub = pd.read_csv('../submission.csv')  # 读取上一次的sub文件，使用用户300w-400w的id
    result = pd.DataFrame()
    result['user_id']=np.array(sub['user_id'])

    age = pd.read_csv(output_dir+'nn_test_age_result.csv') #更新age
    gender = pd.read_csv(output_dir+'nn_test_gender_result.csv') #更新gender


    result['predicted_age'] = age['test_age'].apply(int)+1
    result['predicted_gender']= gender['test_gender'].apply(int)+1

    result.to_csv(sumbit_dir+'submission.csv',sep=',',index=False, encoding='utf-8')
    print("~~~~~~~~~~~~~~~~~~~~~~over~~~~~~~~~~~~~~~~~~~~~~")