#word2vec代码
#作用：针对5种id按照时间顺序对每个用户分别生成用户点击序列，并且使用word2vec代码生成embedding矩阵。
#注：五种id分别是creative_id,ad_id,advertiser_id,product_id,industry

# 导入相关库
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
import tensorflow as tf
import random as rn
import gc
import logging
import gensim
import fasttext
os.environ['PYTHONHASHSEED'] = '0'
# 显卡使用（如没显卡需要注释掉）
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

#指定原始文件路径
raw_data_path = '../../models/data/wujie/rawdata/'

#指定文件输出路径
emb_data_path = '../../models/data/wujie/emb/' #embedding输出路径
index_data_path = '../../models/data/wujie/index/' #word2vec的index输出路径


#生成序列数据
def set_first_times(row):
    return ' '.join(list(row[name]))


### Tokenizer 序列化文本
def set_tokenizer(docs, split_char=' ', max_len=100):
    '''
    输入
    docs:文本列表
    split_char:按什么字符切割
    max_len:截取的最大长度
    
    输出
    X:序列化后的数据
    word_index:文本和数字对应的索引
    '''
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index=tokenizer.word_index
    return X, word_index

### 做embedding 这里采用word2vec
def trian_save_word2vec(docs, embed_size=300, save_name='w2v.txt', split_char=' '):
    '''
    输入
    docs:输入的文本列表
    embed_size:embed长度
    save_name:保存的word2vec位置
    
    输出
    w2v:返回的模型
    '''
    input_docs = []
    for i in docs:
        input_docs.append(i.split(split_char))
    logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=20, seed=1017, workers=64, min_count=1, iter=10)
    w2v.wv.save_word2vec_format(save_name)
    print("w2v model done")
    gc.collect
    return w2v

# 得到embedding矩阵
def get_embedding_matrix(word_index, embed_size=300, Emed_path="w2v_300.txt"):
    '''
    输入
    word_index:文本和数字对应的索引
    embed_size:embed长度
    Emed_path:保存的word2vec位置
    
    输出
    embedding_matrix:对应的embedding矩阵
    '''
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= nb_words:
            break
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
    print("null cnt",count)
    return embedding_matrix

# 针对各个id分别进行word2vec
def tencent_w2v(path,name,size,output_dir):
    '''
    输入
    path:指定该id对应输出的路径（输出文件主要是用户序列和部分中间文件）
    name:id名称
    size:word2vec的size
    output_dir:embedding矩阵的输出路径
    '''
    train = pd.read_pickle(raw_data_path+'train.pkl')
    train.columns
    test = pd.read_pickle(raw_data_path+'test.pkl')
    cols = ['creative_id', 'click_times', 'ad_id', 'product_id','product_category', 'advertiser_id', 'industry']
    for i in cols: # del无关变量，减少内存占用
        if i==name:continue
        del train[i]
        del test[i]
    train = pd.concat([train, test], ignore_index=True)
    del test
    print(train[name].value_counts())
    
    train[name] = train[name].astype(str)
    group_data =  train.groupby('user_id').apply(lambda row:set_first_times(row)).reset_index()  
    group_data = group_data.rename( columns = {0:name})
    #要组合，那不能去除NAN
    group_data[name] = group_data[name].apply(lambda row : row.replace('\\N', "1000000")) # 对NA值使用大正整数填充

    group_data['len'] = group_data[name].apply(lambda row : len(row.split(' ')))
    print(group_data.quantile(q=0.95))

    print('开始序列化')
    text_1_list = list(group_data[name])
    x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=100)
    print('序列化完成')
    gc.collect()

    np.save(path+'x1.npy', x1) # 保存用户点击序列文件
    
    import joblib
    joblib.dump(index_1,path+'index_1.pkl') # 保存该id生成的索引文件，可用于生成别的类型的emb，例如fasttext、doc2vec，保证用户点击序列文件可以复用
    group_data.to_pickle(path+'group_data1.pkl')

    
    import joblib
    index_1 = joblib.load(path+'index_1.pkl')
    group_data= pd.read_pickle(path+'group_data1.pkl')
    text_1_list = list(group_data[name])

    w2v = trian_save_word2vec(text_1_list, save_name=path+f'w2v_in100_size{size}_win20.txt', split_char=' ', embed_size=size)
    gc.collect()

    emb1 = get_embedding_matrix(index_1, embed_size=size,Emed_path=path +f'w2v_in100_size{size}_win20.txt')
    gc.collect()

    np.save(output_dir+f'emb{name}_{size}.npy',emb1)

    print(emb1.shape)
    gc.collect()
    
    
if __name__ == "__main__":
    tencent_w2v(path = index_data_path+'w2v_model1_100_256/',name ='creative_id',size =256,output_dir =emb_data_path)
    tencent_w2v(path = index_data_path+'w2v_model2_100_256/',name ='ad_id',size =256,output_dir =emb_data_path)
    tencent_w2v(path = index_data_path+'w2v_model3_100_128/',name ='advertiser_id',size =128,output_dir =emb_data_path)
    tencent_w2v(path = index_data_path+'w2v_model4_100_128/',name ='product_id',size =128,output_dir =emb_data_path)
    tencent_w2v(path = index_data_path+'w2v_model5_100_64/',name ='industry',size =64,output_dir =emb_data_path)