#Deepwalk代码
#作用：对每个特征id和用户id进行随机游走生成400w+序列，并且使用word2vec代码生成embedding矩阵。
#注：为了区别于前面的word2vec矩阵，这里由deepwalk算法生成的emb，称为deepwalk矩阵，user_id生成的emb，称为user_id下的deepwalk矩阵。
#注：对五种特征id和用户id进行了deepwalk算法训练，五种id分别是creative_id,ad_id,advertiser_id,product_id,industry。
#注：deepwalk由于生成的序列数量特别多，训练特别占内存，需要在128G以上的机器下运行。

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
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1024)
rn.seed(1024)
tf.set_random_seed(1024)

#指定输入文件路径
raw_data_path = '../../models/data/wujie/rawdata/'
index_data_path = '../../models/data/wujie/index/' #word2vec的index输出路径

#指定文件输出路径
deepwalk_data_path = '../../models/data/wujie/deepwalk/data/'

# deepwalk构建图，训练获得deepwalk矩阵
def deepwalk(log,f1,f2,flag,L,lens,path):
    '''
    输入
    log:全部数据集
    f1:主键
    f2:需要训练的特征id
    flag:保存文件算法名称
    lens:构建deepwalk路径长度
    path:指定生成的emb输出的路径
    '''
    print("deepwalk:",f1,f2)
    #构建图
    dic={}
    for item in log[[f1,f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))]=set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))]=set(['item_'+str(int(item[1]))])
    dic_cont={}
    for key in dic:
        dic[key]=list(dic[key])
        dic_cont[key]=len(dic[key])
    print("creating")     
    #构建路径
    path_length=lens        
    sentences=[]
    length=[]
    for key in dic:
        sentence=[key]
        while len(sentence)!=path_length:
            key=dic[sentence[-1]][random.randint(0,dic_cont[sentence[-1]]-1)]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%500000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    #训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, sg=1, window=10, seed=1024, workers=64, min_count=1, iter=15)
    #model = Word2Vec(sentences, size=L, window=4,min_count=1,sg=1, workers=10,iter=20)
    print('outputing...')
    del sentences
    del length
    del dic
    del dic_cont
    gc.collect()
    """
    输出
    out_df：输出主键user_id对应的deepwalk矩阵
    """
    values=set(log[f1].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['user_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f1]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(path +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    del w2v
    gc.collect()
    ########################
    """
    输出
    out_df：输出特征id对应的deepwalk矩阵
    """
    values=set(log[f2].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['item_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f2]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(path +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    gc.collect()

# 对五个特征id分别训练得到的user_id对应的emb进行concat处理，可作为模型的输入
def Deepwalk_user_id(path,out_name):
    '''
    输入
    path:产生的主键user_id的embedding矩阵对应文件路径
    out_name:输出的文件名
    '''
    user_creative_id = pd.read_pickle(path+'user_id_creative_id_user_id_DeepWalk_deepwalk_192.pkl')
    user_ad_id = pd.read_pickle(path+'user_id_ad_id_user_id_DeepWalk_deepwalk_192.pkl')
    user_advertiser_id = pd.read_pickle(path+'user_id_advertiser_id_user_id_DeepWalk_deepwalk_128.pkl')
    user_product_id = pd.read_pickle(path+'user_id_product_id_user_id_DeepWalk_deepwalk_128.pkl')
    user_industry = pd.read_pickle(path+'user_id_industry_user_id_DeepWalk_deepwalk_64.pkl')
    del user_creative_id['user_id']
    del user_ad_id['user_id']
    del user_advertiser_id['user_id']
    del user_product_id['user_id']
    del user_industry['user_id']
    user_creative_id = np.array(user_creative_id)
    user_ad_id = np.array(user_ad_id)
    user_advertiser_id = np.array(user_advertiser_id)
    user_product_id = np.array(user_product_id)
    user_industry = np.array(user_industry)
    all_df = np.hstack((user_creative_id,user_ad_id,user_advertiser_id,user_product_id,user_industry))

    all_df_zip = all_df.astype(np.float32) # 精度减半，减少模型训练的内存
    np.save(path+out_name,all_df_zip) # 将每个用户训练对应的词向量保存了，可设一个mlp作为输入

    
# 得到Deepwalk的embedding矩阵，升级版比较快,利用pandas内置函数进行排序
def DeepWalk_get_embedding_matrix(word_index,embedding,name, embed_size=300):
    '''
    输入
    word_index:文本和数字对应的索引
    embedding:deepwalk算法训练完得到的deepwalk矩阵
    name:特征id名称
    embed_size:embedding矩阵的长度
    
    输出
    embedding_matrix:对应的embedding矩阵
    '''
    index_key = list(word_index.keys())
    embedding[name] = embedding[name].astype('str').astype('category')
    """for i in index_key: #判断是否全部一致
        if i not in embedding[name].value_counts().index:print(i)"""
    print(len(index_key))
    print(embedding[name].shape)
    # inplace = True，使 recorder_categories生效
    embedding[name].cat.reorder_categories(index_key, inplace=True)

    # inplace = True，使 df生效
    embedding.sort_values(name, inplace=True)
    del embedding[name]
    embedding = np.array(embedding)
    embedding_vector = np.zeros(embed_size)
    print(embedding.shape)
    print(embedding_vector.shape)
    print(embedding_vector)
    embedding_matrix = np.insert(embedding, 0, values=embedding_vector, axis=0)
    return embedding_matrix

def get_emb(old_train_emb_dir,path,name,size,output_dir):
    '''
    输入
    old_train_emb_dir:deepwalk算法训练完得到的deepwalk矩阵路径
    path:word2vec算法产生的索引的路径，重复使用这个索引，可以复用用户序列，减少工作量
    name:特征id名称
    size:embedding矩阵的长度
    output_dir:输出文件夹路径
    '''
    old_train_emb = pd.read_pickle(old_train_emb_dir)
    print(old_train_emb.shape)
    import joblib
    index_1 = joblib.load(path+'index_1.pkl')
    print(len(index_1))
    emb = DeepWalk_get_embedding_matrix(index_1,old_train_emb,name,size) 
    np.save(output_dir + 'emb_'+name+'.npy',emb)

if __name__ == "__main__":
    train = pd.read_pickle(raw_data_path+'train.pkl')
    test = pd.read_pickle(raw_data_path+'test.pkl')
    all_df = pd.concat([train,test]) # 合并训练集和测试集
    all_df["product_id"] = all_df["product_id"].apply(lambda row : row.replace('\\N', '1000000')) # 对NA值填一个大正整数
    all_df["industry"] = all_df["industry"].apply(lambda row : row.replace('\\N', '1000000'))

    deepwalk(all_df,'user_id','advertiser_id',"DeepWalk",128,20,path=deepwalk_data_path)
    deepwalk(all_df,'user_id','product_id',"DeepWalk",128,20,path=deepwalk_data_path)
    deepwalk(all_df,'user_id','industry',"DeepWalk",64,20,path=deepwalk_data_path)
    deepwalk(all_df,'user_id','ad_id',"DeepWalk",192,10,path=deepwalk_data_path) 
    deepwalk(all_df,'user_id','creative_id',"DeepWalk",192,10,path=deepwalk_data_path)
    gc.collect()
    
    Deepwalk_user_id(path=deepwalk_data_path,out_name='all_user_data.npy') # 生成用户id编码特征
    
    old_train_emb_dir = deepwalk_data_path+'user_id_industry_industry_DeepWalk_deepwalk_64.pkl'  #训练完的deepwalk
    path = index_data_path+"w2v_model5_100_64/" #之前w2v产生的index，重复使用这个序列，减少工作量
    name = 'industry'
    size =64
    get_emb(old_train_emb_dir,path,name,size,output_dir='./DeepWalk/')#处理得到对应的emb
    
    old_train_emb_dir = deepwalk_data_path+'user_id_product_id_product_id_DeepWalk_deepwalk_128.pkl'  #训练完的deepwalk
    path = index_data_path+"w2v_model4_100_128/" #之前w2v产生的index，重复使用这个序列，减少工作量
    name = 'product_id'
    size =128
    get_emb(old_train_emb_dir,path,name,size,output_dir='./DeepWalk/')#处理得到对应的emb


    old_train_emb_dir = deepwalk_data_path+'user_id_advertiser_id_advertiser_id_DeepWalk_deepwalk_128.pkl'  #训练完的deepwalk
    path = index_data_path+"w2v_model3_100_128/" #之前w2v产生的index，重复使用这个序列，减少工作量
    name = 'advertiser_id'
    size =128
    get_emb(old_train_emb_dir,path,name,size,output_dir='./DeepWalk/')#处理得到对应的emb

    old_train_emb_dir = deepwalk_data_path+'user_id_ad_id_ad_id_DeepWalk_deepwalk_192.pkl'  #训练完的deepwalk
    path = index_data_path+"w2v_model2_100_256/" #之前w2v产生的index，重复使用这个序列，减少工作量
    name = 'ad_id'
    size =192
    get_emb(old_train_emb_dir,path,name,size,output_dir='./DeepWalk/')#处理得到对应的emb

    old_train_emb_dir = deepwalk_data_path+'user_id_creative_id_creative_id_DeepWalk_deepwalk_192.pkl'  #训练完的deepwalk
    path = index_data_path+"w2v_model1_100_256/" #之前w2v产生的index，重复使用这个序列，减少工作量
    name = 'creative_id'
    size =192
    get_emb(old_train_emb_dir,path,name,size,output_dir='./DeepWalk/')#处理得到对应的emb