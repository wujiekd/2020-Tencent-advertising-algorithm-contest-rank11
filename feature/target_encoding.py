#目标编码代码
#作用：对原始数据进行20分类的五折目标编码；
#注：具体实现了3个特征id，分别是creative_id,ad_id,advertiser_id。
#例：对一个特征id中的一个类别特征，统计它在age_gender这20类中的分布，求其概率值，然后对于一个用户，对所有概率值进行mean和max操作。

import pandas as pd
import numpy as np
import gc


#指定输入文件路径
raw_data_path = '../../models/data/wujie/rawdata/'
index_data_path = '../../models/data/wujie/index/' #word2vec的index输出路径
tar_en_path = '../../models/data/wujie/traindata/' #五折处理用于targetencoding的数据

#指定文件输出路径
deepwalk_data_path = '../../models/data/wujie/deepwalk/data/'
target_encoding_path = '../../models/data/wujie/target_encoding/' #五折跑完存放路径
target_out_path =  '../../models/data/wujie/target_out/' #合并五折存放路径

def list_dic(list1,list2):
    '''
    输入
    list1:列表1
    list2:列表2
    
    输出
    dic:以列表1为key，列表2为value的字典
    '''
    dic = dict(map(lambda x,y:[x,y], list1,list2))
    return dic

def target_encoding(df_train,cate,label,df_test): #输入训练集，类别特征，类别标签，测试集 
    '''
    输入
    df_train:输入训练集
    cate:类别特征
    label:类别标签
    df_test:测试集
    
    输出
    hebin:输出类别标签20分类的目标编码概率值
    '''
    abc = df_train[cate].groupby(df_train[label]).value_counts()
    print(abc)
    
    num = df_train[label].value_counts().index.shape[0]  #目标一共num类
    dfs = pd.DataFrame(columns=[cate,label,'num'])
    for i in range(num):     #对num类分成对应的num份类别特征，并用DataFrame进行保存
        j=i+1
        df = pd.DataFrame()
        df[cate] = abc[j].index
        df[label]=j
        df['num']=np.array(abc[j])
        dfs = pd.concat([dfs,df])

    dfs[cate] = dfs[cate].astype(int)   #转换为int类型
    dfs[label] = dfs[label].astype(int)
    dfs['num'] = dfs['num'].astype(int)

    haha = dfs.groupby([cate])['num'].sum()  #统计同一个类别特征在所有类中的总和
    haha= haha.reset_index()
    haha.columns=[cate,'all_num'] 

    wudi = pd.merge(dfs, haha, on=cate)
    wudi['rate'] = wudi['num']/wudi['all_num']    #求该特征在该类别对应的概率
    
    #hebin = pd.concat([df_train,df_test])
    #hebin = df_train
    hebin = df_test
    for i in range(num):
        print(i)
        no_i_age = wudi[wudi[label]==i+1]          #对num类进行依次map，生成num个特征，每个特征为 原cate特征对应该类（1～num）的概率值
        list0 = list(no_i_age[cate])
        list1 = list(no_i_age['rate'])

        dict = list_dic(list0,list1)
         
        hebin[f'add_{i+1}_'+cate] = hebin[cate].map(dict)
    
    gc.collect()
    return hebin

def solve(train,test,name):
    """
    输入
    train:训练集
    test:测试集
    name:进行目标编码的特征
    
    输出
    hebin_all:训练集和测试集合并后的target encoding
    """
    list1 = list(train[name].value_counts().index)#2481135
    list2 = list(test[name].value_counts().index)#2618159
    list3 = list(set(list2)-set(list1))
    
    test_new=test[~test[name].isin(list3)]  #取反 ~
    len(list(test_new[name].value_counts().index))
    len(test_new['user_id'].value_counts())
    test = test_new   #更新test，test中特有的全给删除，统计没屁用
    hebin = target_encoding(train,name,'age_gender',test)
    hebin = hebin.fillna(0)
    gc.collect()
    hebin_mean = hebin.groupby(['user_id'])['add_1_'+name,'add_2_'+name,'add_3_'+name,'add_4_'+name,'add_5_'+name,'add_6_'+name,'add_7_'+name,'add_8_'+name,'add_9_'+name,'add_10_'+name,'add_11_'+name,'add_12_'+name,'add_13_'+name,'add_14_'+name,'add_15_'+name,'add_16_'+name,'add_17_'+name,'add_18_'+name,'add_19_'+name,'add_20_'+name].mean()
    hebin_max = hebin.groupby(['user_id'])['add_1_'+name,'add_2_'+name,'add_3_'+name,'add_4_'+name,'add_5_'+name,'add_6_'+name,'add_7_'+name,'add_8_'+name,'add_9_'+name,'add_10_'+name,'add_11_'+name,'add_12_'+name,'add_13_'+name,'add_14_'+name,'add_15_'+name,'add_16_'+name,'add_17_'+name,'add_18_'+name,'add_19_'+name,'add_20_'+name].max()
    hebin_mean = np.array(hebin_mean)
    hebin_max = np.array(hebin_max)
    hebin_all = np.hstack([hebin_mean,hebin_max])
    gc.collect()
    return hebin_all


if __name__ == "__main__":
    """
    读取划分好的五折数据集 
    """
    train1 = pd.read_pickle(tar_en_path+'train1.pkl')
    train2 = pd.read_pickle(tar_en_path+'train2.pkl')
    train3 = pd.read_pickle(tar_en_path+'train3.pkl')
    train4 = pd.read_pickle(tar_en_path+'train4.pkl')
    train5 = pd.read_pickle(tar_en_path+'train5.pkl')

    vail1 = pd.read_pickle(tar_en_path+'vail1.pkl')
    vail2 = pd.read_pickle(tar_en_path+'vail2.pkl')
    vail3 = pd.read_pickle(tar_en_path+'vail3.pkl')
    vail4 = pd.read_pickle(tar_en_path+'vail4.pkl')
    vail5 = pd.read_pickle(tar_en_path+'vail5.pkl')
    test = pd.read_pickle(raw_data_path+'test.pkl')
    
    """
    对'creative_id','ad_id','advertiser_id'进行target encoding
    """
    names = ['creative_id','ad_id','advertiser_id']
    for name in names: #对3个特征分别五折进行target encoding
        hebin_all = solve(train1,vail1,name)
        np.save(target_encoding_path+name+'_train0',hebin_all)
        gc.collect()
        hebin_all = solve(train1,test,name)
        np.save(target_encoding_path+name+'_test0',hebin_all)
        print(hebin_all.shape)

        hebin_all = solve(train2,vail2,name)
        np.save(target_encoding_path+name+'_train1',hebin_all)
        gc.collect()
        hebin_all = solve(train2,test,name)
        np.save(target_encoding_path+name+'_test1',hebin_all)
        print(hebin_all.shape)

        hebin_all = solve(train3,vail3,name)
        np.save(target_encoding_path+name+'_train2',hebin_all)
        gc.collect()
        hebin_all = solve(train3,test,name)
        np.save(target_encoding_path+name+'_test2',hebin_all)
        print(hebin_all.shape)

        hebin_all = solve(train4,vail4,name)
        np.save(target_encoding_path+name+'_train3',hebin_all)
        gc.collect()
        hebin_all = solve(train4,test,name)
        np.save(target_encoding_path+name+'_test3',hebin_all)
        print(hebin_all.shape)

        hebin_all = solve(train5,vail5,name)
        np.save(target_encoding_path+name+'_train4',hebin_all)
        gc.collect()
        hebin_all = solve(train5,test,name)
        np.save(target_encoding_path+name+'_test4',hebin_all)
        print(hebin_all.shape)
        
    """
    处理五折后数据；
    训练集进行拼接；
    测试集进行平均。
    """
    train=[]
    for i in range(5):
        j = i
        creative_id_train1 = np.load(target_encoding_path+f'creative_id_train{j}.npy')
        ad_id_train1 = np.load(target_encoding_path+f'ad_id_train{j}.npy')
        advertiser_id_train1 = np.load(target_encoding_path+f'advertiser_id_train{j}.npy')
        train1 = np.hstack((creative_id_train1,ad_id_train1,advertiser_id_train1))
        if j==0:
            train = train1
            continue
        train = np.vstack((train,train1))
    np.save(target_out_path+'target_train',train)
    
    
    test=[]
    for i in range(5):
        j = i
        creative_id_train1 = np.load(target_encoding_path+f'creative_id_test{j}.npy')
        ad_id_train1 = np.load(target_encoding_path+f'ad_id_test{j}.npy')
        advertiser_id_train1 = np.load(target_encoding_path+f'advertiser_id_test{j}.npy')
        test1 = np.hstack((creative_id_train1,ad_id_train1,advertiser_id_train1))
        if j==0:
            test = test1
            continue
    test += test1   
    test = test/5  
    np.save(target_out_path+'target_test',test)