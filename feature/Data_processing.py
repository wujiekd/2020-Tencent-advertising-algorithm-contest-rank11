#数据预处理代码
#作用一：对原始数据合并处理，生成一份训练集train.pkl、测试集test.pkl、以及一份训练集的单独标签label.pkl；
#作用二：按照顺序划分五折的训练集和验证集，方便后面进行目标编码；
#注：因为数据预处理代码原本主要为ipynb格式，转为py格式如有报错请及时联系队伍。

# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#指定原始数据文件路径
path1 = './train_preliminary/'
pathsemi = './train_semi_final/'
path2 = './test/'

#指定文件输出路径
raw_data_path = '../../models/data/wujie/rawdata/'
tar_en_path = '../../models/data/wujie/traindata/'

train_user = pd.read_csv(path1+'user.csv', sep=',')
train_semi_user = pd.read_csv(pathsemi+'user.csv', sep=',')

label = pd.concat([train_user,train_semi_user],axis=0)

label.to_pickle(raw_data_path+'label.pkl') # 提取age和gender标签进行单独保存

Train_ad = pd.read_csv(path1+'ad.csv', sep=',')
Train_semi = pd.read_csv(pathsemi+'ad.csv', sep=',')
Train_click_log = pd.read_csv(path1+'click_log.csv', sep=',')
Train_click_log_semi = pd.read_csv(pathsemi+'click_log.csv', sep=',')

print(Train_click_log.shape)
print(Train_click_log_semi.shape)

Train_ad = pd.concat([Train_ad,Train_semi])
Train_click_log= pd.concat([Train_click_log,Train_click_log_semi])
train_user = pd.read_pickle(raw_data_path+'label.pkl')

Train_ad = Train_ad.drop_duplicates(subset=['creative_id', 'ad_id', 'product_id', 'product_category',
       'advertiser_id', 'industry'], keep='first')
del Train_semi
del Train_click_log_semi

train = Train_click_log.merge(Train_ad, on='creative_id', how='left')

del Train_click_log
del Train_ad

train = train.merge(train_user, on='user_id', how='left')

del train_user

train = train.sort_values(['user_id','time'])

train.to_pickle(raw_data_path+'train.pkl') # 将训练集进行左连接合并并保存

del train

Test_ad = pd.read_csv(path2+'ad.csv', sep=',')
Test_click_log = pd.read_csv(path2+'click_log.csv', sep=',')
test = Test_click_log.merge(Test_ad, on='creative_id', how='left')
test = test.sort_values(['user_id','time'])
test.to_pickle(raw_data_path+'test.pkl') # 将测试集进行左连接合并并保存


"""
对age和gender合成目标20类，用于后面的目标编码
"""
train = pd.read_pickle(raw_data_path+'train.pkl')
test = pd.read_pickle(raw_data_path+'test.pkl')
train['age_gender'] = train['age']+(train['gender']-1)*10


train1 = train[train['user_id']<=600000]
train2 = train[(train['user_id']<=2*600000)&(train['user_id']>600000)]
train3 = train[(train['user_id']<=3*600000)&(train['user_id']>2*600000)]
train4 = train[(train['user_id']<=4*600000)&(train['user_id']>3*600000)]
train5 = train[(train['user_id']<=5*600000)&(train['user_id']>4*600000)]


train1.to_pickle(tar_en_path+'vail1.pkl')
train2.to_pickle(tar_en_path+'vail2.pkl')
train3.to_pickle(tar_en_path+'vail3.pkl')
train4.to_pickle(tar_en_path+'vail4.pkl')
train5.to_pickle(tar_en_path+'vail5.pkl')

train_1 = pd.concat([train2,train3,train4,train5])
train_1 = pd.concat([train2,train3,train4,train5])
train_3 = pd.concat([train1,train2,train4,train5])
train_4 = pd.concat([train1,train2,train3,train5])
train_5 = pd.concat([train1,train2,train3,train4])

train_1.to_pickle(tar_en_path+'train1.pkl')
train_2.to_pickle(tar_en_path+'train2.pkl')
train_3.to_pickle(tar_en_path+'train3.pkl')
train_4.to_pickle(tar_en_path+'train4.pkl')
train_5.to_pickle(tar_en_path+'train5.pkl')