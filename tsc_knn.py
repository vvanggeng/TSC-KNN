# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import cydtw #基于C编译的快速DTW方法
import os
from sklearn.metrics import confusion_matrix 

'''
数据导入
'''
def read_data(filepath):
    data = pd.read_table(filepath,header=None,skiprows=[0,1],sep='\s+')
    return data

def data_import(base_path):
    files = os.listdir(base_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    data = np.empty((len(files),1050,4))
    i=0
    for path in files:
        full_path = os.path.join(base_path, path)
        data[i]=MinMaxScaler().fit_transform(read_data(full_path))
        i+=1
    labels=[]
    for i in range(16):
        for j in range(data.shape[0]//16):
            labels.append(i+1)
    labels=np.array(labels)
    return data,labels

'''
设置训练数据和测试数据
'''
train_data,train_labels=data_import(r'./data/2/train') #训练数据
test_data,test_labels=data_import(r'./data/2/test') #测试数据

'''
K-K近邻算法K值 train_data-训练数据,train_labels-训练数据标签,
test_data-测试数据,test_labels-测试数据标签,labels_name-数字标签转换为字母
'''
def predict(K,train_data,train_labels,test_data,test_labels,labels_name):
    i=0
    accuracy=0
    predict_labels = []
    for test in test_data:
        t_dis=[]
        for train in train_data:
           dis=cydtw.dtw(test.T, train.T)#dtw计算距离
           t_dis.append(dis) #距离数组
        #KNN算法预测标签   
        nearest_series_labels = np.array(train_labels[np.argpartition(t_dis, K)[:K]]).astype(int)
        preditc_labels_single = np.argmax(np.bincount(nearest_series_labels))
        predict_labels.append(preditc_labels_single)
        #计算正确率
        if preditc_labels_single==test_labels[i] :
            accuracy+=1
        i+=1
    print('The accuracy is %f (%d of %d)'%((accuracy/test_data.shape[0]),accuracy,test_data.shape[0]))
    cm_plot(test_labels, predict_labels,labels_name)#绘制混淆矩阵
    return accuracy/test_data.shape[0]

labels_name=[]
for i in range(16):
    labels_name.append(chr(ord('A')+i))  
    
predict(1,train_data,train_labels,test_data,test_labels,labels_name)

'''
混淆矩阵绘制代码
'''
def cm_plot(original_label, predict_label,labels_name):    
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    plt.imshow(cm,interpolation='nearest') 
    #plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    cb=plt.colorbar()    # 颜色标签
    cb.ax.tick_params(labelsize=14)  #设置色标刻度字体大小。
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center',fontsize=14)
    num_x = np.array(range(len(labels_name)))
    num_y = np.array(range(len(labels_name)))
    plt.xticks(num_x, labels_name,fontsize=16)    # 将标签印在x轴坐标上
    plt.yticks(num_y, labels_name,fontsize=16)
    plt.ylabel('True Area',fontsize=22)  # 坐标轴标签
    plt.xlabel('Predicted Area',fontsize=22)  # 坐标轴标签
    plt.title('LVI Confusion Matrix',fontsize=22)
    plt.ylim([-0.5,15.5])

    




