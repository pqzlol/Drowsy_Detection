#coding=utf-8 
#import numpy as np 
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
#从两个txt文件中解析数据，提取出特征向量，放入列表train中，同时并把对应的标签放入列表labels中
train = []
labels = []

print('Reading train_open.txt...')

train_open_txt=open('train_open.txt',"r") 
for txt_str in train_open_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    #print(datas)
    for data in datas:
        # print(data)
        data = float(data)
        temp.append(data)
    # print(temp)
    train.append(temp)
    labels.append('open')

print('Reading train_close.txt...')


train_close_txt=open('train_close.txt',"r") 
#读取全部内容 ，并以列表方式返回
for txt_str in train_close_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    #print(datas)
    for data in datas:
        # print(data)
        data = float(data)
        temp.append(data)
    # print(temp)
    train.append(temp)
    labels.append('close')
   
print train   
count=0
for i in range(len(labels)):
    count+=1
    #print("count:{}".format(count))
    #print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()


#训练并保存模型
#使用sickit-learn中的svm模块，为了简便就直接调用实现好的api函数
#参数解析
#--C=0.8惩罚系数，用来控制损失函数的惩罚系数
    #--C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    #--这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    #--C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合； 
#--kernel='linear'表示采用线性核；kernel='rbf'时（default），为高斯核，
#--gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”； 默认是'auto'，那么将会使用特征位数的倒数，即1 / n_features。（即核函数的带宽，超圆的半径）。
    #--gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。
#--decision_function_shape='ovr'，表示one v rest，即一个类别与其他类别划分，多分类；
#--decision_function_shape='ovo'，表示one v one，即一个类别与另一个类别划分，二分类；

#clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
#clf = svm.SVC(C=0.8, kernel='linear', gamma='auto', decision_function_shape='ovo')
#clf = svm.SVC(C=0.8, kernel='linear', gamma=30, decision_function_shape='ovo')
clf = svm.SVC(C=2.04, kernel='rbf', gamma=0.9, decision_function_shape='ovo')
# 拟合模型
clf.fit(train, labels)

#计算样本点到分割超平面的函数距离
#print('decision_function:{}'.format(clf.decision_function(train))) 

#生成模型
joblib.dump(clf, "ear_svm.m")
#测试训练集的准确率
print("accuracy:{}".format(clf.score(train,labels)))

# 输出更加详细的其他评价分类性能的指标。
print("acy_true:{}".format(classification_report(labels, clf.predict(train))))

print("acy:{}".format(classification_report(clf.predict(train),labels)))
# 函数解析
#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2)，显示主要的分类指标，返回每个类标签的精确、召回率及F1值
# 主要参数说明：
    # y_true：测试数据的真实分类 如‘open’ ‘close’
    # y_pred：测试数据预测的分类
    # labels：分类报告中显示的类标签的索引列表
    # target_names：显示与labels对应的名称
    # digits：指定输出格式的精确度，默认3维
# 精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
# 召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
# F1 = 2*精度*召回率/(精度+召回率)
