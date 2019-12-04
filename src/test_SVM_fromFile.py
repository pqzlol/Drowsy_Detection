#coding=utf-8 

#测试训练模型

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report


#从两个txt文件中解析数据，提取出特征向量，放入列表test中，同时并把对应的标签放入列表labels中
test = []
labels = []

print('Reading test_open.txt...')

test_open_txt=open('test_open.txt',"r") 
for txt_str in test_open_txt.readlines():
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
    test.append(temp)
    labels.append('open')

print('Reading test_close.txt...')
test_close_txt=open('test_close.txt',"r") 
#读取全部内容 ，并以列表方式返回
for txt_str in test_close_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    for data in datas:
        data = float(data)
        temp.append(data)
    test.append(temp)
    labels.append('close')
    
count=0
for i in range(len(labels)):
    count+=1
    print("count:{}".format(count))
    print("{0} --> {1}".format(test[i], labels[i]))

test_close_txt.close()
test_open_txt.close()

# 导入模型
clf = joblib.load("ear_svm.m")

#测试训练集的准确率
print("accuracy:{}".format(clf.score(test,labels)))

# 输出更加详细的其他评价分类性能的指标。
print("acy_true:{}".format(classification_report(labels, clf.predict(test))))

print("acy:{}".format(classification_report(clf.predict(test),labels)))