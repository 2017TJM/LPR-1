import numpy as np
import os
import random
from PIL import Image
import string
# from sklearn import svm
import time
# from sklearn.externals import joblib

classfies = []

def get_labels():
    dict_labels = {}
    list_labels = []
    for i in range(10):
        dict_labels[i] = i
    for word in string.ascii_uppercase:
        dict_labels[word] = word
    dict_labels["川"] = 'zh_cuan'
    dict_labels["鄂"] = 'zh_e'
    dict_labels["赣"] = 'zh_gan'
    dict_labels["甘"] = 'zh_gan1'
    dict_labels["贵"] = 'zh_gui'
    dict_labels["桂"] = 'zh_gui1'
    dict_labels["黑"] = 'zh_hei'
    dict_labels["沪"] = 'zh_hu'
    dict_labels["冀"] = 'zh_ji'
    dict_labels["津"] = 'zh_jin'
    dict_labels["京"] = 'zh_jing'
    dict_labels["吉"] = 'zh_jl'
    dict_labels["辽"] = 'zh_liao'
    dict_labels["鲁"] = 'zh_lu'
    dict_labels["蒙"] = 'zh_meng'
    dict_labels["闽"] = 'zh_min'
    dict_labels["宁"] = 'zh_ning'
    dict_labels["青"] = 'zh_qing'
    dict_labels["琼"] = 'zh_qiong'
    dict_labels["陕"] = 'zh_shan'
    dict_labels["苏"] = 'zh_su'
    dict_labels["晋"] = 'zh_sx'
    dict_labels["皖"] = 'zh_wan'
    dict_labels["湘"] = 'zh_xiang'
    dict_labels["新"] = 'zh_xin'
    dict_labels["豫"] = 'zh_yu'
    dict_labels["渝"] = 'zh_yu1'
    dict_labels["粤"] = 'zh_yue'
    dict_labels["云"] = 'zh_yun'
    dict_labels["藏"] = 'zh_zang'
    dict_labels["浙"] = 'zh_zhe'
    for value in dict_labels.values():
        list_labels.append(value)

    return dict_labels,list_labels

def get_label(_preds):
    dict_labels,list_labels = get_labels()
    results = [list(dict_labels.keys())[list(dict_labels.values()).index(
                list_labels[pred])]for pred in _preds]
    results = [str(re) for re in results]
    return results

def open_dir():
    global classfies
    for root, dirs, filename in os.walk('D:\workspace\EasyPR-python\data\easypr_train_data\chars', topdown=False):
        for dir in dirs:
            classfies.append(dir)


def init_data():
    open_dir()
    dict_labels,list_labels = get_labels()
    # print(len(list_labels))
    x_data = []
    y_data = []
    for index,label in enumerate(list_labels):
        print(label)
        for root,_,filename in os.walk(r'H:/毕业论文/car_recognition/chars/'+str(label),topdown=False):
            for name in filename:
                # print(name)
                img = Image.open(os.path.join(root,name)).convert('L')
                img = np.round(np.array(img,'i')/255)
                x_data.append(img)
                # print(os.path.join(root,name))
                y_data.append(index)
                # print(count)
    return x_data,y_data

def create_svm(dataMat, dataLabel, decision='ovo'):
    clf = svm.SVC(kernel='rbf',decision_function_shape=decision)
    clf.fit(dataMat, dataLabel)
    joblib.dump(clf,r'D:\workspace\car_card_recognition\model\train_model.m')
    return clf

def train():

    tst = time.clock()
    x_data, y_data = init_data()
    x_test = []
    y_test = []
    rans = [ i for i in range(len(x_data))]
    for i in range(1000):
        a = random.choice(rans)
        x_test.append(x_data[a])
        y_test.append(y_data[a])
    x_train = np.array(x_data).reshape([len(x_data), 400])
    y_train = np.array(y_data).reshape([len(y_data), 1])
    clf = create_svm(x_train,y_train)
    pre_st = time.clock()
    print("Training Finshed! {:.3f}".format((pre_st - tst)))
    x_test = np.array(x_test).reshape([len(x_test),400])
    y_test = np.array(y_test).reshape([len(y_test),1])
    score = clf.score(x_test,y_test)
    print("Trained Score {:.3f}%".format(score*100))

def prediction(imgs):
    clf = joblib.load(r'D:\workspace\EasyPR-python\models\svm\model\train_model.m')
    print("Read Model Ready!")
    pre_start = time.clock()
    _preds = []
    for img in imgs:
        image = np.round(img/255)
        image = image.reshape([1,400])
        _pred = clf.predict(image)
        _preds.append(_pred[0])
    results = get_label(_preds)
    pre_end = time.clock()
    print("Predtion:"+''.join(results))
    # print("Predtion spent time: {:.4f}s.".format((pre_end-pre_start)))

if __name__ == '__main__':
    #训练
    train()
    #预测
    # init_data()
