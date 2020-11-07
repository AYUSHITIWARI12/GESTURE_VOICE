import cnn_sgn
import cv2
import numpy as np
import os
from random import shuffle
IMG_SIZE = 96
LR = 1e-3

def hello():
    path = 'data'
    IMG_SIZE = 96

    def create_train_data():
        training_data = []
        label = 0
        for (dirpath, dirnames, filenames) in os.walk(path):
            for dirname in dirnames:
                print(dirname)
                for (direcpath, direcnames, files) in os.walk(path + "/" + dirname):
                    for file in files:
                        actual_path = path + "/" + dirname + "/" + file
                        print(files)

                        path1 = path + "/" + dirname + '/' + file
                        img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img), label])
                label = label + 1
                print(label)
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        print(training_data)
        return training_data

    return create_train_data()


nb_classes=28

MODEL_NAME = 'handsign.model'

def one_hot_targets_(labels_dense,nb_classes):
    targets = np.array(Y).reshape(-1)
    print(targets)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets

train_data = hello()


train = train_data[:]
test = train_data[:100]

print('traindatlen:'+str(len(train)))
print('testdatalen:'+str(len(test)))

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
Y1=one_hot_targets_(Y,nb_classes)

print('val y'+str(Y1))
print('len X:'+str(len(X)))
print('len Y:'+str(len(Y)))
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
test_y1=one_hot_targets_(test_y,nb_classes)
test_y=test_y1
Y=Y1
print('test_x:'+str(len(test_x)))
print('test_y:'+str(len(test_y)))
print('val y'+str(test_y1))

model=cnn_sgn.cnn_model()

model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
score = model.evaluate(test_x, test_y)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
