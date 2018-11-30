import os
import scipy.io as sio
import numpy as np
def trainImage(RootPath):
    with open(RootPath +'/ingredients-101/annotations/train_images.txt','rb') as file:
        lines = file.readlines()
        writefile = open('./train.txt','ab')
        #len(lines)
        for line in lines:
            result = line[:-1] + b'.jpg' + b'\n'
            writefile.write(result)

    file.close()
    with open(RootPath + '/ingredients-101/annotations/train_labels.txt', 'rb') as f:
        loadlabels = f.readlines()
        with open('./train_label.txt', 'ab') as b:
            for line in loadlabels:
                b.write(line)
        b.close()
    f.close()

def testImage(RootPath):
    with open(RootPath +'/ingredients-101/annotations/test_images.txt','rb') as file:
        lines = file.readlines()
        writefile = open('./test.txt','ab')
        #len(lines)
        for line in lines:
            #print ("------")
            #print(line)
            result = line[:-1] + b'.jpg' + b'\n'
            writefile.write(result)

    file.close()
    with open(RootPath + '/ingredients-101/annotations/test_labels.txt', 'rb') as f:
        loadlabels = f.readlines()
        with open('./test_label.txt', 'ab') as b:
            for line in loadlabels:
                b.write(line)
        b.close()
    f.close()
def valImage(RootPath):
    with open(RootPath +'/ingredients-101/annotations/val_images.txt','rb') as file:
        lines = file.readlines()
        writefile = open('./val.txt','ab')
        #len(lines)
        for line in lines:
            #print ("------")
            #print(line)
            result = line[:-1] + b'.jpg' + b'\n'
            writefile.write(result)

    file.close()
    with open(RootPath + '/ingredients-101/annotations/val_labels.txt', 'rb') as f:
        loadlabels = f.readlines()
        with open('./val_label.txt', 'ab') as b:
            for line in loadlabels:
                b.write(line)
        b.close()
    f.close()

RootPath = os.getcwd()

trainImage(RootPath)
print('train finish')
testImage(RootPath)
print('test finish')
valImage(RootPath)
print ('val finish')
# Readmat(RootPath)
