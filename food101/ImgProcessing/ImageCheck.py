import cv2
import os
from PIL import Image
import shutil
import os
# import imutils
# RootPath = os.getcwd()
# img = cv2.imread('./food-101/images/apple_pie/134.jpg' , 0)
# img = imutils.resize(img, width=1280)
# cv2.imshow('image' , img)

from PIL import Image, ImageOps
import cv2

desired_size = 256
def GetFileList(dir, fileList):

    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
def check(im_pth,k,l):
    l+=1
    if l%1000 ==0:
        print("l")
    im = cv2.imread(im_pth)
    h,w = im.shape[:2]
    #h = h
    if h!=desired_size or w!=desired_size:
        k+=1
        print("Catch!")
        print(im_pth)
        new_im = cv2.resize(im,(256,256))
        print(new_im.shape[:2])
        cv2.imwrite(im_pth,new_im)


namelist = GetFileList('./food-101/images',[])
k = 0
l = 0
print("start")
for im_pth in namelist:
    check(im_pth,k,l)