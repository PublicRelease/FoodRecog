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
#im_pth = "./food-101/images/apple_pie/134.jpg"

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


def ratio(im_pth):
    print(im_pth)
    im = cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    #cv2.imshow("image", new_im)
    cv2.imwrite('thumb2.jpg',new_im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
def cut(im_pth):
    print(im_pth)
    im = Image.open(im_pth)
    (x,y) = im.size
    #cut_side = max(old_size)
    #index = old_size.index(cut_side)
    if x > y:
        region = (int(x / 2 - y / 2), 0, int(x / 2 + y / 2), y)
        crop_img = im.crop(region)
        #crop_img.save(self.outfile)
    elif x < y:
        region = (0, int(y / 2 - x / 2), x, int(y / 2 + x / 2))
        crop_img = im.crop(region)
        #crop_img.save(self.outfile)
    else:
        crop_img = im

    crop_img.thumbnail((desired_size, desired_size))
    crop_img.save(im_pth)


namelist = GetFileList('./food-101/images',[])
for im_pth in namelist:
    #ratio(im_pth)
    cut(im_pth)
