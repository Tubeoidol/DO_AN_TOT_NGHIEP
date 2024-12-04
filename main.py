import os
import shutil
import cv2
import numpy as np

digit_w = 30
digit_h = 60


def takeSecond(elem):
    return elem[1]


def takeFirst(elem):
    return elem[0]


def takeChar(elem):
    return elem[2]


def Pretreatment(imgLP):
    grayImg = cv2.cvtColor(imgLP, cv2.COLOR_BGR2GRAY)
    ret, binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    binImg = cv2.morphologyEx(binImg, cv2.MORPH_DILATE, kerel3)
    return binImg


def contours_detect(binImg):
    cnts, _ = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return cnts


def draw_rects_on_img(img, cnts):
    imgtemp = img.copy()
    cv2.drawContours(imgtemp, cnts, -1, (0, 120, 0), 1)
    return imgtemp


plate_number = ''
coorarr = []
firstrow = []
lastrow = []
model_svm = cv2.ml.SVM_load('dataset/trained/svm.xml')
plate_cascade = cv2.CascadeClassifier("dataset/detect_resource/cascade.xml")


def find_number(cnts, binImg, imgtemp):
    count = 0
    global plate_number
    global coorarr
    global firstrow
    global lastrow
    (himg, wimg, chanel) = imgtemp.shape
    if (wimg / himg > 2):
        hf = himg
        hl = himg * 0.8
    else:
        hf = 0.3 * himg
        hl = 0.5 * himg
    plate_number = ''
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if h / w > 1.5 and h / w < 4 and cv2.contourArea(c) > 4000 and h <= hl:
            cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop = imgtemp[y:y + h, x:x + w]
            count += 1
            cv2.imwrite('dataset/number/number%d.jpg' % count, crop)
            binImgtemp = binImg
            curr_num = binImgtemp[y:y + h, x:x + w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
            curr_num = np.array(curr_num, dtype=np.float32)
            curr_num = curr_num.reshape(-1, digit_w * digit_h)
            result = model_svm.predict(curr_num)[-1]
            result = int(result[0, 0])
            if result <= 9:
                result = str(result)
            else:
                result = chr(result)
            coorarr.append((x, y, result))
            cv2.putText(imgtemp, result, (x - 50, y + 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
    coorarr.sort(key=takeSecond)
    firstrow = coorarr[:4]
    firstrow.sort(key=takeFirst)
    lastrow = coorarr[4:]
    lastrow.sort(key=takeFirst)
    for x, y, c in firstrow:
        plate_number += c
    for x, y, c in lastrow:
        plate_number += c
    return imgtemp, plate_number


def detect(img):
    (himg, wimg, chanel) = img.shape
    if (wimg / himg > 2):
        img = cv2.resize(img, dsize=(1000, 200))
    else:
        img = cv2.resize(img, dsize=(800, 500))
    binImg = Pretreatment(img)
    cnts = contours_detect(binImg)
    imgtemp = draw_rects_on_img(img, cnts)
    imgtemp2, sort_number = find_number(cnts, binImg, imgtemp)
    cv2.imshow('binary', binImg)
    cv2.imshow('result', imgtemp)
    print('License plates: ', sort_number)
    plate_number = ''
    coorarr.clear()
    return sort_number


def findLP_img(OriImg):
    shutil.rmtree('dataset/number', ignore_errors=True)
    os.mkdir('dataset/number')
    plates = plate_cascade.detectMultiScale(OriImg, 1.1, 3)
    img = OriImg
    for (x, y, w, h) in plates:
        cv2.rectangle(OriImg, (x, y), (x + w, y + h), (255, 0, 0), 1)
        img = OriImg[y:y + h, x:x + w]
        plate_num = detect(img)
        cv2.putText(OriImg, plate_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow("Original image", OriImg)
    return img, plate_num


if __name__ == "__main__":
    OriImg = cv2.imread('dataset/test/anh_nghieng_ro.jpg', 1)
    img, plate_num = findLP_img(OriImg)
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
