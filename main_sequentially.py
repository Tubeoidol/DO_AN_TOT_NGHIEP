import cv2
import numpy as np

digit_w = 30
digit_h = 60

model_svm = cv2.ml.SVM_load('dataset/trained/svm.xml')

OriImg = cv2.imread('dataset/test/anh_nghieng_ro.jpg', 1)
plate_cascade = cv2.CascadeClassifier("dataset/detect_resource/cascade.xml")
plates = plate_cascade.detectMultiScale(OriImg, 1.1, 3)
img = OriImg

for (x, y, w, h) in plates:
    cv2.rectangle(OriImg, (x, y), (x + w, y + h), (255, 0, 0), 1)
    img = OriImg[y:y + h, x:x + w]

cv2.imshow("Original image", OriImg)
cv2.imshow("crop", img)

(himg, wimg, chanel) = img.shape

if (wimg / himg > 2):
    img = cv2.resize(img, dsize=(1000, 200))
else:
    img = cv2.resize(img, dsize=(800, 500))

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(grayImg, 9, 75, 75)
ret, binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
binImg = cv2.morphologyEx(binImg, cv2.MORPH_DILATE, kerel3)

cnts, _ = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
imgtemp = img.copy()
cv2.drawContours(imgtemp, cnts, -1, (0, 120, 0), 1)

cv2.imshow('Khoa', imgtemp)

plate_number = ''
count = 0
coorarr = []

for c in (cnts):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if h / w > 1.5 and h / w < 4 and cv2.contourArea(c) > 4500:
        cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 0, 255), 2)
        crop = img[y:y + h, x:x + w]
        count += 1
        cv2.imwrite('dataset/number/number%d.jpg' % count, crop)
        coorarr.append((x, y))

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
        plate_number += result + ' '
        cv2.putText(imgtemp, result, (x - 50, y + 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

stringarr = plate_number.strip()
stringarr = stringarr.split(" ")

for i in range(len(coorarr)):
    for j in range(i + 1, len(coorarr)):
        if coorarr[i][1] - coorarr[j][1] > 15:
            temp = stringarr[i]
            stringarr[i] = stringarr[j]
            stringarr[j] = temp
            tempp = coorarr[i]
            coorarr[i] = coorarr[j]
            coorarr[j] = tempp
        elif coorarr[i][0] - coorarr[j][0] > 0:
            temp = stringarr[i]
            stringarr[i] = stringarr[j]
            stringarr[j] = temp
            tempp = coorarr[i]
            coorarr[i] = coorarr[j]
            coorarr[j] = tempp

plate_number = ''.join(stringarr)
print('bien so xe: ', plate_number)

cv2.imshow('binary', binImg)
cv2.imshow('result', imgtemp)

cv2.waitKey()
cv2.destroyAllWindows()
