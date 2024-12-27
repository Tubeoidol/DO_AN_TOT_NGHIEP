import glob
import cv2
import numpy as np

digit_w = 30
digit_h = 60

write_path = "dataset/train/"


def get_digit_data(path):  #: digit_list, label_list:

    digit_list = []
    label_list = []

    for number in range(10):

        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)

            # print(img.shape)

            digit_list.append(img)
            label_list.append([int(number)])

    for number in range(65, 91):
        print(number)

        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)

            # print(img.shape)

            digit_list.append(img)
            label_list.append([int(number)])

    return digit_list, label_list


# lấy dữ liệu
digit_path = "dataset/train/"
digit_list, label_list = get_digit_data(digit_path)

digit_list = np.array(digit_list, dtype=np.float32)
digit_list = digit_list.reshape(-1, digit_h * digit_w)

label_list = np.array(label_list)
label_list = label_list.reshape(-1, 1)

svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)
svm_model.setKernel(cv2.ml.SVM_INTER)
svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm_model.train(digit_list, cv2.ml.ROW_SAMPLE, label_list)

svm_model.save("svm.xml")



# đánh giá mô hình
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    digit_list, label_list, test_size=0.2, random_state=42
)

# Huấn luyện lại mô hình trên tập huấn luyện
svm_model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Dự đoán trên tập kiểm tra
_, y_pred = svm_model.predict(X_test)

# Chuyển đổi nhãn về định dạng phù hợp
y_test = y_test.flatten()
y_pred = y_pred.flatten()

# Tính các chỉ số
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' cho nhiều lớp
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# In kết quả
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
