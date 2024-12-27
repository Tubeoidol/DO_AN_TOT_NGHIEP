import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

digit_w = 30
digit_h = 60
write_path = "dataset/train/"

def get_digit_data(path):
    digit_list = []
    label_list = []

    for number in range(10):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = img.reshape(-1, digit_h * digit_w)
            digit_list.append(img)
            label_list.append(int(number))  # Thêm nhãn trực tiếp dạng số nguyên

    for number in range(65, 91):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            print(img_org_path)
            img = cv2.imread(img_org_path, 0)
            img = img.reshape(-1, digit_h * digit_w)
            digit_list.append(img)
            label_list.append(int(number))  # Thêm nhãn trực tiếp dạng số nguyên

    return digit_list, label_list

# Lấy dữ liệu
digit_path = "dataset/train/"
digit_list, label_list = get_digit_data(digit_path)

digit_list = np.array(digit_list, dtype=np.float32).reshape(-1, digit_h * digit_w)
label_list = np.array(label_list)  # Giữ dạng 1D, không cần reshape(-1, 1)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    digit_list, label_list, test_size=0.2, random_state=42
)

# Huấn luyện mô hình KNN với scikit-learn
knn = KNeighborsClassifier(n_neighbors=3)  # Chọn k=3
knn.fit(X_train, y_train)

# Tạo mô hình KNN bằng OpenCV
knn_opencv = cv2.ml.KNearest_create()
# Huấn luyện mô hình KNN OpenCV
knn_opencv.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
# Lưu mô hình KNN OpenCV dưới dạng .xml
knn_opencv.save("knn_model.xml")
# Dự đoán trên tập kiểm tra sử dụng mô hình OpenCV
ret, results, neighbours, dist = knn_opencv.findNearest(X_test, 3)
# Chuyển đổi y_test và y_pred từ mảng 2D sang 1D
y_test = y_test.flatten()
y_pred = results.flatten()
# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# In kết quả
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
