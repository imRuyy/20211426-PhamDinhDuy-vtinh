import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh đầu vào
image = cv2.imread("vetinh.jpg")

# Chuyển ảnh sang xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng bộ lọc Gaussian để làm mượt ảnh, giảm nhiễu
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 1. Toán tử Sobel
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)  # Tính độ lớn gradient tổng hợp

# 2. Toán tử Prewitt (bằng cách sử dụng kernel của Prewitt)
kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(blurred_image, cv2.CV_32F, kernel_prewitt_x)  # Đảm bảo CV_32F
prewitt_y = cv2.filter2D(blurred_image, cv2.CV_32F, kernel_prewitt_y)  # Đảm bảo CV_32F

# Tính toán độ lớn Prewitt
prewitt = cv2.magnitude(prewitt_x, prewitt_y)


# 3. Toán tử Roberts
kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(blurred_image, cv2.CV_32F, kernel_roberts_x)  # Đảm bảo CV_32F
roberts_y = cv2.filter2D(blurred_image, cv2.CV_32F, kernel_roberts_y)  # Đảm bảo CV_32F

# Tính toán độ lớn Roberts
roberts = cv2.magnitude(roberts_x, roberts_y)


# 4. Phát hiện biên Canny
canny = cv2.Canny(blurred_image, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(10, 8))

plt.subplot(231), plt.imshow(gray_image, cmap='gray'), plt.title('Ảnh gốc')
plt.subplot(232), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.subplot(233), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
plt.subplot(234), plt.imshow(roberts, cmap='gray'), plt.title('Roberts')
plt.subplot(235), plt.imshow(canny, cmap='gray'), plt.title('Canny')

plt.tight_layout()
plt.show()
