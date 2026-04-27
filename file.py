### KONVOLUSI (SOBEL)

# Sobel kernel untuk deteksi tepi horizontal (Gx)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Sobel kernel untuk deteksi tepi vertikal (Gy)
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Terapkan filter Sobel pada gambar yang sudah di-enhance
img_sobel_x = cv2.filter2D(img_enhanced, cv2.CV_64F, sobel_x)
img_sobel_y = cv2.filter2D(img_enhanced, cv2.CV_64F, sobel_y)

imgblur_sobel_x = cv2.filter2D(img_blur, cv2.CV_64F, sobel_x)
imgblur_sobel_y = cv2.filter2D(img_blur, cv2.CV_64F, sobel_y)

# Gabungkan Gx dan Gy
img_sobel_combined = cv2.magnitude(img_sobel_x, img_sobel_y)
imgblur_sobel_combined = cv2.magnitude(imgblur_sobel_x, imgblur_sobel_y)

# Normalisasi
img_sobel_combined = cv2.normalize(img_sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
imgblur_sobel_combined = cv2.normalize(imgblur_sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print("Sobel Convolution berhasil!")

### THRESHOLDING (OTS U)

_, img_thresholded = cv2.threshold(
    img_sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

_, imgblur_thresholded = cv2.threshold(
    imgblur_sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("Thresholding berhasil!")