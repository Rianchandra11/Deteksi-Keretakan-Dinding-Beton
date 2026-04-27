import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from pathlib import Path
import kagglehub

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


#Inisialisasi Kernel
kernel = np.ones((3,3), np.uint8)

#Operasi Opening
opening = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, kernel)
opening_blur = cv2.morphologyEx(imgblur_thresholded, cv2.MORPH_OPEN, kernel)

#Operasi Closing
closing = cv2.morphologyEx(img_thresholded, cv2.MORPH_CLOSE, kernel)
closing_blur = cv2.morphologyEx(imgblur_thresholded, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(6, 4, figsize=(16, 18))

# Baris 1  : Gambar awal (RGB)
img_original = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)

# Operasi Dilasi
dilation_opening = cv2.dilate(opening, kernel, iterations=1)
dilation_opening_blur = cv2.dilate(opening_blur, kernel, iterations=1)

dilation_closing = cv2.dilate(closing, kernel, iterations=1)
dilation_closing_blur = cv2.dilate(closing_blur, kernel, iterations=7)

axes[0,0].imshow(img_original)
axes[0,0].set_title("Gambar Asli")
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')

# Baris 2 : Gambar Grayscale
axes[1,0].imshow(img_gray, cmap='gray')
axes[1,0].set_title("Grayscale")
axes[1,0].axis('off')

axes[1,1].imshow(img_blur, cmap='gray')
axes[1,1].set_title("Grayscale + Blur")
axes[1,1].axis('off')

axes[1,2].axis('off')
axes[1,3].axis('off')


# Gambar Enhancement (Tidak Blur dan Blur)
axes[2,0].imshow(img_enhanced, cmap='gray')
axes[2,0].set_title("Grayscale + Blur")
axes[2,0].axis('off')

axes[2,1].imshow(imgenhancedblur, cmap='gray')
axes[2,1].set_title("Grayscale + Blur + Enhancement")
axes[2,1].axis('off')

axes[2,2].axis('off')
axes[2,3].axis('off')


# Baris 3 : Konvolusi dan Threshold
axes[3,0].imshow(img_sobel_combined, cmap='gray')
axes[3,0].set_title("Sobel")
axes[3,0].axis('off')

axes[3,1].imshow(imgblur_sobel_combined, cmap='gray')
axes[3,1].set_title("Sobel + Blur")
axes[3,1].axis('off')

axes[3,2].imshow(img_thresholded, cmap='gray')
axes[3,2].set_title("Threshold")
axes[3,2].axis('off')

axes[3,3].imshow(imgblur_thresholded, cmap='gray')
axes[3,3].set_title("Threshold + Blur")
axes[3,3].axis('off')


# Baris 4 : Perbedaan Opening Dan Closing
axes[4,0].imshow(opening, cmap='gray')
axes[4,0].set_title("Opening")
axes[4,0].axis('off')

axes[4,1].imshow(opening_blur, cmap='gray')
axes[4,1].set_title("Opening + Blur")
axes[4,1].axis('off')

axes[4,2].imshow(closing, cmap='gray')
axes[4,2].set_title("Closing")
axes[4,2].axis('off')


axes[4,3].imshow(closing_blur, cmap='gray')
axes[4,3].set_title("Closing + Blur")
axes[4,3].axis('off')
# Baris 5 : Dilasi
axes[5,0].imshow(dilation_opening, cmap='gray')
axes[5,0].set_title("Dilasi (Opening)")
axes[5,0].axis('off')

axes[5,1].imshow(dilation_opening_blur, cmap='gray')
axes[5,1].set_title("Dilasi (Opening + Blur)")
axes[5,1].axis('off')

axes[5,2].imshow(dilation_closing, cmap='gray')
axes[5,2].set_title("Dilasi (Closing)")
axes[5,2].axis('off')

axes[5,3].imshow(dilation_closing_blur, cmap='gray')
axes[5,3].set_title("Dilasi (Closing + Blur)")
axes[5,3].axis('off')

plt.suptitle("Tahapan Preprocessing hingga Morphology", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()