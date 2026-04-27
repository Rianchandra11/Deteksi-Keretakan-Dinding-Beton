### KONVOLUSI (CANNY)
img_canny = cv2.Canny(img_enhanced, 50, 150)
imgblur_canny = cv2.Canny(img_blur, 50, 150)
print("Canny Edge Detection berhasil!")

### THRESHOLDING
img_thresholded = img_canny.copy()
imgblur_thresholded = imgblur_canny.copy()
print("Thresholding berhasil!")