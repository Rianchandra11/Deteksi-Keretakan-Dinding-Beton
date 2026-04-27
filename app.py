import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

st.title("Deteksi Retakan (Clustering)")

# Upload
uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

# Pilih metode clustering
method = st.selectbox("Pilih Clustering", ["K-Means", "DBSCAN"])

# Parameter
if method == "K-Means":
    k = st.slider("Jumlah Cluster (K)", 2, 4, 2)
else:
    eps = st.slider("EPS (DBSCAN)", 0.1, 1.5, 0.5)
    min_samples = st.slider("Min Samples", 2, 10, 3)

def grayscale(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray
def blurimg(img_gray):
    blur = cv2.medianBlur(img_gray, 21,0)
    return blur
def image_enhancement(img_gray):
    clahe = cv2.createCLAHE(2.0,(8,8))
    enhanced = clahe.apply(img_gray)
    return enhanced

def process_image(file_bytes):
    try:
        file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Gagal membaca gambar")
            return None, None

        original = img.copy()

        gray = grayscale(img)
        blur = blurimg(gray)
        enhanced = image_enhancement(blur)
        canny = cv2.Canny(enhanced, 50, 150)

        # Morphology
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closing, kernel, iterations=7)

        # Contour
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_close = np.zeros_like(dilated)
        total_contour = 0
        detect = img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                if(cv2.arcLength(cnt, True)) > 100:
                    total_contour +=1
                    print(f"What is : {cv2.contourArea(cnt)}")
                    pjg = cv2.arcLength(cnt, True)
                    print(cv2.arcLength(cnt, True))
                    cv2.drawContours(detect,[cnt], 0,255,0)
                    cv2.drawContours(clean_close, [cnt], -1, 255, -1)

     
        

        

        return original,closing,dilated,detect

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None


# MAIN
if uploaded_file is None:
    st.info("Silakan upload gambar dulu 👆")
else:
    file_bytes = uploaded_file.read()

    original,clos,dil,result = process_image(file_bytes)

    if original is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(original, caption="Gambar Asli", channels="BGR")
        with col2:
            st.image(clos, caption= "Closing")
        with col3: 
            st.image(dil,caption="Dilation (Closing)")
        with col4:
            st.image(result, caption="Hasil Contour", channels="BGR")