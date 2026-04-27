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
        enhanced = image_enhancement(gray)
        canny = cv2.Canny(enhanced, 30, 100)

        # Morphology
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closing, kernel, iterations=1)

        # Contour
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        valid_contours = []

        for cnt in contours:
            length = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)

            if area < 30:
                continue

            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = w / (h + 1e-5)

            features.append([length, area, aspect_ratio])
            valid_contours.append(cnt)

        if len(features) < 2:
            st.warning("Contour terlalu sedikit ⚠️")
            return original, original

        X = np.array(features)

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Clustering
        if method == "K-Means":
            model = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = model.fit_predict(X_scaled)

            means = [X[labels==i][:,0].mean() for i in range(k)]
            crack_cluster = int(np.argmax(means))

        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

            unique = [l for l in set(labels) if l != -1]
            if len(unique) == 0:
                crack_cluster = -1
            else:
                means = [X[labels==i][:,0].mean() for i in unique]
                crack_cluster = unique[int(np.argmax(means))]

        # Visualisasi
        vis = original.copy()

        for i, cnt in enumerate(valid_contours):
            if labels[i] == crack_cluster:
                color = (0,0,255)  # merah
            else:
                color = (255,0,0)  # biru

            cv2.drawContours(vis, [cnt], -1, color, 2)

            x,y,w,h = cv2.boundingRect(cnt)
            length = int(X[i][0])
            area = int(X[i][1])

            cv2.putText(vis, f"L:{length} A:{area}",
                        (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1)

        return original, vis

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None


# MAIN
if uploaded_file is None:
    st.info("Silakan upload gambar dulu 👆")
else:
    file_bytes = uploaded_file.read()

    original, result = process_image(file_bytes)

    if original is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Gambar Asli", channels="BGR")

        with col2:
            st.image(result, caption="Hasil Clustering", channels="BGR")