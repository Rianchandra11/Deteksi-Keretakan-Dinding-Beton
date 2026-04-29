import streamlit as st
import cv2
import numpy as np

st.title("Deteksi Retakan Dinding Beton")

# Upload
uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

def grayscale(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray
def blurimg(img_gray):
    blur = cv2.GaussianBlur(img_gray, (9,9),0)
    return blur
def image_enhancement(img_blur):
    clahe = cv2.createCLAHE(2.0,(8,8))
    enhanced = clahe.apply(img_blur)
    return enhanced
def process_image(file_bytes):
    
    try:
        file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Gagal membaca gambar")
            return None, None, None, None, None, None, None, None, None, 0

        original = img.copy()

        gray = grayscale(img)
        blur = blurimg(gray)
        enhanced = image_enhancement(blur)
        canny = cv2.Canny(enhanced, 50, 150)

        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closing, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ket = []
        detect = img.copy()
        total_contour = 0
        for cnt in contours:
            lengths = cv2.arcLength(cnt,True)
            area = cv2.contourArea(cnt)
            if area > 100 and lengths > 150:
                length = int(cv2.arcLength(cnt, True))
                
                cv2.drawContours(detect, [cnt], -1, (0,0,0), 2)
                x, y, w, h = cv2.boundingRect(cnt)

                posisi_x = x - 35 
                posisi_y = y -35 

                
                if posisi_x < 0: posisi_x = 5
                if posisi_y <0 : posisi_y = 0
                cv2.putText(detect, f"{total_contour+1}", (posisi_x, posisi_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.putText(detect, f"#{total_contour+1}", 
                #             (cX, cY + offset_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                ket.append(f"Retakan ke - #{total_contour+1} : {int(length)} pixel")
                total_contour += 1
                
        return original, gray, blur, enhanced, canny, closing, dilated, detect,ket,  total_contour

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None, None, None, None, 0


if uploaded_file is None:
    st.info("Silakan upload gambar terlebih dahulu")
else:
    file_bytes = uploaded_file.read()

    original, gray, blur, enhanced, canny, closing, dilated, detect, ket, total_contour = process_image(file_bytes)
    st.divider()
    st.subheader("Image Preprocessing")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(original, caption="Gambar Awal", channels="BGR")
    with col2:
        st.image(gray, caption="Grayscale")
    with col3:
        st.image(blur, caption="Gaussian Blur (9x9)")
    with col4:
        st.image(enhanced, caption="Enhancement")
    st.divider()
    st.subheader("Canny, Morphology & Contour")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.image(canny, caption="Canny")
    with col6:
        st.image(closing, caption="Closing (3x3)")
    with col7:
        st.image(dilated, caption="Dilation (3x3)")
    with col8:
        st.image(detect, caption="Contour", channels="BGR")
    st.divider()
    st.subheader(f"Detail Panjang Retakan ({total_contour})")
    if total_contour > 0:
        for k in ket:
            st.write(k)
    else:
        st.write("Tidak ada retakan yang terdeteksi")