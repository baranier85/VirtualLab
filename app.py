import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- UI Setup ---
st.set_page_config(page_title="DIP Virtual Lab", layout="wide")
st.title("🔬 Digital Image Processing Virtual Lab")
st.sidebar.header("Lab Controls")

# --- 1. Experiment Selection ---
experiment = st.sidebar.selectbox(
    "Select an Experiment",
    ["Introduction", "Point Processing (Gamma)", "Edge Detection (Sobel)", "Thresholding"]
)

# --- 2. Image Upload ---
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    raw_image = Image.open(uploaded_file)
    img_array = np.array(raw_image)
    
    # Display Original vs Processed
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_array, use_container_width=True)

    # --- 3. Experiment Logic ---
    with col2:
        st.subheader("Processed Image")
        
        if experiment == "Point Processing (Gamma)":
            gamma = st.slider("Gamma Value (r)", 0.1, 5.0, 1.0)
            # Apply Gamma: s = c * r^gamma
            processed = np.array(255*(img_array/255)**gamma, dtype='uint8')
            st.image(processed, use_container_width=True)
            st.latex(r"s = cr^{\gamma}")

        elif experiment == "Edge Detection (Sobel)":
            k_size = st.select_slider("Kernel Size", options=[3, 5, 7])
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)
            processed = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
            st.image(processed, use_container_width=True)
        elif experiment == "Histogram Equalization":
            st.header("Experiment 2: Histogram Equalization")
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Global Histogram Equalization
            equalized = cv2.equalizeHist(gray)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Original Grayscale", use_container_width=True)
                st.bar_chart(np.histogram(gray, bins=256, range=(0, 255))[0])
                
            with col2:
                st.image(equalized, caption="Equalized Image", use_container_width=True)
                st.bar_chart(np.histogram(equalized, bins=256, range=(0, 255))[0])
                    
else:
    st.info("Please upload an image from the sidebar to begin.")
