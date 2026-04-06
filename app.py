import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- UI Setup ---
st.set_page_config(page_title="DIP Virtual Lab", layout="wide")
st.title("🔬 Digital Image Processing Virtual Lab")
st.sidebar.header("Lab Controls")

# Use tabs to organize the workspace
tab1, tab2, tab3 = st.tabs(["🏗️ Experiment Workspace", "📚 Theory & Math", "✍️ Final Quiz"])

with tab1:
    # Move all your if experiment == "..." code here
    st.write("Adjust parameters in the sidebar to see results.")

with tab2:
    st.subheader("The Mathematics of DIP")
    st.write("Here you can add images of formulas or diagrams.")
    st.header("Mathematical Foundations")
    theory_choice = st.selectbox("Select Theory to Review", 
                                ["Gamma Correction", "Histogram Equalization", "Mean Filter", "Thresholding"])

    if theory_choice == "Gamma Correction":
        st.subheader("Power-Law (Gamma) Transformations")
        st.write("Gamma correction is used to capture the non-linear relationship between pixel value and the perceived brightness.")
        st.latex(r"s = c \cdot r^{\gamma}")
        st.markdown("""
        - **r**: Input gray level.
        - **s**: Output gray level.
        - **c**: A constant (usually 1).
        - **γ < 1**: Expands dark regions (brightens image).
        - **γ > 1**: Compresses dark regions (darkens image).
        """)

    elif theory_choice == "Histogram Equalization":
        st.subheader("Histogram Equalization (HE)")
        st.write("HE is a spatial domain method that redistributes the probability distribution of pixels.")
        st.latex(r"s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p_r(r_j)")
        st.markdown("""
        Where $p_r(r_j)$ is the probability of occurrence of intensity level $r_j$. This formula represents the **Cumulative Distribution Function (CDF)**.
        """)
        

    elif theory_choice == "Mean Filter":
        st.subheader("Linear Smoothing (Mean Filter)")
        st.write("The mean filter is a sliding window spatial filter that replaces the center value with the average of all pixel values in the kernel window.")
        st.latex(r"g(x,y) = \frac{1}{M \times N} \sum_{i \in S} \sum_{j \in S} f(i,j)")
        st.markdown("""
        This filter is a **Low Pass Filter**, meaning it removes high-frequency noise but also blurs sharp edges.
        """)
        

    elif theory_choice == "Thresholding":
        st.subheader("Image Segmentation via Thresholding")
        st.write("Thresholding creates a binary image based on an intensity constant $T$.")
        st.latex(r"g(x,y) = \begin{cases} 255 & \text{if } f(x,y) > T \\ 0 & \text{if } f(x,y) \leq T \end{cases}")
        st.write("In **Otsu's Method**, the threshold $T$ is chosen to minimize the within-class variance.")
    # Example: 
    st.latex(r"g(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x+s, y+t)")

with tab3:
    # Move the Quiz code here
    st.write("Complete this to finish the lab.")

# --- 1. Experiment Selection ---
experiment = st.sidebar.selectbox(
    "Select an Experiment",
    ["Introduction", "Point Processing (Gamma)", "Edge Detection (Sobel)", "Histogram Equalization", "Thresholding", "Mean Filtering"]
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
        elif experiment == "Image Restoration (Mean Filter)":
            st.header("Experiment 6: Noise Reduction using Mean Filter")
    
            # Step 1: Add Synthetic Noise (Gaussian Noise)
            st.subheader("1. Add Noise to Image")
            noise_level = st.slider("Select Noise Intensity", 0, 100, 25)
    
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gauss_noise = np.zeros(gray.shape, dtype=np.uint8)
            cv2.randn(gauss_noise, 0, noise_level)
            noisy_img = cv2.add(gray, gauss_noise)
    
            # Step 2: Apply Mean Filter (Blur)
            st.subheader("2. Restore using Mean Filter")
            kernel_size = st.select_slider("Select Kernel Size (N x N)", options=[3, 5, 7, 9, 11])
    
            # cv2.blur implements the normalized box filter (Mean Filter)
            restored_img = cv2.blur(noisy_img, (kernel_size, kernel_size))
    
            # Step 3: Visualization
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gray, caption="Original", use_container_width=True)
            with col2:
                st.image(noisy_img, caption="Noisy Image", use_container_width=True)
            with col3:
                st.image(restored_img, caption="Restored (Mean Filter)", use_container_width=True)

            # Step 4: Quantitative Analysis
            mse = np.mean((gray - restored_img) ** 2)
            st.metric("Mean Squared Error (MSE)", round(mse, 2))
            st.info("A lower MSE indicates a better restoration. Notice how a larger kernel reduces noise but also blurs the edges.")
                    
else:
    st.info("Please upload an image from the sidebar to begin.")
