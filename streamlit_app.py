import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Function to display images side by side
def display_side_by_side(original, processed, title_original="Original Image", title_processed="Processed Image"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    if len(original.shape) == 2:  # Grayscale
        ax1.imshow(original, cmap='gray')
        ax2.imshow(processed, cmap='gray')
    else:  # Color
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if len(processed.shape) == 3 else processed, cmap='gray' if len(processed.shape) == 2 else None)
    ax1.set_title(title_original)
    ax2.set_title(title_processed)
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    return fig

# Streamlit app
st.title("🖼️ Image Processing Playground")
st.write("Upload an image and apply various image processing techniques interactively.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    # Processing technique selection
    technique = st.selectbox(
        "Select Image Processing Technique",
        [
            "Thresholding",
            "Blurring & Smoothing",
            "Edge Detection",
            "Contour Detection",
            "Template Matching",
            "Watershed Segmentation",
            "Color Space Conversion",
            "Image Operations"
        ]
    )

    if technique == "Thresholding":
        st.subheader("Thresholding")
        thresh_type = st.selectbox(
            "Select Thresholding Method",
            ["Simple Thresholding", "Adaptive Thresholding", "Otsu's Thresholding"]
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if thresh_type == "Simple Thresholding":
            thresh_value = st.slider("Threshold Value", 0, 255, 127)
            thresh_mode = st.selectbox(
                "Threshold Type",
                ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
            )
            thresh_modes = {
                "BINARY": cv2.THRESH_BINARY,
                "BINARY_INV": cv2.THRESH_BINARY_INV,
                "TRUNC": cv2.THRESH_TRUNC,
                "TOZERO": cv2.THRESH_TOZERO,
                "TOZERO_INV": cv2.THRESH_TOZERO_INV
            }
            ret, thresh = cv2.threshold(gray, thresh_value, 255, thresh_modes[thresh_mode])
            fig = display_side_by_side(gray, thresh, "Grayscale", f"Simple Thresholding ({thresh_mode})")
            st.pyplot(fig)

        elif thresh_type == "Adaptive Thresholding":
            block_size = st.slider("Block Size (odd)", 3, 21, 11, step=2)
            C = st.slider("Constant C", -10, 10, 2)
            adapt_method = st.selectbox("Adaptive Method", ["Mean", "Gaussian"])
            adapt_methods = {
                "Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
                "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            }
            thresh = cv2.adaptiveThreshold(
                gray, 255, adapt_methods[adapt_method], cv2.THRESH_BINARY, block_size, C
            )
            fig = display_side_by_side(gray, thresh, "Grayscale", f"Adaptive {adapt_method} Thresholding")
            st.pyplot(fig)

        elif thresh_type == "Otsu's Thresholding":
            apply_gaussian = st.checkbox("Apply Gaussian Blur Before Otsu")
            if apply_gaussian:
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fig = display_side_by_side(gray, thresh, "Grayscale", "Otsu's Thresholding (with Gaussian)")
            else:
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fig = display_side_by_side(gray, thresh, "Grayscale", "Otsu's Thresholding")
            st.pyplot(fig)

    elif technique == "Blurring & Smoothing":
        st.subheader("Blurring & Smoothing")
        blur_type = st.selectbox(
            "Select Blur Type",
            ["Averaging", "Gaussian", "Median", "Custom 2D Convolution"]
        )

        if blur_type == "Averaging":
            kernel_size = st.slider("Kernel Size (odd)", 3, 25, 5, step=2)
            blur = cv2.blur(img, (kernel_size, kernel_size))
            fig = display_side_by_side(img, blur, "Original", "Averaging Blur")
            st.pyplot(fig)

        elif blur_type == "Gaussian":
            kernel_size = st.slider("Kernel Size (odd)", 3, 25, 5, step=2)
            sigma = st.slider("Sigma", 0.0, 10.0, 0.0)
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            fig = display_side_by_side(img, blur, "Original", "Gaussian Blur")
            st.pyplot(fig)

        elif blur_type == "Median":
            kernel_size = st.slider("Kernel Size (odd)", 3, 25, 5, step=2)
            blur = cv2.medianBlur(img, kernel_size)
            fig = display_side_by_side(img, blur, "Original", "Median Blur")
            st.pyplot(fig)

        elif blur_type == "Custom 2D Convolution":
            kernel_size = st.slider("Kernel Size (odd)", 3, 25, 7, step=2)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            dst = cv2.filter2D(img, -1, kernel)
            fig = display_side_by_side(img, dst, "Original", "Custom 2D Convolution")
            st.pyplot(fig)

    elif technique == "Edge Detection":
        st.subheader("Edge Detection")
        edge_type = st.selectbox("Select Edge Detection Method", ["Sobel", "Laplacian", "Canny"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if edge_type == "Sobel":
            ksize = st.slider("Kernel Size (odd, or -1 for Scharr)", -1, 31, 5, step=2)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sobel_combined = cv2.convertScaleAbs(sobel_combined)
            fig = display_side_by_side(gray, sobel_combined, "Grayscale", "Sobel Edge Detection")
            st.pyplot(fig)

        elif edge_type == "Laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            fig = display_side_by_side(gray, laplacian, "Grayscale", "Laplacian Edge Detection")
            st.pyplot(fig)

        elif edge_type == "Canny":
            thresh1 = st.slider("Lower Threshold", 0, 255, 100)
            thresh2 = st.slider("Upper Threshold", 0, 255, 200)
            edges = cv2.Canny(gray, thresh1, thresh2)
            fig = display_side_by_side(gray, edges, "Grayscale", "Canny Edge Detection")
            st.pyplot(fig)

    elif technique == "Contour Detection":
        st.subheader("Contour Detection")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_value = st.slider("Threshold Value for Binary Image", 0, 255, 127)
        ret, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        fig = display_side_by_side(img, output, "Original", f"Contours ({len(contours)} detected)")
        st.pyplot(fig)

    elif technique == "Template Matching":
        st.subheader("Template Matching")
        template_file = st.file_uploader("Upload Template Image", type=["png", "jpg", "jpeg"])
        if template_file is not None:
            template_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
            template = cv2.imdecode(template_bytes, cv2.IMREAD_GRAYSCALE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            method = st.selectbox(
                "Matching Method",
                ["TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
            )
            method_val = getattr(cv2, method)
            res = cv2.matchTemplate(gray, template, method_val)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"] else max_loc
            w, h = template.shape[::-1]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            output = img.copy()
            cv2.rectangle(output, top_left, bottom_right, (0, 255, 0), 2)
            fig = display_side_by_side(img, output, "Original", f"Template Matching ({method})")
            st.pyplot(fig)

    elif technique == "Watershed Segmentation":
        st.subheader("Watershed Segmentation")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        dist_thresh = st.slider("Distance Transform Threshold", 0.0, 1.0, 0.7)
        ret, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        output = img.copy()
        output[markers == -1] = [255, 0, 0]
        fig = display_side_by_side(img, output, "Original", "Watershed Segmentation")
        st.pyplot(fig)

    elif technique == "Color Space Conversion":
        st.subheader("Color Space Conversion")
        color_space = st.selectbox("Target Color Space", ["Grayscale", "RGB", "HSV"])
        if color_space == "Grayscale":
            output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fig = display_side_by_side(img, output, "Original", "Grayscale")
        elif color_space == "RGB":
            output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig = display_side_by_side(img, output, "Original", "RGB")
        elif color_space == "HSV":
            output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            fig = display_side_by_side(img, output, "Original", "HSV")
        st.pyplot(fig)

    elif technique == "Image Operations":
        st.subheader("Image Operations")
        operation = st.selectbox("Select Operation", ["Resize", "Crop", "Mask", "Brightness/Contrast"])
        
        if operation == "Resize":
            scale = st.slider("Scale Factor", 0.1, 2.0, 0.5)
            output = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            fig = display_side_by_side(img, output, "Original", f"Resized (Scale: {scale})")
            st.pyplot(fig)

        elif operation == "Crop":
            y1 = st.slider("Y1", 0, img.shape[0], 0)
            y2 = st.slider("Y2", y1, img.shape[0], img.shape[0]//2)
            x1 = st.slider("X1", 0, img.shape[1], 0)
            x2 = st.slider("X2", x1, img.shape[1], img.shape[1]//2)
            output = img[y1:y2, x1:x2]
            fig = display_side_by_side(img, output, "Original", "Cropped")
            st.pyplot(fig)

        elif operation == "Mask":
            mask = np.zeros(img.shape[:2], dtype='uint8')
            x, y = st.slider("Center X", 0, img.shape[1], img.shape[1]//2), st.slider("Center Y", 0, img.shape[0], img.shape[0]//2)
            w, h = st.slider("Width", 10, img.shape[1], 100), st.slider("Height", 10, img.shape[0], 100)
            cv2.rectangle(mask, (x-w//2, y-h//2), (x+w//2, y+h//2), 255, -1)
            output = cv2.bitwise_and(img, img, mask=mask)
            fig = display_side_by_side(img, output, "Original", "Masked")
            st.pyplot(fig)

        elif operation == "Brightness/Contrast":
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
            output = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            fig = display_side_by_side(img, output, "Original", "Brightness/Contrast Adjusted")
            st.pyplot(fig)

    # Save processed image
    if st.button("Save Processed Image"):
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR) if len(output.shape) == 3 else output
        cv2.imwrite("processed_image.jpg", output_bgr)
        st.success("Image saved as 'processed_image.jpg'")