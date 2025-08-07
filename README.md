# 🛠️ Day 3 Mini-Project (SfAI) Step-by-Step Guide
Day 3 for SfAI CV Track [https://github.com/Society-For-AI/AI-Skill-Accelerator-Computer-Vision](https://github.com/Society-For-AI/AI-Skill-Accelerator-Computer-Vision)

---

## ✅ 1. Create and Activate a Conda Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n image_playground python=3.12 -y

# Activate the environment
conda activate image_playground
```

---

## ✅ 2. Install Dependencies

```bash
# Install core dependencies
pip install streamlit opencv-python numpy matplotlib
```

---

## ✅ 3. Create `streamlit_app.py` and Add Your Code

```bash
# Create a Python file
touch streamlit_app.py
```

Then, open it in your code editor and write your Streamlit app.

💡 Example code to get started (inside `streamlit_app.py`):

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Processing Playground")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_container_width=True)

    # Example: Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    st.image(gray, caption="Grayscale", use_container_width=True, channels="GRAY")
```

---

## ✅ 4. Run Your Streamlit App Locally

```bash
streamlit run streamlit_app.py
```

This will open the app in your browser at:  
[http://localhost:8501](http://localhost:8501)

---

## ✅ 5. OPTIONAL: Deploy to Streamlit Cloud

### Step-by-Step:

1. **Push your project to GitHub**  
   Create a repo and push your `streamlit_app.py` and `requirements.txt`:

```bash
pip freeze > requirements.txt
```

2. **Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)**  
   - Sign in with GitHub  
   - Click **“New app”**  
   - Choose your GitHub repo and branch  
   - Set file path to: `streamlit_app.py`  
   - Click **“Deploy”**

🎉 Your app will be live on a public Streamlit URL like:

```
https://your-app-name.streamlit.app
```

---

## ✅ You're Done!

Now you can interactively apply image processing techniques and share your app with others.