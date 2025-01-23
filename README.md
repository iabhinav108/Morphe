# **Deepfake Video Detection using Multi-Branch Deep Learning Model**

## **Overview**  
This project focuses on developing an AI/ML-based solution for detecting face-swap-based deepfake videos. It employs a supervised deep learning model trained on labeled datasets and follows a multi-domain approach to extract crucial features from facial images.  

## **Proposed Solution**  
The model comprises three parallel branches that process different aspects of the input data:

1. **Color Branch** - Extracts features from RGB facial images.  
2. **Edge Branch** - Processes edge-detected images using the **Canny Edge Detector**.  
3. **Texture Branch** - Captures fine texture details from color images using the **Laplacian operator**.  

Each branch utilizes a **customized DenseNet121**, where the last three dense and transition blocks are removed to reduce model size. **Coordinate attention** is applied to extracted features, improving attention mechanisms. The final classification is achieved by merging 1D feature vectors (length 16) from all three branches.  

## **Tech Stack**  
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Metrics Library:** TorchMetrics  
- **Computer Vision Libraries:** OpenCV, Pillow  
- **Hardware:** NVIDIA CUDA-enabled GPUs  
- **Numerical Computing:** NumPy  
- **Visualization:** Matplotlib  

## **Methodology**  

### **1. Data Preparation**  
- **Datasets Used:**  
  - **Deepfake Detection Challenge (DFDC)**  
  - **FaceForensics++**  
  - **CelebDF**  
- **Preprocessing:**  
  - Resize images to **256×256**  
  - Apply **random flips**  
  - Convert images to **tensors**  
  - Generate **edge** and **texture** images  
  - **Due to computational constraints, we randomly selected 64,000 images** (32,000 real and 32,000 deepfake) for training and evaluation.  
  - Dataset split: **80% training, 10% validation, 10% testing**  

### **2. Model Design**  
- **Multi-Branch Network:**  
  - **Color Branch:** Extracts RGB image features.  
  - **Edge Branch:** Uses a **Canny edge detector** for contour-based feature extraction.  
  - **Texture Branch:** Uses **Laplacian filtering** to enhance texture information.  
- **Modified DenseNet121:**  
  - Removed last **three dense and transition blocks** to optimize model size.  
  - Applied **coordinate attention** to enhance feature learning.  
- **Final Classification:**  
  - Feature vectors from each branch are concatenated and passed through a fully connected layer for **binary classification** (Real vs. Deepfake).  

### **3. Video Processing Strategy**  
Instead of processing the entire video, our application follows an optimized approach:
1. The first frame of the video where a **human-like face** is detected is extracted.
2. This extracted frame is then passed to the deep learning model for classification.
3. The output (Real or Deepfake) is generated based on this **single-frame analysis**.  

This method significantly **reduces computational cost** while maintaining detection accuracy.  

## **Performance & Observations**  
- **Accuracy:** **93%**  
- **Computational Cost:** Low (optimized DenseNet121 reduces model size).  
- **User-Friendly:** Can be integrated into real-world applications.  

### **Challenges & Mitigation Strategies**  
| **Challenge** | **Mitigation Strategy** |  
|--------------|------------------------|  
| **Extracting face frames from videos** | Ensure videos are pre-processed into face images before classification. |  
| **Handling multiple faces in a video** | Crop and analyze each detected face separately. |  
| **Lack of audio manipulation detection** | Focus is on visual deepfakes; audio deepfake detection is outside scope. |  

## **How to Use**  
### **1. Installation**  
```bash
git clone https://github.com/iabhinav108/Morphe.git
cd Morphe
pip install -r backend/requirements.txt
```

### **2. Running Inference**  
```python
cd backend
python app.py
```


## **Future Improvements**  
- Incorporating temporal analysis for video-based deepfake detection.  
- Exploring **GAN-based adversarial training** for robustness.  
- Enhancing performance on low-quality or compressed videos.  
  


