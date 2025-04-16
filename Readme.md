# Food Quality Detection Using Deep Learning and Clustering

This project focuses on detecting fruit quality by combining image preprocessing, data augmentation, clustering (K-Means, Agglomerative, DBSCAN), and deep learning model training. The models were evaluated on clustered data to assess the impact of clustering techniques on classification performance.

---

## 📁 Dataset Preparation
The dataset contains fruit images labeled under the categories "Fresh" and "Rotten". The images were processed as follows:

### ✅ Data Preprocessing
- **Blurred** – Noise reduction using Gaussian blur.
- **Edge Detection** – Applied Canny edge detector.
- **Normalized** – Pixel values scaled between 0 and 1.
- **Original Image** – Base reference image.
- **Resized** – All images resized to **128 x 128**.
- **Grayscale** – Converted to grayscale for feature simplification.

### 🔄 Data Augmentation
- **Original Image**
- **Horizontal Flipping**
- **Rotation (±15°)**
- **Increased Brightness**
- **180° Rotation**

These augmentations helped to improve model generalization and prevent overfitting.

---

## 🔍 Clustering
Three clustering methods were used to label the data based on visual similarity:

### 1. K-Means Clustering
- **Clusters Chosen**: 5
- **Cluster Labels**:
  - `Cluster 0`: Fresh
  - `Cluster 1`: Slightly Aged
  - `Cluster 2`: Slate
  - `Cluster 3`: Spoiled
  - `Cluster 4`: Rotten

### 2. Agglomerative Clustering
- **Linkage Method**: Ward
- **Distance Metric**: Euclidean
- **Clusters Chosen**: 5 (Same as above)

### 3. DBSCAN
- Not used in final training due to inconsistent cluster sizes and undefined cluster count.

---

## 🧠 Model Training
Models were trained separately using data labeled with **K-Means** and **Agglomerative Clustering** results.

Each model was evaluated on four metrics:
- **Accuracy**
- **Precision (Macro Average)**
- **Recall (Macro Average)**
- **F1-Score (Macro Average)**

### 🏗️ Model Architectures Used
| Model | Architecture Summary |
|-------|----------------------|
| AlexNet | 5 Conv layers + 3 FC layers + ReLU + Dropout |
| DenseNet121 | Dense Blocks + Transition layers + Global Avg Pool + FC |
| InceptionV3 | Inception modules + Global Avg Pool + FC |
| MobileNetV2 | Depthwise separable convolutions + Bottleneck blocks |
| ResNet50 | Residual blocks with identity shortcut connections |
| VGG16 | 13 Conv layers + 3 FC layers |
| VGG19 | 16 Conv layers + 3 FC layers |
| Xception | Depthwise separable convs + linear stack of 36 conv layers |

---

## 📊 Results Comparison
### Using K-Means Clustered Data
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|-------|----------|------------------|----------------|-----------------|
| AlexNet | 0.72 | 0.71 | 0.72 | 0.70 |
| DenseNet121 | 0.75 | 0.73 | 0.74 | 0.74 |
| InceptionV3 | 0.74 | 0.74 | 0.72 | 0.72 |
| MobileNetV2 | 0.77 | 0.75 | 0.76 | 0.75 |
| ResNet50 | 0.37 | 0.20 | 0.32 | 0.23 |
| VGG16 | 0.66 | 0.65 | 0.63 | 0.64 |
| VGG19 | 0.68 | 0.66 | 0.65 | 0.65 |
| Xception | 0.72 | 0.70 | 0.70 | 0.70 |

### Using Agglomerative Clustered Data
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|-------|----------|------------------|----------------|-----------------|
| AlexNet | 0.84 | 0.82 | 0.86 | 0.83 |
| DenseNet121 | 0.83 | 0.81 | 0.83 | 0.82 |
| MobileNetV2 | 0.82 | 0.82 | 0.83 | 0.83 |
| Xception | 0.81 | 0.80 | 0.81 | 0.80 |
| InceptionV3 | 0.74 | 0.73 | 0.73 | 0.73 |
| VGG16 | 0.70 | 0.56 | 0.59 | 0.57 |
| VGG19 | 0.66 | 0.53 | 0.55 | 0.53 |
| ResNet50 | 0.42 | 0.34 | 0.34 | 0.34 |

---

## 🏆 Best Model
**MobileNetV2 with Agglomerative Clustering** achieved the highest balance of performance:
- **Accuracy**: 0.82
- **F1 Score**: 0.83
- Efficient and lightweight model ideal for deployment on resource-constrained environments.

---

## 👨‍💻 Contributors
1. Dogga Pavan Sekhar  
2. Saragadam Kundana Chinni  
3. Thalluru Lakshmi Prasanna  
4. Kesa Veera Venkata Yaswanth  

---

## 📌 Future Scope
- Integration of real-time quality detection using camera feed
- Extending to multi-fruit classification
- Real-time deployment on mobile devices using TensorFlow Lite or ONNX

