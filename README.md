## 🚀 Project: Developing an Object Instance Segmentation Model using YOLOv11s

## 🌟 Project Overview

This project focuses on building and training an advanced model for **Instance Segmentation** using the **YOLOv11s** architecture. This specific model was chosen to achieve the best balance between **speed, lightweightness, and accuracy** compared to other versions.

* The YOLOv11s model is distinguished by being **highly compact**, ensuring it runs smoothly even on devices with limited resources.
* The low **Inference Time** makes it ideal for **Real-Time Applications**.
* The training process was straightforward and did not require a lot of memory.

---

## ⚙️ Model Setup and Configuration

| Parameter | Value | Description | Source |
| :--- | :--- | :--- | :--- |
| **Task** | `segment` | Specifies the instance segmentation task | |
| **Model** | `yolo11s-seg.pt` | The selected model (YOLOv11s) | |
| **Epochs** | 150 | Maximum number of training epochs | |
| **Patience** | 20 | Number of epochs to wait for improvement before Early Stopping | |
| **Batch Size** | 16 | Number of images in each training step | |
| **imgsz** | 640 | Image resolution used for training and inference | |

---

## 🧪 Data Pipeline & Validation Strategy

The project relied on a comprehensive data processing pipeline to compensate for the **scarcity of the original dataset**.

### 1. 🖼️ Tiling and Conversion Pipeline

A pipeline was implemented to convert COCO data to YOLO format, focusing on the **Tiling** technique to:

* Effectively increase the number of training images without collecting new data.
* Enable the model to see smaller details more clearly.
* Use **SAHI Slicing** to divide COCO images and correctly adjust their **Annotations** into tiles of size **640×640**.
* Conversion to the YOLO format was used with the `use_segments=True` option to create `.txt` files and correctly save the **Segmentation Masks**.

### 2. 🎨 Data Augmentation

Since the number of images after tiling was small (about 300 images), data augmentation was used to increase variety and scene coverage:

* **Tool**: The **Roboflow** platform was relied upon for its ready-made and organized tools, which simplified the process.
* **Applied Transformations**: Rotation, resizing, brightness adjustment, application of **Flips**, and combining multiple transformations.

### 3. 📊 Data Splitting Strategy

* **Ratio**: The dataset was split with a ratio of **0.8 for Training and 0.2 for Validation**.
* **Test Set**: A separate test set was not allocated due to data scarcity, and the **Validation Set** was relied upon to evaluate the model's final performance.
* **Quality**: Care was taken to prevent **Data Leakage** to ensure reliable evaluation, as the validation images were never seen by the model.

---

## 📈 Key Results and Performance

Strong and reliable results were obtained, as the Loss Curves were smooth, and validation metrics remained close to training metrics, indicating reliable model training **without Overfitting**.

The following table summarizes the best Mask Segmentation performance on the Validation Set:

| Metric (Mask Segmentation) | Value (Best/Final) | Significance |
| :--- | :--- | :--- |
| **mAP@0.5** | 0.98533 (Final Epoch 117) | High average precision exceeding 98%. |
| **Precision** | 0.99605 (Final Epoch 117) | Near-perfect precision (approaching 1.00). |
| **Recall** | 0.96338 (Best Epoch 111) | High recall (around 95.8%) indicating detection of the majority of objects. |
| **mAP@0.5:0.95** | 0.90066 (Best Epoch 111) | Excellent average precision across a wide range of IoU thresholds. |

**Conclusion:** These results prove that **YOLOv11s** is a practical and trustworthy choice for this task.