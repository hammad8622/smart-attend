## 📸 SmartAttend: Facial Recognition-Based Attendance System

SmartAttend is an intelligent, desktop-based attendance system using facial recognition. Designed for educational institutions, it enables teachers to manage classes, enroll students, gather facial image samples, train a classifier, and mark attendance automatically — all through a sleek and user-friendly GUI built with `customtkinter`.

### 🚀 Features

* 🧑‍🏫 **Teacher Authentication**: Sign-up and log in functionality using a MongoDB backend.
* 📚 **Class Management**: Create, view, and manage multiple classes.
* 🧑‍🎓 **Student Enrollment**: Add detailed student profiles to classes.
* 📷 **Image Collection**: Capture and augment face images through webcam for training.
* 🧠 **Model Training**: Automatically trains a Random Forest Classifier on captured and augmented images.
* ✅ **Attendance Marking**: Detect and identify students in real-time using a Caffe-based face detector and trained model.
* 📊 **Report Generation**: Export attendance records as CSV reports per class and date.
* 🗂️ **File Management**: Automatically manages folder structures and model file storage.

### 🧠 Technologies Used

* Python, OpenCV, NumPy, Scikit-learn, Albumentations
* MongoDB (via `pymongo`)
* Caffe DNN (for face detection)
* CustomTkinter for UI
* CSV and JSON for structured data storage

### 📁 Project Structure

* `main2.py`: Core application including GUI, data handling, model training, and attendance.
* `initialize_folder_structure_for_testing.py`: Utility to reset the class folder structure for testing.
* MongoDB collections: `teacher`, `class`, `attendance`, `usedIDs`.

### 📦 Installation

```bash
pip install -r requirements.txt
```

> Ensure MongoDB is installed and running on `localhost:27017`.

### 🛠️ Running the App

```bash
python main2.py
```

### 📌 Notes

* Requires webcam access.
* A pre-trained Caffe model (`architecture.prototxt` and `weights.caffemodel`) should be placed in the `SmartAttend/detection_model` directory under the user’s home path.
* Designed for Windows/Linux desktops only.

