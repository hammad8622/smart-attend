from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from abc import ABC
from tkinter import messagebox
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
import customtkinter as ctk
import albumentations as A
import numpy as np
import random
import shutil
import cv2
import os
import uuid
import json
import pickle
import csv



class StateInfo:

    def __init__(self):
        self._username = None
        self._password = None
        self._user_ID = None
        self._classes_IDs = None
        self._students_classes_dicts = None
        self._base_path = None
        self._classes_path = None
        self._detection_model_path = None
        self._current_class_ID = None
        self._current_student_ID = None

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        self._password = value

    @property
    def user_ID(self):
        return self._user_ID

    @user_ID.setter
    def user_ID(self, value):
        self._user_ID = value

    @property
    def classes_IDs(self):
        return self._classes_IDs

    @classes_IDs.setter
    def classes_IDs(self, value):
        self._classes_IDs = value

    @property
    def students_classes_dicts(self):
        return self._students_classes_dicts

    @students_classes_dicts.setter
    def students_classes_dicts(self, value):
        self._students_classes_dicts = value

    @property
    def current_class_ID(self):
        return self._current_class_ID

    @current_class_ID.setter
    def current_class_ID(self, value):
        self._current_class_ID = value

    @property
    def current_student_ID(self):
        return self._current_student_ID

    @current_student_ID.setter
    def current_student_ID(self, value):
        self._current_student_ID = value

    @property
    def base_path(self):
        return self._base_path
    
    @base_path.setter
    def base_path(self, value):
        self._base_path = value

    @property
    def classes_path(self):
        return self._classes_path
    
    @classes_path.setter
    def classes_path(self, value):
        self._classes_path = value

    @property
    def detection_model_path(self):
        return self._detection_model_path
    
    @detection_model_path.setter
    def detection_model_path(self, value):
        self._detection_model_path = value
    

    def print_attributes(self):
        print(f"Username: {self.username}")
        print(f"Password: {self.password}")
        print(f"User ID: {self.user_ID}")
        print(f"Classes IDs: {self.classes_IDs}")
        print(f"Base Path: {self.base_path}")
        print(f'Classes Path: {self.classes_path}')
        print(f"Detection Model Path: {self.detection_model_path}")
        print(f"Students Classes Dicts: {self.students_classes_dicts}")
        print(f"Current Class ID: {self.current_class_ID}")
        print(f"Current Student ID: {self.current_student_ID}")


def initialize_database():
    client = MongoClient('localhost', 27017)
    DB = client['smartAttend']

    # Initialize the used IDs collection structure
    used_IDs_collection = DB['usedIDs']

    if used_IDs_collection.count_documents({}) > 0:
        return DB, client

    entry = {
        'teacherIDs': [],
        'classIDs': [],
        'studentIDs': []
    }
    
    used_IDs_collection.insert_one(entry)

    return DB, client


def initialize_folder_structures():
    global state
    base_path = Path.home()
    project_path = base_path / 'SmartAttend'

    if not project_path.exists():
        project_path.mkdir(exist_ok=True, parents=True)

    classes_path = project_path / 'classes'
    detection_model_path = project_path / 'detection_model'

    if not classes_path.exists():
        classes_path.mkdir(exist_ok=True, parents=True)

    state.base_path = project_path
    state.classes_path = classes_path
    state.detection_model_path = detection_model_path

    if not detection_model_path.exists():
        detection_model_path.mkdir(exist_ok=True, parents=True)
        
        src_architecture = Path(__file__).parent.parent / 'detection_model' / 'architecture.prototxt'
        src_weights = Path(__file__).parent.parent / 'detection_model' / 'weights.caffemodel'
        
        dst_architecture = detection_model_path / 'architecture.prototxt'
        dst_weights = detection_model_path / 'weights.caffemodel'
        
        # Debug: Print the absolute paths to ensure correctness
        print(f"Source Architecture Path: {src_architecture.resolve()}")
        print(f"Destination Architecture Path: {dst_architecture.resolve()}")
        print(f"Source Weights Path: {src_weights.resolve()}")
        print(f"Destination Weights Path: {dst_weights.resolve()}")
        
        # Check if source files exist
        if not src_architecture.exists():
            print(f"Source file {src_architecture} does not exist.")
        if not src_weights.exists():
            print(f"Source file {src_weights} does not exist.")
        
        # Copy the files
        print('got here')
        shutil.copy(src_architecture, dst_architecture)
        shutil.copy(src_weights, dst_weights)

    print(f"Folders initialized at {project_path}")
    

def generate_teacher_ID():

    global DB
    used_ID_collection = DB['usedIDs']
    used_teacher_ID_document = used_ID_collection.find_one({}, {'_id': False, 'teacherIDs': True})
    
    # Handle the case when no document is found
    if not used_teacher_ID_document:
        used_teacher_ID_list = []
    else:
        used_teacher_ID_list = used_teacher_ID_document.get('teacherIDs', [])
    
    ID = None

    while True:
        ID = f'T{random.randint(100, 999)}'  

        if ID in used_teacher_ID_list:
            continue

        break

    used_teacher_ID_list.append(ID)
    used_ID_collection.update_one(
        {}, 
        {'$set': {'teacherIDs': used_teacher_ID_list}}, 
        upsert=True
    )

    return ID


def generate_class_ID():

    global DB
    used_ID_collection = DB['usedIDs']
    used_class_ID_document = used_ID_collection.find_one({}, {'_id': False, 'classIDs': True})

    if not used_class_ID_document:
        used_class_ID_list = []
    else:
        used_class_ID_list = used_class_ID_document.get('classIDs', [])
    
    ID = None

    while True:
        ID = f'C{random.randint(100, 999)}'
        if ID in used_class_ID_list:
            continue
        break

    used_class_ID_list.append(ID)
    used_ID_collection.update_one(
        {}, 
        {'$set': {'classIDs': used_class_ID_list}}, 
        upsert=True
    )

    return ID


def generate_student_ID():

    global DB
    used_ID_collection = DB['usedIDs']
    used_student_ID_document = used_ID_collection.find_one({}, {'_id': False, 'studentIDs': True})

    if not used_student_ID_document:
        used_student_ID_list = []
    else:
        used_student_ID_list = used_student_ID_document.get('studentIDs', [])
    
    ID = None

    while True:
        ID = f'S{random.randint(100, 999)}'
        if ID in used_student_ID_list:
            continue
        break

    used_student_ID_list.append(ID)
    used_ID_collection.update_one(
        {}, 
        {'$set': {'studentIDs': used_student_ID_list}}, 
        upsert=True
    )

    return ID

def add_student(class_ID, student_name, student_age, student_gender, batch, major_name):

    # the current_directory_path is the path to the specific class 
    # we should already know where the classes folder is present

    global db
    class_collection = db['class']
    entry = {'studentID' : generate_student_ID(), 
             'name': student_name,
             'age': student_age,
             'gender': student_gender,
             'batch': batch,
             'major': major_name}

    class_collection.update_one({'classID': class_ID}, {'$push': {'students': entry}})


def add_class(class_name):

    global DB, state

    class_collection = DB['class']
    teacher_ID = state.user_ID

    entry = {
        'classID': generate_class_ID(),
        'className': class_name,
        'teacherID': teacher_ID,
        'students': []
    }

    class_collection.insert_one(entry)
    new_class_path = Path(state.classes_path / class_name)
    attendance_dir_path = Path(new_class_path / 'attendance')
    new_class_path.mkdir(exist_ok=False)
    attendance_dir_path.mkdir(exist_ok=True)


def add_teacher(username, password):

    global DB
    teacher_collection = DB['teacher']


    entry = {
        'teacherID': generate_teacher_ID(),
        'userName': username,
        'passWord': password,
    }

    teacher_collection.insert_one(entry)


def update_state(username, password):

    global state, DB

    state._username = username
    state._password = password

    teacher_collection = DB['teacher']
    teacher_document = teacher_collection.find_one({'userName': state._username, 'passWord': state._password})
    state._user_ID = teacher_document.get('teacherID')

    classes_collection = DB['class']

    if classes_collection is None:
        return 

    class_documents = classes_collection.find({'teacherID': state._user_ID})
    classes_IDs_list = []
    class_student_ID_dict = {}

    for document in class_documents:

        classes_IDs_list.append(document.get('classID'))
        students_list = document.get('students')
        
        for student in students_list:
            class_student_ID_dict[student.get('studentID')] = document.get('classID')

    state._students_classes_dicts = class_student_ID_dict
    state._classes_IDs = classes_IDs_list

def students_in_class_exist(class_ID):

    global DB
    classes_collection = DB['class']
    class_document = classes_collection.find_one({'classID': class_ID})
    students_list = class_document.get('students')

    if students_list is not None:
        return True
    
    return False


def load_caffe_model(model_architecture_path, model_weights_path):

    return cv2.dnn.readNetFromCaffe(model_architecture_path, model_weights_path)


def caffe_detect_faces(bgr_frame, model):

    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    
    formatted_list = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Add a confidence threshold based on your requirement
            box = detections[0, 0, i, 3:7]
            (x1, y1, x2, y2) = box.tolist()
            formatted_list.append({'conf': confidence, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return formatted_list


def gathering_samples(class_path):
    model = load_caffe_model(
        state.detection_model_path / 'architecture.prototxt',
        state.detection_model_path / 'weights.caffemodel'
    )
    SAMPLE_AMOUNT = 300

    students_list = os.listdir(class_path)
    students_list.remove('attendance')
    current_idx = 0
    save_path = os.path.join(class_path, students_list[current_idx])
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()  # Frame in BGR format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for saved images
        height, width, _ = frame.shape

        key = cv2.waitKey(10) & 0xFF

        detections = caffe_detect_faces(frame, model)

        if detections:
            for detection in detections:
                x1, y1, x2, y2 = int(detection['x1'] * width), int(detection['y1'] * height), int(detection['x2'] * width), int(detection['y2'] * height)
                # Draw bounding boxes on the live feed (frame) only in BGR
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the number of images captured and current folder name
        image_count = len(os.listdir(save_path))
        folder_name = students_list[current_idx]
        cv2.putText(frame, f'Images: {image_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Folder: {folder_name}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Capture', frame)  # Display the live feed with bounding boxes and text

        if key == ord('c'):
            for detection in detections:
                x1, y1, x2, y2 = int(detection['x1'] * width), int(detection['y1'] * height), int(detection['x2'] * width), int(detection['y2'] * height)
                cropped_face = frame_rgb[y1:y2, x1:x2]  # Extract face region from RGB image
                file_path = os.path.join(save_path, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(file_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))  # Save in BGR format
                print(f'Image saved on path {file_path}')

        elif key == ord('q'):
            break

        elif key == ord('a'):
            if current_idx > 0:
                current_idx -= 1
                save_path = os.path.join(class_path, students_list[current_idx])

        elif key == ord('d'):
            no_of_students = len(students_list)
            if current_idx < no_of_students - 1:
                current_idx += 1
                save_path = os.path.join(class_path, students_list[current_idx])

    # Define the augmentation pipeline
    augmentations = A.Compose([
        A.Rotate(limit=10, p=0.7),
        A.HorizontalFlip(p=0.7),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(p=0.3)
    ])

    for student_directory in os.scandir(class_path):
        save_path = os.path.abspath(student_directory)
        if len(os.listdir(save_path)) == 0 or len(os.listdir(save_path)) == 100:
            continue

        while True:
            # Update count at the beginning of each loop
            count = len([entry for entry in os.scandir(save_path) if entry.is_file() and entry.name.endswith(('.jpeg', '.jpg'))])

            if count >= SAMPLE_AMOUNT:
                break

            for entry in os.scandir(save_path):
                if entry.is_file() and entry.name.endswith(('.jpeg', '.jpg')) and count < SAMPLE_AMOUNT:
                    image = cv2.imread(entry.path)  # Load the image
                    if image is not None:
                        augmented = augmentations(image=image)
                        augmented_image = augmented['image']
                        save_path_augmented = os.path.join(save_path, f'{uuid.uuid4()}.jpg')
                        cv2.imwrite(save_path_augmented, augmented_image)
                        print(f'Augmented image saved on path {save_path_augmented}')
                        count += 1

                if count >= SAMPLE_AMOUNT:
                    break

    cap.release()
    cv2.destroyAllWindows()

def shuffle_directory_contents(directory_path):
    # Get a list of all files in the directory
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]

    # Shuffle the list of files
    random.shuffle(files)

    # Rename the files to a temporary name
    temp_names = [f"temp_{i}" for i in range(len(files))]
    for old_name, temp_name in zip(files, temp_names):
        os.rename(os.path.join(directory_path, old_name), os.path.join(directory_path, temp_name))

    # Shuffle the temp names back to original names
    random.shuffle(temp_names)
    for temp_name, old_name in zip(temp_names, files):
        os.rename(os.path.join(directory_path, temp_name), os.path.join(directory_path, old_name))


def create_dataset(class_path):
    label_dict = {}
    labels_path = os.path.join(class_path, 'labels.json')
    images_path = os.path.join(class_path, 'images')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for student_directory in os.scandir(class_path):

        if student_directory.name == 'images' or student_directory.name == 'attendance':
            continue 

        if student_directory.is_dir():
            label = student_directory.name
            
            for student_image in os.scandir(student_directory.path):
                if student_image.is_file():
                    # Strip the extension
                    key = os.path.splitext(student_image.name)[0]  # Removes the extension
                    label_dict[key] = label
                    # Rename and move the file without changing the extension
                    new_image_path = os.path.join(images_path, student_image.name)
                    os.rename(student_image.path, new_image_path)

    # shuffle_directory_contents(images_path)

    with open(labels_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=4)

    print("dataset successfully created")


def train_classification_model(class_path):

    image_directory = os.path.join(class_path, 'images')
    label_file = os.path.join(class_path, 'labels.json')
    
    base_model_path = os.path.join(class_path, 'models')
    rfc_path = os.path.join(base_model_path, 'random_forest_classifier.pkl')
    # knn_path = os.path.join(base_model_path, 'knn.pkl')
    # svc_path = os.path.join(base_model_path, 'svc.pkl')
    encoder_path = os.path.join(base_model_path, 'label_encoder.pkl')
    
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)

    if not os.path.exists(base_model_path):
        os.mkdir(base_model_path)

    images = []
    labels = []
    
    # Load labels dictionary
    with open(label_file, 'r') as json_file:
        labels_dict = json.load(json_file)

    # Process each image
    for entry in os.scandir(image_directory):
        if entry.is_file():
            entry_without_extension = entry.name.split('.')[0]
            label = labels_dict.get(entry_without_extension)
            if label is not None:
                img = cv2.imread(entry.path)
                if img is not None:
                    # Preprocess the image
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_resized = cv2.resize(img_rgb, (128, 128))
                    img_flattened = img_resized.flatten() / 255.0
                    images.append(img_flattened)
                    labels.append(label)

    # Check if images are loaded correctly
    if not images:
        print("No images found or processed.")
        return

    X = np.array(images)
    y = np.array(labels)


    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f'y (encoded): {y_encoded}')

    # Training multiple models
    rfc = RandomForestClassifier(n_estimators=200, max_depth=7)


    rfc.fit(X_train, y_train)
    print('RFC Trained')

    with open(rfc_path, 'wb') as rfc_model_file:
        pickle.dump(rfc, rfc_model_file)

    with open(encoder_path, 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)


    print(f"Models trained and saved to {base_model_path}")

    # rfc_cv_score = cross_val_score(rfc, X, y, cv=3, scoring='accuracy')
    # rfc_y_pred = rfc.predict(X_test)
    # rfc_score = accuracy_score(y_test, rfc_y_pred)

    # print(f'RFC:\CV: {rfc_cv_score}\nTest Accuracy: {rfc_score}')



def mark_attendance(class_name):

    global DB

    attendance_image_path = ctk.filedialog.askopenfilename(title='Enter the input image', filetypes=[('JPG Files', '*.jpg')])
    detection_model = load_caffe_model(r'C:\Users\laptop zone\SmartAttend\detection_model\architecture.prototxt', 
                             r'C:\Users\laptop zone\SmartAttend\detection_model\weights.caffemodel')
    classification_model_path = os.path.join(class_name, 'models', 'random_forest_classifier.pkl')
    encoder_path = os.path.join(class_name, 'models', 'label_encoder.pkl')
    classification_model = None
    encoder = None
    THRESHOLD = 0.3
    detected_students = []
    now = datetime.now()

    with open(classification_model_path, 'rb') as f:
        classification_model = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    if not attendance_image_path:
        messagebox.showerror('Error', 'The input image was not able to be found')
        return
    
    frame = cv2.imread(filename=attendance_image_path)
    height, width, _ = frame.shape
    detections = caffe_detect_faces(frame, detection_model)
    output_frame = frame.copy()

    if detections:
        for detection in detections:
            x1, y1, x2, y2 = int(detection['x1'] * width), int(detection['y1'] * height), int(detection['x2'] * width), int(detection['y2'] * height)
            
            if x1 < x2 and y1 < y2:  # Ensure the coordinates are valid
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:  # Check if crop is not empty
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop = cv2.resize(crop, (128, 128))
                    crop = crop.flatten() / 255.0
                    input_data = np.array(crop).reshape(1, -1)  # Reshape to 2D array
                    prediction_probabilities = classification_model.predict_proba(input_data)
                    confidence_score = np.max(prediction_probabilities)
                    most_confident_class = np.argmax(prediction_probabilities)
                    prediction = encoder.inverse_transform([most_confident_class])
                    text = str(prediction[0]) if confidence_score >= THRESHOLD else 'Unknown'
                    
                    if text != 'Unknown':
                        if text not in detected_students:
                            detected_students.append(text)
                    
                    height, width, _ = output_frame.shape
                    while width >= 2000 or height >= 1000:
                        output_frame = cv2.resize(output_frame, ((height // 2), (width // 2)))
                        height, width, _ = output_frame.shape  # Update dimensions after resizing
                    
                    X1, Y1, X2, Y2 = int(detection['x1'] * width), int(detection['y1'] * height), int(detection['x2'] * width), int(detection['y2'] * height)
                    output_frame = cv2.rectangle(output_frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                    position = (X1, Y1 - 10)  # Placing text above the rectangle
                    # Add the text above the rectangle
                    cv2.putText(output_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    attendance_collection = DB['attendance']

    print(detected_students)

    if detected_students:

        entry = {
            'classID': state.current_class_ID,
            'className': os.path.basename(class_name),
            'date': f'{now.day}-{now.month}-{now.year}',
            'presentStudents': detected_students
        }
        
        attendance_collection.insert_one(entry)

    
    # if height >= 1000 or width >= 1000:
    #     output_frame = cv2.resize(output_frame, ((height // 2), (width // 2)))
    # elif height >= 2000 or width >= 2000:
    #     output_frame = cv2.resize(output_frame, ((height // 2), (width // 2)))
    print(output_frame.shape)
    cv2.imshow('Frame', output_frame)  # Show the frame regardless of detections
    


def generate_reports(class_path):
    global DB, state
    attendance_collection = DB['attendance']
    class_collection = DB['class']

    class_document = class_collection.find_one({'classID': state.current_class_ID})
    students_list = class_document.get('students')

    cursor = attendance_collection.find({'classID': state.current_class_ID})
    class_name = os.path.basename(class_path)
    attendance_dir_path = Path(class_path) / 'attendance'
    now = datetime.now()
    date_str = f'{now.day}_{now.month}_{now.year}'
    file_path = Path(attendance_dir_path) / f'{state.current_class_ID}_{class_name}_{date_str}.csv'

    if not attendance_dir_path.exists():
        attendance_dir_path.mkdir(parents=True)

    attendance = {}

    # Create a dictionary to map student names to IDs
    name_ID_dict = {student_doc.get('name'): student_doc.get('studentID') for student_doc in students_list}

    # Debug: Print the name_ID_dict
    print("name_ID_dict:", name_ID_dict)

    # Iterate over attendance documents
    for document in cursor:
        date = document.get('date')
        present_students = document.get('presentStudents')

        # Initialize list for the date if not already present
        if date not in attendance:
            attendance[date] = []

        # Add student names and IDs to the attendance dictionary
        for student_name in present_students:
            # Debug: Check if the student_name is in name_ID_dict
            if student_name not in name_ID_dict:
                print(f"Warning: student_name {student_name} not found in name_ID_dict")
            student_ID = name_ID_dict.get(student_name, 'Unknown')
            attendance[date].append({'studentID': student_ID, 'studentName': student_name})

    # Prepare data for CSV
    data = []
    for date, students in attendance.items():
        for student in students:
            data.append([student['studentID'], student['studentName'], date])

    # Write data to CSV file
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Student_ID', 'Student_Name', 'Date_Present'])
        writer.writerows(data)

    print(f'Report generated and saved at {file_path}')


def check_for_samples(class_path):
    list_ = os.listdir(class_path)
    list_.remove('attendance')
    path_list = [os.path.join(class_path, element) for element in list_]

    # Check if each directory in path_list is empty
    for path in path_list:
        if not os.listdir(path):
            return False

    return True


class ClassButton(ctk.CTkFrame):

    def __init__(self, master, class_ID, class_name, student_amount, command=None, **kwargs):
        
        super().__init__(master, corner_radius=5, fg_color='#4B4747', **kwargs)
        button_font = ctk.CTkFont(family='Arial', size=14)
        button_font_2 = ctk.CTkFont(family='Arial', size=14, weight='bold')
        
        self.command = command

        # Configure the grid layout for the button frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.class_ID = class_ID

        # Create labels and place them in the grid
        self.ID_label = ctk.CTkLabel(master=self, text="Class ID", text_color="white", font=button_font_2)
        self.ID_label.grid(row=0, column=0, sticky='w', padx=10)

        self.name_label = ctk.CTkLabel(master=self, text="Class Name", text_color="white", font=button_font_2)
        self.name_label.grid(row=0, column=1, sticky='ew')

        self.student_amount_label = ctk.CTkLabel(master=self, text="Number of Students", text_color="white", font=button_font_2)
        self.student_amount_label.grid(row=0, column=2, sticky='e', padx=10)

        self.ID_label_input = ctk.CTkLabel(master=self, text=class_ID, text_color="white", font=button_font)
        self.ID_label_input.grid(row=1, column=0, sticky='w', padx=10)

        self.name_label_input = ctk.CTkLabel(master=self, text=class_name, text_color="white", font=button_font)
        self.name_label_input.grid(row=1, column=1, sticky='ew')

        self.student_amount_label_input = ctk.CTkLabel(master=self, text=student_amount, text_color="white", font=button_font)
        self.student_amount_label_input.grid(row=1, column=2, sticky='e', padx=10)

        # Bind a click event to the frame and labels
        self.bind('<Button-1>', self.on_click)
        self.ID_label.bind('<Button-1>', self.on_click)
        self.name_label.bind('<Button-1>', self.on_click)
        self.student_amount_label.bind('<Button-1>', self.on_click)
        self.ID_label_input.bind('<Button-1>', self.on_click)
        self.name_label_input.bind('<Button-1>', self.on_click)
        self.student_amount_label_input.bind('<Button-1>', self.on_click)

        # Pack the button frame into the parent window
        self.pack(fill='x', expand=True, pady=20, padx=20)

    def on_click(self, event):
        global state
        if self.command:
            state.current_class_ID = self.class_ID
            self.command()


class StudentButton(ctk.CTkFrame):

    def __init__(self, master, student_ID, student_name, major, command=None, **kwargs):
        
        super().__init__(master, corner_radius=5, fg_color='#4B4747', **kwargs)
        button_font = ctk.CTkFont(family='Arial', size=14)
        button_font_2 = ctk.CTkFont(family='Arial', size=14, weight='bold')

        self.command = command
        self.student_ID = student_ID

        # Configure the grid layout for the button frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Create labels and place them in the grid
        self.ID_label = ctk.CTkLabel(master=self, text="Student ID", text_color="white", font=button_font_2)
        self.ID_label.grid(row=0, column=0, sticky='w', padx=10)

        self.name_label = ctk.CTkLabel(master=self, text="Student Name", text_color="white", font=button_font_2)
        self.name_label.grid(row=0, column=1, sticky='ew')

        self.major_label = ctk.CTkLabel(master=self, text="Major", text_color="white", font=button_font_2)
        self.major_label.grid(row=0, column=2, sticky='e', padx=10)

        self.ID_label_input = ctk.CTkLabel(master=self, text=student_ID, text_color="white", font=button_font)
        self.ID_label_input.grid(row=1, column=0, sticky='w', padx=10)

        self.name_label_input = ctk.CTkLabel(master=self, text=student_name, text_color="white", font=button_font)
        self.name_label_input.grid(row=1, column=1, sticky='ew')

        self.major_label_input = ctk.CTkLabel(master=self, text=major, text_color="white", font=button_font)
        self.major_label_input.grid(row=1, column=2, sticky='e', padx=10)

        # Bind a click event to the frame and labels
        self.bind('<Button-1>', self.on_click)
        self.ID_label.bind('<Button-1>', self.on_click)
        self.name_label.bind('<Button-1>', self.on_click)
        self.major_label.bind('<Button-1>', self.on_click)
        self.ID_label_input.bind('<Button-1>', self.on_click)
        self.name_label_input.bind('<Button-1>', self.on_click)
        self.major_label_input.bind('<Button-1>', self.on_click)

        # Pack the button frame into the parent window
        self.pack(fill='x', expand=True, pady=20, padx=20)

    def on_click(self, event):
        global state
        if self.command:
            state.current_student_ID = self.student_ID
            self.command()


class BaseScreen(ABC):

    ctk.set_appearance_mode('dark')
    ctk.set_default_color_theme('green')

    def fix_window(self, width=480, height=320):
        
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()

        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))

        self.geometry(f'{width}x{height}+{x}+{y}')
        self.resizable(False, False)


class AuthenticationScreen(BaseScreen, ctk.CTk):
    def __init__(self, **kwargs):
        super(AuthenticationScreen, self).__init__(**kwargs)
        BaseScreen.fix_window(self)

        self.title('Authentication')
        initialize_database()
        initialize_folder_structures()
        self.setup_ui()

    def setup_ui(self):
        self.label_font = ctk.CTkFont(family='Arial', size=16)
        self.button_font = ctk.CTkFont(family='Segoe UI', size=14)
        
        self.title_label = ctk.CTkLabel(master=self, text='SmartAttend', font=ctk.CTkFont(family='Arial', size=24, weight='bold'))
        self.title_label.pack(pady=16)

        self.username_label = ctk.CTkLabel(master=self, text='Username', font=self.label_font)
        self.username_label.pack(pady=8)

        self.username_entry = ctk.CTkEntry(master=self, width=200)
        self.username_entry.pack()

        self.password_label = ctk.CTkLabel(master=self, text='Password', font=self.label_font)
        self.password_label.pack(pady=8)

        self.password_entry = ctk.CTkEntry(master=self, width=200, show="*")
        self.password_entry.pack()

        self.signup_button = ctk.CTkButton(master=self, text='Sign-Up', font=self.button_font, width=100, command=self.signup)
        self.signup_button.place(x=130, y=235)

        self.login_button = ctk.CTkButton(master=self, text='Login', font=self.button_font, width=100, command=self.login)
        self.login_button.place(x=250, y=235)

        self.mainloop()

    def login(self):
        input_username = self.username_entry.get()
        input_password = self.password_entry.get()

        if not all([input_username, input_password]):
            messagebox.showerror('Error', 'One or more of the input fields are empty!')
            return

        teacher_collection = DB['teacher']
        teacher_document = teacher_collection.find_one({'userName': input_username, 'passWord': input_password})

        if not teacher_document:
            messagebox.showerror('Error!', 'Invalid username or password')
            return
        
        update_state(input_username, input_password)
        self.open_main_application()

    def signup(self):
        global DB, state

        input_username = self.username_entry.get()
        input_password = self.password_entry.get()

        if not all([input_username, input_password]):
            messagebox.showerror('Error', 'One or more of the input fields are empty!')
            return

        teacher_collection = DB['teacher']
        teacher_document = teacher_collection.find_one({'userName': input_username, 'passWord': input_password})

        if teacher_document:
            messagebox.showerror('Error!', f'Input username: {input_username}, already exists!')
            return
        
        add_teacher(input_username, input_password)
        update_state(input_username, input_password)

        self.open_main_application()

    def open_main_application(self):
        self.withdraw()
        self.classes_screen = ClassesScreen(master=self)
        self.classes_screen.protocol('WM_DELETE_WINDOW', self.close_screen)

    def close_screen(self):
        self.classes_screen.destroy()
        self.destroy()


class ClassesScreen(BaseScreen, ctk.CTkToplevel):
    def __init__(self, master, **kwargs):
        super(ClassesScreen, self).__init__(master, **kwargs)
        BaseScreen.fix_window(self)
        self.title('Classes')

        self.setup_ui()

    def setup_ui(self):

        label_font = ctk.CTkFont(family='Arial', size=14, weight='bold')
        button_font = ctk.CTkFont(family='Arial', size=12)
        
        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=480)
        self.scrollable_frame.pack(fill='both', expand=True)

        # Create the top frame
        self.top_frame = ctk.CTkFrame(master=self.scrollable_frame, width=480, height=60)
        self.top_frame.pack()

        # Create the bottom frame
        self.bottom_frame = ctk.CTkFrame(master=self.scrollable_frame, width=480)
        self.bottom_frame.pack(fill=ctk.BOTH, expand=True)

        # Add some labels
        self.username_label = ctk.CTkLabel(master=self.top_frame, text=f'Username: {state.username}', font=label_font)
        self.username_label.place(x=10, y=5)

        self.user_ID_label = ctk.CTkLabel(master=self.top_frame, text=f'User ID: {state.user_ID}', font=label_font)
        self.user_ID_label.place(x=10, y=25)

        self.add_class_option_button = ctk.StringVar(value='Options')
        self.option_menu = ctk.CTkOptionMenu(self.top_frame, values=['Add Class'], variable=self.add_class_option_button, 
                                            command=lambda x: self.add_class_option(), width=40, font=button_font)
        self.option_menu.place(x=365, y=0)

        # Load and display the classes
        self.load_classes()

    def load_classes(self):

        update_state(state.username, state.password)
        state.print_attributes()
        # Clear the current list of class buttons
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

        if isinstance(state.classes_IDs, list) and state.classes_IDs:
            # Add some buttons
            classes_collection = DB['class']
            for class_ID in state.classes_IDs:
                class_document = classes_collection.find_one({'classID': class_ID})
                if class_document:
                    class_name = class_document.get('className')
                    students_list = class_document.get('students')
                    number_of_students = len(students_list) if students_list else 0

                    class_button = ClassButton(master=self.bottom_frame, class_ID=class_ID, class_name=class_name, student_amount=number_of_students, command=self.view_class)
                    class_button.pack(pady=5)
        else:
            self.no_classes_label = ctk.CTkLabel(master=self.bottom_frame, text='No Classes Found!', font=ctk.CTkFont(family='Arial', size=20))
            self.no_classes_label.place(x=150, y=90)

    def add_class_option(self):
        self.dialog = ctk.CTkInputDialog(text='Enter a name for the class', title='Add Class', font=ctk.CTkFont(family='Arial', size=16))
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = 280
        height = 160
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        self.input_class_name = self.dialog.get_input()
        classes_collection = DB['class']
        class_document = classes_collection.find_one({'className': self.input_class_name})

        if class_document is not None:
            messagebox.showerror('Error!', f'A class called {self.input_class_name} already exists!')
            return

        if self.input_class_name is None:
            return

        if self.input_class_name in state.classes_IDs:
            messagebox.showerror('Error', 'The input class name already exists!')
            return

        if self.input_class_name.isspace() or not self.input_class_name:
            messagebox.showerror('Error', 'The input class name is invalid!')
            return

        add_class(self.input_class_name)
        # Refresh the class list after adding a new class
        self.load_classes()

    def view_class(self):
        self.withdraw()
        self.students_screen = StudentsScreen(master=self)
        self.students_screen.protocol('WM_DELETE_WINDOW', self.on_close)

    def on_close(self):
        self.students_screen.destroy()
        self.load_classes()
        self.deiconify()
        
        
class StudentsScreen(BaseScreen, ctk.CTkToplevel):
    def __init__(self, master, **kwargs):
        super(StudentsScreen, self).__init__(master, **kwargs)
        BaseScreen.fix_window(self)
        self.title('Students')
        self.setup_ui()

    def setup_ui(self):

        label_font = ctk.CTkFont(family='Arial', size=14, weight='bold')
        button_font = ctk.CTkFont(family='Arial', size=12)

        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=480)
        self.scrollable_frame.pack(fill='both', expand=True)

        # Create the top frame
        self.top_frame = ctk.CTkFrame(master=self.scrollable_frame, width=480, height=60)
        self.top_frame.pack()

        # Create the bottom frame
        self.bottom_frame = ctk.CTkFrame(master=self.scrollable_frame, width=480)
        self.bottom_frame.pack(fill=ctk.BOTH, expand=True)

        # Add some labels
        self.username_label = ctk.CTkLabel(master=self.top_frame, text=f'Username: {state.username}', font=label_font)
        self.username_label.place(x=10, y=5)

        self.user_ID_label = ctk.CTkLabel(master=self.top_frame, text=f'User ID: {state.user_ID}', font=label_font)
        self.user_ID_label.place(x=10, y=25)

        self.class_ID_label = ctk.CTkLabel(master=self.top_frame, text=f'Class ID: {state.current_class_ID}', font=label_font)
        self.class_ID_label.place(x=305, y=5)

        class_collection = DB['class']
        class_document = class_collection.find_one({'classID': state.current_class_ID})
        self.class_name = class_document.get('className')

        class_name_label = ctk.CTkLabel(master=self.top_frame, text=f'Class Name: {self.class_name}', font=label_font)
        class_name_label.place(x=305, y=25)

        self.option_button = ctk.StringVar(value='Options')
        self.option_menu = ctk.CTkOptionMenu(self.top_frame, values=['Add Samples', 'Train Model', 'Mark Attendance', 'Generate Report'], 
                                             variable=self.option_button, command=self.selected_option, width=40, font=button_font)
        self.option_menu.place(x=185, y=0)

        # Load and display the students
        self.load_students()

    def selected_option(self, selection):
        path = os.path.join(state.classes_path, self.class_name)
        
        if selection == 'Add Samples':
            self.add_student_samples(path)            

        elif selection == 'Train Model':

            if not check_for_samples(path):
                messagebox.showerror('Error', 'No samples exist for the students')
                return
            
            if len(os.listdir(path)) == 1:
                messagebox.showerror('Error', 'No students have been added to the class!')
                return
            
            create_dataset(path)
            messagebox.showwarning('Warning', 'The classification model is being trained, Do NOT close the program!')
            train_classification_model(path)

        elif selection == 'Mark Attendance':
            model_path = os.path.join(path, 'models')
            if not os.path.exists(model_path):
                messagebox.showerror('Error', 'The model was not trained')
                return
            mark_attendance(path)

        elif selection == 'Generate Report':
            global DB
            model_path = os.path.join(path, 'models')
            attendance_collection = DB['attendance']
            cursor = attendance_collection.find({'classID': state.current_class_ID})

            if not os.path.exists(model_path):
                messagebox.showerror('Error', 'The model was not trained')
                return
            
            if cursor is None:
                messagebox.showerror('Error', f'No attendance records for the class: {state.current_class_ID} were found!')
                return
            
            generate_reports(path)


    def load_students(self):
        # Clear the current list of student buttons
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

        class_collection = DB['class']
        class_document = class_collection.find_one({'classID': state.current_class_ID})
        
        if not class_document:
            messagebox.showerror('Error', 'Class not found!')
            return
        
        students_list = class_document.get('students')
        if not students_list:
            self.no_students_label = ctk.CTkLabel(master=self.bottom_frame, text='No Students Found!')
            self.no_students_label.pack(padx=5, pady=5)
            self.add_students_button = ctk.CTkButton(master=self.bottom_frame, text='Add Students', command=self.open_add_students_screen)
            self.add_students_button.pack(padx=5, pady=5)
        else:
            for student_document in students_list:
                student_ID = student_document.get('studentID')
                student_name = student_document.get('name')
                student_major = student_document.get('major')

                class_button = StudentButton(master=self.bottom_frame, student_ID=student_ID, student_name=student_name, major=student_major, command=self.open_student_info_screen)
                class_button.pack(pady=5)

    def add_student_samples(self, class_name):
        path = os.path.join(state.classes_path, class_name)
        gathering_samples(path)
        messagebox.showinfo('Info', 'The images were saved successfully, The model can now be trained!')


    def open_student_info_screen(self):
        self.withdraw()
        self.student_info_screen = StudentInfoScreen(master=self)
        self.student_info_screen.protocol('WM_DELETE_WINDOW', self.on_info_close)

    def on_info_close(self):
        self.student_info_screen.destroy()
        self.deiconify()

    def open_add_students_screen(self):
        self.withdraw()
        self.students_add_screen = AddStudentsScreen(master=self)
        self.students_add_screen.protocol('WM_DELETE_WINDOW', self.on_add_students_close)

    def on_add_students_close(self):
        self.students_add_screen.destroy()
        self.load_students()
        self.deiconify()


class AddStudentsScreen(BaseScreen, ctk.CTkToplevel):

    def __init__(self, master, **kwargs):

        super(AddStudentsScreen, self).__init__(master, **kwargs)
        BaseScreen.fix_window(self)
        self.students_list = []
        self.title('Add Students')
        self.setup_ui()

    def setup_ui(self):
        # Add labels and entries for student information
        self.scrollable_frame = ctk.CTkScrollableFrame(master=self)
        self.scrollable_frame.pack(expand=True, fill=ctk.BOTH)

        label_font = ctk.CTkFont(family='Arial', size=14)

        self.top_frame = ctk.CTkFrame(master=self.scrollable_frame)
        self.top_frame.pack()

        self.bottom_frame = ctk.CTkFrame(master=self.scrollable_frame)
        self.bottom_frame.pack()

        self.student_name_label = ctk.CTkLabel(self.top_frame, text='Student Name', font=label_font)
        self.student_name_label.pack(pady=5)
        self.student_name_entry = ctk.CTkEntry(self.top_frame)
        self.student_name_entry.pack(pady=5)

        self.student_age_label = ctk.CTkLabel(self.top_frame, text='Student Age', font=label_font)
        self.student_age_label.pack(pady=5)
        self.student_age_entry = ctk.CTkEntry(self.top_frame)
        self.student_age_entry.pack(pady=5)

        self.student_gender_label = ctk.CTkLabel(self.top_frame, text='Student Gender', font=label_font)
        self.student_gender_label.pack(pady=5)
        self.student_gender_entry = ctk.CTkEntry(self.top_frame)
        self.student_gender_entry.pack(pady=5)

        self.student_batch_label = ctk.CTkLabel(self.top_frame, text='Student Batch', font=label_font)
        self.student_batch_label.pack(pady=5)
        self.student_batch_entry = ctk.CTkEntry(self.top_frame)
        self.student_batch_entry.pack(pady=5)

        self.student_major_label = ctk.CTkLabel(self.top_frame, text='Student Major', font=label_font)
        self.student_major_label.pack(pady=5)
        self.student_major_entry = ctk.CTkEntry(self.top_frame)
        self.student_major_entry.pack(pady=5)

        self.add_button = ctk.CTkButton(self.bottom_frame, text='Add Student', command=self.add_student)
        self.add_button.grid(row=0, column=0, padx=15, pady=20, sticky="nsew")

        self.done_button = ctk.CTkButton(self.bottom_frame, text='Done', command=self.done)
        self.done_button.grid(row=0, column=1, padx=15, pady=20, sticky="nsew")

    def add_student(self):
        student_name = self.student_name_entry.get()
        student_age = self.student_age_entry.get()
        student_gender = self.student_gender_entry.get()
        student_batch = self.student_batch_entry.get()
        student_major = self.student_major_entry.get()

        if not all([student_name, student_age, student_gender, student_batch, student_major]):
            messagebox.showerror('Error', 'All fields are required')
            return
        
        if any([student_name.isspace(), student_age.isspace(), student_gender.isspace(), student_batch.isspace(), student_major.isspace()]):
            messagebox.showerror('Error', 'One or more of the input fields are blank')
            return

        student = {
            'name': student_name,
            'age': student_age,
            'gender': student_gender,
            'batch': student_batch,
            'major': student_major
        }

        self.students_list.append(student)

        self.student_name_entry.delete(0, ctk.END)
        self.student_age_entry.delete(0, ctk.END)
        self.student_gender_entry.delete(0, ctk.END)
        self.student_batch_entry.delete(0, ctk.END)
        self.student_major_entry.delete(0, ctk.END)

        messagebox.showinfo('Added', f'Student: {student_name} was added successfully!')


    def done(self):
        global DB
        class_collection = DB['class']

        if not self.students_list:
            messagebox.showerror('Error', 'No students were added!')
            return
        
        for student in self.students_list:
            global state
            student['studentID'] = generate_student_ID()
            class_collection.update_one({'classID': state.current_class_ID}, {'$push': {'students': student}})
            class_document = class_collection.find_one({'classID': state.current_class_ID})
            class_name = class_document.get('className')
            student_path = Path(state.classes_path / class_name / student['name'])
            if not student_path.exists():
                student_path.mkdir(exist_ok=True, parents=True)

        self.master.load_students()
        self.master.deiconify()
        self.destroy()


class StudentInfoScreen(BaseScreen, ctk.CTkToplevel):
    def __init__(self, master, **kwargs):
        super(StudentInfoScreen, self).__init__(master, **kwargs)
        BaseScreen.fix_window(self)
        self.title('Student Info')
        self.setup_ui()

    def setup_ui(self):
        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=480)
        self.scrollable_frame.pack(fill='both', expand=True)
        
        # Fetch class data
        class_collection = DB['class']
        class_document = class_collection.find_one({'classID': state.current_class_ID})
        if not class_document:
            messagebox.showerror('Error', 'Class not found!')
            return
        
        students_list = class_document.get('students')
        if not students_list:
            messagebox.showerror('Error', 'No students found in the class!')
            return
        
        label_font = ctk.CTkFont(family='Arial', size=18)

        # Initialize student variables
        student_ID = None
        student_name = None
        student_age = None
        student_gender = None
        student_batch = None
        student_major = None

        # Find the current student
        for student in students_list:
            if student.get('studentID') == state.current_student_ID:
                student_ID = student.get('studentID')
                student_name = student.get('name')
                student_age = student.get('age')
                student_gender = student.get('gender')
                student_batch = student.get('batch')
                student_major = student.get('major')
                break

        # Display student information
        if student_ID:
            student_ID_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'ID: {student_ID}', font=label_font)
            student_ID_label.pack(pady=5)
            student_name_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'Name: {student_name}', font=label_font)
            student_name_label.pack(pady=5)
            student_age_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'Age: {student_age}', font=label_font)
            student_age_label.pack(pady=5)
            student_gender_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'Gender: {student_gender}', font=label_font)
            student_gender_label.pack(pady=5)
            student_batch_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'Batch: {student_batch}', font=label_font)
            student_batch_label.pack(pady=5)
            student_major_label = ctk.CTkLabel(master=self.scrollable_frame, text=f'Major: {student_major}', font=label_font)
            student_major_label.pack(pady=5)
        else:
            messagebox.showerror('Error', 'Student not found!')


DB, client = initialize_database()
state = StateInfo()
initialize_folder_structures()

AuthenticationScreen()
