import cv2
import pickle
import numpy as np
import os

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

model_2 = 'random_forest_classifier'
model = f'{model_2}.pkl'
detection_model_path = os.path.normpath(r'C:\Users\laptop zone\SmartAttend\classes\DSA109\models')
detection_model_path = os.path.join(detection_model_path, model)
encoder_path = os.path.normpath(r'C:\Users\laptop zone\SmartAttend\classes\DSA109\models\label_encoder.pkl')
detection_model = load_caffe_model(r'C:\Users\laptop zone\SmartAttend\detection_model\architecture.prototxt', 
                             r'C:\Users\laptop zone\SmartAttend\detection_model\weights.caffemodel')

classification_model = None
encoder = None

with open(detection_model_path, 'rb') as model_file:
    classification_model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()

    detections = caffe_detect_faces(frame, detection_model)
    height, width, _ = frame.shape
    threshold = 0.4

    output_frame = frame.copy()  # Copy the original frame

    if detections:
        for detection in detections:
            x1, y1, x2, y2 = int(detection['x1'] * width), int(detection['y1'] * height), int(detection['x2'] * width), int(detection['y2'] * height)
            
            if x1 < x2 and y1 < y2:  # Ensure the coordinates are valid
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:  # Check if crop is not empty
                    crop = cv2.resize(crop, (128, 128))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop = crop.flatten() / 255.0
                    input_data = np.array(crop).reshape(1, -1)  # Reshape to 2D array
                    prediction_encoded = classification_model.predict(input_data)
                    prediction_probabilities = classification_model.predict_proba(input_data)
                    confidence_score = np.max(prediction_probabilities)
                    most_confident_class = np.argmax(prediction_probabilities)
                    prediction = encoder.inverse_transform([most_confident_class])
                    text = str(prediction[0]) if confidence_score >= threshold else 'Unknown'
                    output_frame = cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    position = (x1, y1 - 10)  # Placing text above the rectangle
                    # Add the text above the rectangle
                    cv2.putText(output_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow('Frame', output_frame)  # Show the frame regardless of detections

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
