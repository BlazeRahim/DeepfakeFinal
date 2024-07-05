import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
import cv2
from keras_preprocessing.image import img_to_array
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

a = 0

def extract_faces_from_video(video_path):
    global a
    print(f"Extracting Vid No. {a}")
    a += 1
    cap = cv2.VideoCapture(video_path)

    # Set the desired frames per second (fps)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # Initialize the CPU-based face detector
    face_cascade_cpu = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces on CPU
            detected_faces = face_cascade_cpu.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                face = frame[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))
                faces.append(face)

        frame_count += 1

    cap.release()
    return faces


def extract_features_vgg16(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    features = []

    for img in images:
        img = cv2.resize(img, (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feature = model.predict(x)

        features.append(feature.flatten())

    return features

# Function to plot ROC curve and calculate AUC
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

real_folder = r'Videos\Real'
fake_folder = r'Videos\Fake'

real_faces = []
fake_faces = []

for filename in os.listdir(real_folder):
    if filename.endswith('.mp4'):
        real_faces.extend(extract_faces_from_video(os.path.join(real_folder, filename)))

for filename in os.listdir(fake_folder):
    if filename.endswith('.mp4'):
        fake_faces.extend(extract_faces_from_video(os.path.join(fake_folder, filename)))

real_features = extract_features_vgg16(real_faces)
fake_features = extract_features_vgg16(fake_faces)

# Create labels for real and fake faces
real_labels = np.ones(len(real_features))
fake_labels = np.zeros(len(fake_features))

# Combine features and labels
all_features = np.concatenate([real_features, fake_features], axis=0)
all_labels = np.concatenate([real_labels, fake_labels], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# Train a classifier (Support Vector Machine)
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# Predict probabilities
y_score = clf.predict_proba(X_test)[:, 1]

# Plot ROC curve and calculate AUC
plot_roc_curve(y_test, y_score)

# Calculate and plot confusion matrix
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, ['Fake', 'Real'])

print('Model trained successfully.')
