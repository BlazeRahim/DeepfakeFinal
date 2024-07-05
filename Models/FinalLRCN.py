import os
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set global video counter
a = 0


def extract_faces_from_video(video_path, fps=15, sequence_length=30):
    global a
    print(f"Extracting Vid No. {a}")
    a += 1
    cap = cv2.VideoCapture(video_path)

    # Initialize the CPU-based face detector
    face_cascade_cpu = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
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

        if len(faces) >= sequence_length:
            break

    cap.release()

    # Pad sequences if necessary
    while len(faces) < sequence_length:
        faces.append(np.zeros((224, 224, 3)))

    return np.array(faces)


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

    return np.array(features)


# Function to create LRCN model
def create_lrcn_model(sequence_length, feature_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(512, activation='relu'), input_shape=(sequence_length, feature_dim)))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model


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


# Define paths and sequence length
real_folder = r'Videos/Real'
fake_folder = r'Videos/Fake'
sequence_length = 30  # Length of the video sequence to use

# Extract faces from videos
real_faces = []
fake_faces = []

for filename in os.listdir(real_folder):
    if filename.endswith('.mp4'):
        real_faces.append(
            extract_faces_from_video(os.path.join(real_folder, filename), sequence_length=sequence_length))

for filename in os.listdir(fake_folder):
    if filename.endswith('.mp4'):
        fake_faces.append(
            extract_faces_from_video(os.path.join(fake_folder, filename), sequence_length=sequence_length))

# Extract features for each frame
real_features = [extract_features_vgg16(faces) for faces in real_faces]
fake_features = [extract_features_vgg16(faces) for faces in fake_faces]

# Create labels for real and fake faces
real_labels = np.ones(len(real_features))
fake_labels = np.zeros(len(fake_features))

# Combine features and labels
all_features = np.array(real_features + fake_features)
all_labels = to_categorical(np.concatenate([real_labels, fake_labels]), num_classes=2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# Create and train LRCN model
feature_dim = real_features[0].shape[1]
lrcn_model = create_lrcn_model(sequence_length, feature_dim)
lrcn_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

# Evaluate the model
y_pred = lrcn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Plot ROC curve and calculate AUC
plot_roc_curve(y_true_classes, y_pred[:, 1])

# Calculate and plot confusion matrix
print(classification_report(y_true_classes, y_pred_classes))
plot_confusion_matrix(y_true_classes, y_pred_classes, ['Fake', 'Real'])

print('Model trained successfully.')
