import wfdb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Specify the directory where the MIT-BIH Arrhythmia Database files are located
mitdb_dir = 'D:/Arrhythmia/mit-bih-arrhythmia-database-1.0.0/'

# List of record names to include in the dataset
record_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
                '108', '109', '111', '112', '113', '114', '115', '116', 
                '117', '118', '119', '121', '122', '123', '124', '200', 
                '201', '202', '203', '205', '207', '208', '209', '210', 
                '212', '213', '214', '215', '217', '219', '220', '221', 
                '222', '223', '228', '230', '231', '232', '233', '234']

# Mapping symbol to description
symbol_to_description = {
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'A': 'Atrial premature contraction',
    'a': 'Aberrated atrial premature beat',
    'V': 'Premature ventricular contraction',
    'F': 'Fusion of ventricular and normal beat',
    'j': 'Nodal (junctional) escape beat',
    'E': 'Ventricular escape beat',
    '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat',
    'Q': 'Unclassified',
    '?': 'Beat not classified during learning'
}

# Function to get description from symbol
def get_description(symbol):
    return symbol_to_description.get(symbol, 'Unknown')

# Initialize empty lists to store the segments and labels
segments = []
labels = []

# Define the length of each segment (e.g., 360 samples for 1 second at 360 Hz)
segment_length = 360

# Iterate over the record names
for record_name in record_names:
    # Construct the file path using os.path.join()
    record_path = os.path.join(mitdb_dir, record_name)
    
    # Read the ECG signals
    signal, _ = wfdb.rdsamp(record_path)

    # Read the annotations
    annotations = wfdb.rdann(record_path, 'atr')
    
    # Extract the signal data and annotations
    signal_data = signal[:,0]  # Use the first channel of the signal
    annotation_samples = annotations.sample
    annotation_symbols = annotations.symbol
    
    # Segment the signal based on the annotation points
    for sample, symbol in zip(annotation_samples, annotation_symbols):
        if sample - segment_length // 2 >= 0 and sample + segment_length // 2 <= len(signal_data):
            segment = signal_data[sample - segment_length // 2 : sample + segment_length // 2]
            segments.append(segment)
            labels.append(symbol)

# Convert segments and labels to numpy arrays
segments = np.array(segments)
labels = np.array(labels)

# Perform label encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for input to the CNN model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert the labels to categorical format
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the CNN model architecture
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot the training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Convert the predicted labels back to the original label names
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Print the predicted labels for the test set
print("Predicted Labels:")
print(predicted_labels)

# Generate classification report
print("Classification Report:")
print(classification_report(label_encoder.inverse_transform(np.argmax(y_test, axis=1)), predicted_labels))


# Plot confusion matrix with descriptions
cm = confusion_matrix(label_encoder.inverse_transform(np.argmax(y_test, axis=1)), predicted_labels)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(labels)))
plt.xticks(tick_marks, [get_description(symbol) for symbol in label_encoder.classes_], rotation=90)
plt.yticks(tick_marks, [get_description(symbol) for symbol in label_encoder.classes_])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Compute class counts
class_counts = np.bincount(labels)

# Sort class counts and get corresponding labels
sorted_indices = np.argsort(class_counts)[::-1]  # Sort in descending order
top_n = 10  # Number of top classes to display
sorted_class_counts = class_counts[sorted_indices][:top_n]
sorted_class_labels = label_encoder.classes_[sorted_indices][:top_n]

# Compute counts for the "Other" category
other_count = np.sum(class_counts[top_n:])

# Plot pie chart for class distribution with descriptions
class_descriptions = [get_description(symbol) for symbol in sorted_class_labels]
class_descriptions.append('Other')
sorted_class_counts = np.append(sorted_class_counts, other_count)

plt.figure(figsize=(10, 10))
plt.pie(sorted_class_counts, labels=class_descriptions, autopct='%1.1f%%', startangle=140, counterclock=False)
plt.axis('equal')
plt.title('Top {} Class Distribution'.format(top_n))
plt.show()
