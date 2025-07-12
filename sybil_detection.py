#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Input, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score



data = pd.read_csv('sybil_attack_dataset-1.csv')


#Convert Timestamp tp datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

#Extract temporal features
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour
data['Day_Of_Week'] = data['Timestamp'].dt.dayofweek

#Encode cyclic features
data['Hour_Sin'] = np.sin(2*np.pi*data['Hour']/24)
data['Hour_Cos'] = np.cos(2*np.pi*data['Hour']/24)
data['Day_Of_Week_Sin'] = np.sin(2*np.pi*data['Day_Of_Week']/7)
data['Day_Of_Week_Cos'] = np.cos(2*np.pi*data['Day_Of_Week']/7)


#Process IP_Address
data[['IP_Octel_1','IP_Octel_2','IP_Octel_3','IP_Octel_4']] = data['IP_Address'].str.split('.', expand=True).astype(int)


#Data Encoding
le_user = LabelEncoder()
le_action = LabelEncoder()

data['User_ID'] = le_user.fit_transform(data['User_ID'])
data['Action_Type'] = le_action.fit_transform(data['Action_Type'])

data = data.drop(columns = ['IP_Address'])


#Compute Correlations
correlation_matrix = data.corr()

#Plot
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

data = data.drop(columns=['Timestamp','Hour_Sin','Day_Of_Week_Sin','Day_Of_Week_Cos'])


#Sort data by User_ID and Timestamp
data = data.sort_values(by=['User_ID','Action_Type','Year','Month','Day'])


#Group data by User_ID to create sequences
sequence_data = []
labels = []

for user_id, group in data.groupby('User_ID'):
    label = group['Is_Sybil'].values[0]
    features = group.drop(columns = ['Is_Sybil']).values
    sequence_data.append(features)
    labels.append(label)


max_sequence_length = max(len(seq) for seq in sequence_data)
X_train, X_test, Y_train, Y_test = train_test_split(sequence_data, labels, test_size=0.2, random_state=42)
X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, padding='post', dtype='float32')
X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length, padding='post', dtype='float32')

#Input shape
input_shape = (X_train_padded.shape[1], X_train_padded.shape[2])
input_layer = Input(shape=input_shape)

#CNN Layer
cnn_layer = Conv1D(filters=2, kernel_size=1, activation='relu',padding='same')(input_layer)
cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)

#Reshaping before feeding into LSTM
cnn_layer = tf.keras.layers.Reshape((cnn_layer.shape[1], cnn_layer.shape[2]))(cnn_layer)

#RNN Layer
rnn_layer = LSTM(units=2, return_sequences=True, dropout=0.01, recurrent_dropout=0.01)(cnn_layer)
rnn_layer = Flatten()(rnn_layer)

dense_layer = Dense(units=1, activation='relu', kernel_regularizer=l2(0.001))(rnn_layer)
dense_layer = Dropout(0.01)(dense_layer)
output = Dense(units=1, activation='sigmoid')(dense_layer)

#Model Summary
model = Model(inputs=input_layer, outputs=output)
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

Y_train = np.array(Y_train).reshape(-1,1)
Y_test = np.array(Y_test).reshape(-1,1)

#Weight Calculation
Y_train_labels = np.argmax(Y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_labels), y=Y_train_labels)
class_weight_dict = dict(enumerate(class_weights))


#Training 
model.fit(X_train_padded, Y_train, epochs=5, batch_size=32, validation_data=(X_test_padded, Y_test), class_weight=class_weight_dict)

Y_pred = model.predict(X_test_padded)
Y_pred = (Y_pred > 0.5).astype("int32")
Y_pred = Y_pred.reshape(-1)

print("Test Accuracy :", accuracy_score(Y_test, Y_pred))
report = classification_report(Y_test, Y_pred, target_names=["Non-Sybil","Sybil"], output_dict=True)
print("\n Classification Report : \n", classification_report(Y_test, Y_pred, target_names=["Non-Sybil","Sybil"]))

# Extract precision, recall, and F1-score for each class
metrics = ['precision', 'recall', 'f1-score']
classes = ["Non-Sybil", "Sybil"]
data = {}
for metric in metrics:
  data[metric] = [report[cls][metric] for cls in classes]

# Create bar chart
x = np.arange(len(classes))
width = 0.2  # Width of each bar

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, data['precision'], width, label='Precision')
rects2 = ax.bar(x, data['recall'], width, label='Recall')
rects3 = ax.bar(x + width, data['f1-score'], width, label='F1-score')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Display the chart
plt.grid(axis='y')
fig.tight_layout()
plt.show()

