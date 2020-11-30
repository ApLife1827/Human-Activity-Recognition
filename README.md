# Human-Activity-Recognition
Human Activity Recognition, or HAR for short, is the problem of predicting what a person is doing based on a trace of their movement using sensors.
Movements are often normal indoor activities such as standing, sitting, jumping, and going up stairs.
Sensors are often located on the subject, such as a smartphone or vest, and often record accelerometer data in three dimensions (x, y, z).
It is a challenging problem because there is no clear analytical way to relate the sensor data to specific actions in a general way. It is technically challenging because of the large volume of sensor data collected (e.g. tens or hundreds of observations per second) and the classical use of hand crafted features and heuristics from this data in developing predictive models.

# Dataset
The dataset was made available and can be downloaded for free from the UCI Machine Learning Repository:
 Human Activity Recognition Using Smartphones Data Set
 
 he raw data is not available. Instead, a pre-processed version of the dataset was made available.

The pre-processing steps included:
<ul>
  <li>Pre-processing accelerometer and gyroscope using noise filters.</li>
  <li>Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap.</li>
  <li>Splitting of accelerometer data into gravitational (total) and body motion components.</li></ul>
  
  The dataset was split into train (70%) and test (30%) sets based on data for subjects, e.g. 21 subjects for train and nine for test.
  # load dataset
import pandas as pd
import numpy as np

DATADIR = 'HARDataset'
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

def _read_csv(filename):

    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(subset):

    signals_data = []
    for signal in SIGNALS:
        filename = '{0}/{1}/Inertial Signals/{2}_{3}.txt'.format(DATADIR,subset,signal,subset)
        signals_data.append(
            _read_csv(filename).as_matrix()
        )
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):
    
    filename = '{0}/{1}/y_{2}.txt'.format(DATADIR,subset,subset)
    y = _read_csv(filename)[0]
    return pd.get_dummies(y).as_matrix()


def load_data():
    
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')
    print(X_train.shape)
    return X_train, X_test, y_train, y_test

X_train, X_test, Y_train, Y_test = load_data()

y_test=Y_test.argmax(1)

# CNN+LSTM Model  -- Accuracy of model= 95.68%
# CNN
model = Sequential()

model.add(Conv1D(64,2,input_shape=(128, 9),activation='relu'))

model.add(MaxPooling1D(pool_size=2,padding='valid'))

model.add(Conv1D(128,2,activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.2))

#model.add(Flatten())

#model.add(Dense(n_classes, activation='sigmoid')) #n_classes

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# LSTM
model.add(LSTM(n_hidden,input_shape=(timesteps, input_dim)))

model.add(Dropout(0.2))

model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=2,callbacks=[tensorboard, checkpoint])
          
# Save and Load model
model.save_weights('cnn_lstm.h5')

model.load_weights('cnn_lstm.h5')

# Evaluate
predict=model.predict(X_test)

pred_index_total=[]

for pred in predict:

    pred_index = []
    pred_list=pred.tolist()
    index_max=pred_list.index(max(pred_list))
    pred_index.append(index_max)
    pred_index_total.append(np.array(pred_index))
    
print(pred_index_total)

one_hot_predictions=one_hot(np.array(pred_index_total))

prediction=one_hot_predictions.argmax(1)

confusion_matrix = metrics.confusion_matrix(y_test, prediction)

print("%%%%%%%%%%%%%%%",confusion_matrix)

# Plot Results:
width = 12
height = 12

normalised_confusion_matrix = np.array(confusion_matrix, dtype=float32)/np.sum(confusion_matrix)*100

print(normalised_confusion_matrix)

plt.figure(figsize=(width, height))

plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)

plt.title("Confusion matrix \n(normalised to % of total test data)")

plt.colorbar()

tick_marks = np.arange(n_classes)

plt.xticks(tick_marks,lables,rotation=90)

plt.yticks(tick_marks,lables)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()


  
