import numpy as np
from numpy import float32,int32
np.random.seed(42)
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout,Permute,Reshape
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten,Bidirectional, RepeatVector, TimeDistributed, Concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


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

LABELS = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
lables=np.array(LABELS)

CHECK_ROOT = 'checkpoint/'
if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)
epochs = 20 # 30
batch_size = 16
n_hidden = 32

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()
y_test=Y_test.argmax(1)

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)


#CNN
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

# callback: draw curve on TensorBoard
tensorboard = TensorBoard(log_dir='log_cnnlstm', histogram_freq=0, write_graph=True, write_images=True)
# callback: save the weight with the highest validation accuracy
filepath=os.path.join(CHECK_ROOT, 'weights-improvement-{val_acc:.4f}-{epoch:04d}.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')

model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=2,callbacks=[tensorboard, checkpoint])
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
