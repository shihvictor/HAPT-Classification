from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path
from model_5 import model


print("Running...")

MODEL_NAME = 'model_5'

### HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 10

### CIFAR10 dataset loading:
### Partition data - data is already partioned from unpacking here:
# (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = []
y_train = []
x_test = []
y_test = []
with open("data/train/X_train.txt") as fp:
    while True:
        line = fp.readline()
        if not line:
            break
        t0 = [float(x) for x in line.split(" ")]
        x_train.append(t0)

with open("data/train/y_train.txt") as fp:
    while True:
        line = fp.readline()
        if not line:
            break
        t1 = float(line)
        y_train.append([t1])

with open("data/test/X_test.txt") as fp:
    while True:
        line = fp.readline()
        if not line:
            break
        t2 = [float(x) for x in line.split(" ")]
        x_test.append(t2)

with open("data/test/y_test.txt") as fp:
    while True:
        line = fp.readline()
        if not line:
            break
        t3 = float(line)
        y_test.append([t3])

x_train = np.reshape(np.array(x_train), (7767, 561, 1))
x_test = np.reshape(np.array(x_test), (3162, 561, 1))
y_train = np.array(np.subtract(y_train, 1))
y_test = np.array(np.subtract(y_test, 1))
# Note that there are 12 classes.
input_shape = (561,1) # get 1st sample's shape.

# Check shape of each partition. Each img is 32x32x3. 50000 in training set, 10000 in test set.
print("x_train shape = " + str(np.shape(x_train)))
print("y_train shape = " + str(np.shape(y_train)))
print("x_test shape = " + str(np.shape(x_test)))
print("y_test shape = " + str(np.shape(y_test)))


def plot_Acc_And_Loss(history_dict, save=True):
    """
    Plots loss and accuracy of train and val data over epochs.
    :return:
    """
    plt.plot(history_dict['accuracy'])
    plt.plot(history_dict['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save: plt.savefig('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+"_accuracy.png")
    plt.show()

    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save: plt.savefig('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+"_loss.png")
    plt.show()



### Compile a model.
model = model(input_shape)
opt = Adam(learning_rate=.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=['accuracy']
model.compile(optimizer=opt, loss=loss, metrics=metrics)
model.summary()


### Train and Predict.
model_checkpoint = ModelCheckpoint(filepath='model/'+MODEL_NAME,
                                       verbose=1,
                                       monitor='val_loss',
                                       save_best_only=True)
Path('model_logs/'+MODEL_NAME+'_logs/').mkdir(parents=True)
csv_logger = CSVLogger(filename='model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_log.csv', separator=',', append=True)
# t0 = len(x_train)//BATCH_SIZE
model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, callbacks=[csv_logger, model_checkpoint], validation_data=(x_test, y_test))

"""Save model history and plot loss and acc"""
with open('model/'+MODEL_NAME+'/trainHistoryDict', 'wb') as file_name:
    pickle.dump(model_history.history, file_name)       # Save history dict
plot_Acc_And_Loss(model_history.history)        # Plot acc and loss over epochs
with open('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_summary', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


### Evaluate model.
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

pred_outs = model.predict(x_test)

pred_labels = np.argmax(pred_outs,axis=1)




