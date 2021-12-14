"""
Author: Ryan Lynch
CS401 Assignment 2
"""


import os
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def standardiseData(matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def plot_roc(labels, data, model):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False Positive Rate [%]")
    plt.ylabel("True Positive Rate [%]")
    plt.show()


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset = loadtxt('train-io.txt', delimiter=' ')

X = dataset[:, 0:10]
y = dataset[:, 10]

train_data_scaled = standardiseData(X)


def train():

    model = Sequential()

    model.add(Dense(128, input_dim=10, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    model.fit(train_data_scaled, y, epochs=1000)

    save_q = input("Save the model? (y/n").upper()

    if save_q == 'Y':
        model.save('ML_model')


def kfoldCrossValidation(X, y, k=8):
    num_validation_samples = len(X) // k
    validation_scores = []
    for fold in range(k):
        valid_x = X[num_validation_samples * fold: num_validation_samples * (fold + 1)]
        valid_y = y[num_validation_samples * fold: num_validation_samples * (fold + 1)]
        train_x = np.concatenate((X[:num_validation_samples * fold], X[num_validation_samples * (fold + 1):]))
        train_y = np.concatenate((y[:num_validation_samples * fold], y[num_validation_samples * (fold + 1):]))
        cur_model = keras.models.load_model('ML_model')
        cur_model.fit(train_x, train_y, epochs=175)
        valid_x = np.asarray(valid_x)
        valid_y = np.asarray(valid_y)
        eval = cur_model.evaluate(valid_x, valid_y)
        validation_scores.append(eval)
    return validation_scores


trainingQ = input("Train?").upper()
# train()


train_data_scaled = StandardScaler().fit_transform(X)
results = kfoldCrossValidation(train_data_scaled, y, k=8)
avg_acc = 0
avg_loss = 0
k = len(results)
for i in results:
    avg_loss += i[0] / k
    avg_acc += i[1] / k
print('AVG LOSS: ' + str(avg_loss))
print('AVG ACCURACY: ' + str(avg_acc))
print(avg_loss, avg_acc)

# model.fit(X, y, epochs=150, batch_size=32)
model = keras.models.load_model('ML_model')
_, accuracy = model.evaluate(train_data_scaled, y)
print('Accuracy: %.2f' % (accuracy * 100))

test_dataset = loadtxt('test-i.txt', delimiter=' ')

X = test_dataset[:, 0:10]

testing_output = model.predict(X)

test_output_file = open('test-o.txt', 'w')
for prediction in testing_output:
    # Round the outputs to 0 or 1 and write to file
    test_output_file.write(str(round(float(prediction))) + '\n')

plot_roc(y[-1000:], train_data_scaled[-1000:], model)

# data = pd.DataFrame(X, columns=range(1, 11))

