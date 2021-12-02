"""
Author: Ryan Lynch
CS401 Assignment 2
"""


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow

dataset = loadtxt('/usr/local/home/u180518/Documents/CS401/Ass2/cs401-f2021-master-hw2/hw2//CS401-HW2/train-io.txt', delimiter=' ')

X = dataset[:, 0:10]
y = dataset[:, 10]

model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=32)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

