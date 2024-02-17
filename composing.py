## getting a list of all the possible characters in the text file
text = open('C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\Project 3\\composing\\pianoabc.txt', 'r').read()
unique_text = list(set(text))
unique_text.sort()

## assigning each character a unique number
text_to_num = {}
num_to_text = {}
for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

## converting the entire text file into numbers
num_text = []
for i in text:
    num_text.append(text_to_num[i])

## --------------------------------------------------------------------------------------

import tensorflow as tf

## filling in trainX and trainY with values from num_text
trainX = []
trainY = []
for i in range(0, len(num_text) - 25):
    trainX.append(num_text[i:i + 25])
    trainY.append(num_text[i + 25])

## use one-hot-encoding so that, for example, the number 2 will become [0,0,1,0,0 ... 0,0]
trainX = tf.one_hot(trainX, 31)
trainY = tf.one_hot(trainY, 31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, 31)), ## doesn't need an activation function
    tf.keras.layers.Dense(31, activation='softmax') ## softmax because we're using crossentropy
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) ## not spase because we used one-hot-encoding
model.fit(trainX, trainY, batch_size=64, epochs=40, verbose=2) ## takes forever with my terrible laptop
model.save("C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\Project 3\\composing\\model1")  