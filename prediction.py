import tensorflow as tf
import numpy as np

## we will try making predictions using the model here
Pmodel = tf.keras.models.load_model("C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\Project 3\\composing\\model1")

## --------------------------------------------------------------------------------------

text = open('C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\Project 3\\composing\\pianoabc.txt', 'r').read()
unique_text = list(set(text))
unique_text.sort()
text_to_num = {}
num_to_text = {}
for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data
num_text = []
for i in text:
    num_text.append(text_to_num[i])

## --------------------------------------------------------------------------------------

music = [] ## will store all the new values

## first input value that we will pass through the model
first_input = num_text[117:117+25]
first_input = tf.one_hot(first_input, 31)
first_input = tf.expand_dims(first_input, axis=0)

for i in range(200): ## repeat 200 times
    prediction = Pmodel.predict(first_input) ## making a prediction using the model and an initial input
    """ prediction = np.argmax(prediction[0]) """ ## extracting only the best prediction 
    prediction = text_to_num[np.random.choice(unique_text, 1, p=prediction[0])[0]] ## use prediction based on probability instead
    music.append(prediction)
    next_input = first_input.numpy()[0][1:] ## new input shifted one index to the right
    one_hot_num = tf.one_hot(prediction, 31) ## using the prediction we got from earlier for the new prediction
    first_input = np.vstack([next_input, one_hot_num.numpy()]) ## update first_input so that it is shifted
    first_input = tf.expand_dims(first_input, axis=0)

## converting each number in music[] to text
music_text = []
for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))

