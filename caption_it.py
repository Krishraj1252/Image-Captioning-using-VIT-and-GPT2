#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


# In[51]:


from tensorflow.keras.models import Model,load_model
from keras.preprocessing import image


# In[91]:


model = load_model("model_9.h5")
model.make_predict_function()


# In[92]:


model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))
# re-structure the model
model_temp = Model(inputs=model_temp.inputs, outputs=model_temp.layers[-2].output)
model_temp.make_predict_function()


# In[93]:


with open("word_to_idx.pkl",'rb') as w2i:
    word_to_idx = pickle.load(w2i)

with open("idx_to_word.pkl",'rb') as i2w:
    idx_to_word = pickle.load(i2w)


# In[94]:


def preprocess_image(img):
    img = image.load_img(img, target_size = (224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    img = preprocess_input(img)
    return img


# In[95]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_temp.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector


# In[96]:


def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


# In[97]:


def caption_this_image(input_img): 

    photo = encode_image(input_img)
    

    caption = predict_caption(photo)
    # keras.backend.clear_session()
    return caption


# In[98]:

caption_this_image("image(1).jpg")

# In[ ]:




