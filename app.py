import streamlit as st
import base64
import numpy as np
import pandas as pd
import emoji
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.utils import to_categorical
data = pd.read_csv('emoji_data.csv', header = None)
data.head()
X = data[0].values
Y = data[1].values
Y[Y=='0v2']=0

emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}

def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])
tokenizer = Tokenizer()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2index = tokenizer.word_index

PAGE_TITLE = "Emoji suggestions"
st.set_page_config(page_title=PAGE_TITLE,layout="wide")
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""<style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-attachment: fixed;
        background-size: cover
    }}</style>""",unsafe_allow_html=True)
add_bg_from_local('source/charcoal.png') 
with open("css/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
st.write("we trained on less data, so it may predict few incorrect emojies. ")
with st.form("form 3",clear_on_submit=True):
    input_=st.text_area(label="Generate Emojies based on our feeling")
    out=st.form_submit_button("Submit")
    if out:
        model = load_model("network.h5")
        print(input_)
        test=[input_]
        test_seq = tokenizer.texts_to_sequences(test)
        print(test_seq)
        Xtest = pad_sequences(test_seq, maxlen = 10, padding = 'post', truncating = 'post')
        y_pred = model.predict(Xtest)
        y_pred = np.argmax(y_pred, axis = 1)
        print(y_pred)
        st.write(input_+label_to_emoji(y_pred[0]))