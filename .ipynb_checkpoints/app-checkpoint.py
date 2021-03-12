from model import *
import pickle
import numpy as np
import streamlit as st
from PIL import Image

# with open('model.pkl', 'rb') as f:
#     model = pickle.loads(f)
    
st.title('UFC Fight Predictor :punch:')
st.write('''This app predicts a UFC fight winner with a Random Forest binary classifier. It uses various data 
            preprocessing methods such as KNN imputation for missing data and one-hot encoding for categorical
            variables.
         ''')
st.write('Please fill out all the information for fighter 1 and fighter 2 on the left sidebar.')
st.write('When you\'re done, please click the "Predict" button at the bottom to predict who won the fight!')


st.sidebar.title('Fighter Stats')

image = Image.open('ufc.jpg') 
st.image(image)

gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
gender = gender.upper()

st.sidebar.write('Fighter 1')
r_name = st.sidebar.text_input('Name', key=1)
r_odds = st.sidebar.slider('American Odds', -500, 500, key=2)
r_age = st.sidebar.slider('Age', 18, 50, key=3)
r_wins = st.sidebar.slider('Total Win Record', 0, 50, key=4)
r_losses = st.sidebar.slider('Total Loss Record', 0, 50, key=5)
r_height = st.sidebar.slider('Height (cm)', 150, 220, key=6)
r_reach = st.sidebar.slider('Reach (cm)', 150, 220, key=7)
r_stance = st.sidebar.selectbox('Stance', ['Orthodox', 'Southpaw', 'Switch', 'Open Stance'], key=8)
r_sig_str = st.sidebar.slider('Significant Strikes Landed (per minute)', 0, 175, key=9)
r_sig_str_pct = st.sidebar.slider('Significant Striking Accuracy (percentage)', 0.0, 1.0, key=10)
r_avg_sub = st.sidebar.slider('Average Submissions Attempted (per 15 minutes)', 0.0, 10.0, step=0.1, key=11)
r_avg_td = st.sidebar.slider('Average takedowns landed (per 15 minutes)', 0, 15, key=12)
r_avg_td_pct = st.sidebar.slider('Takedown Accuracy (percentage)', 0.0, 1.0, key=13)

st.sidebar.write('Fighter 2')
b_name = st.sidebar.text_input('Name', key=14)
b_odds = st.sidebar.slider('American Odds', -500, 500, key=15)
b_age = st.sidebar.slider('Age', 18, 50, key=16)
b_wins = st.sidebar.slider('Total Win Record', 0, 50, key=17)
b_losses = st.sidebar.slider('Total Loss Record', 0, 50, key=18)
b_height = st.sidebar.slider('Height (cm)', 150, 220, key=19)
b_reach = st.sidebar.slider('Reach (cm)', 150, 220, key=20)
b_stance = st.sidebar.selectbox('Stance', ['Orthodox', 'Southpaw', 'Switch', 'Open Stance'], key=21)
b_sig_str = st.sidebar.slider('Significant Strikes Landed (per minute)', 0, 175, key=22)
b_sig_str_pct = st.sidebar.slider('Significant Striking Accuracy (percentage)', 0.0, 1.0, key=23)
b_avg_sub = st.sidebar.slider('Average Submissions Attempted (per 15 minutes)', 0.0, 10.0, step=0.1, key=24)
b_avg_td = st.sidebar.slider('Average takedowns landed (per 15 minutes)', 0, 15, key=25)
b_avg_td_pct = st.sidebar.slider('Takedown Accuracy (percentage)', 0.0, 1.0, key=26)

load = st.button('Predict')
if load and r_name != '' and b_name != '':
    input_data = np.array([r_odds, b_odds, gender, b_sig_str,
           r_sig_str, b_sig_str_pct, r_sig_str_pct,
           b_avg_sub, r_avg_sub, b_avg_td, r_avg_td,
           b_avg_td_pct, r_avg_td_pct, b_losses, r_losses, b_wins,
           r_wins, b_stance, r_stance, b_height, r_height,
           b_reach, r_reach, b_age, r_age])
    
    prediction = pipe.predict(input_data.reshape(1,25))[0]
    if prediction == 1:
        st.write(f'{r_name} will win! :muscle:')
    elif prediction == 0:
        st.write(f'{b_name} will win! :muscle:')
    st.sidebar.balloons()
elif load and (r_name == '' or b_name == ''):
    st.write('You didn\'t input your fighter\'s names! :angry:')