import streamlit as st
import pickle
import sklearn
import numpy as np

pipe = pickle.load(open('D:\Project\Price Regression\pipe.pkl','rb'))
df = pickle.load(open('D:\Project\Price Regression\df.pkl','rb'))

st.title("Laptop Price Predictor")

#brand
company = st.selectbox('Brand',df['Company'].unique())

#type
type = st.selectbox('Type',df['TypeName'].unique())

#Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

#weight
Weight = st.number_input('Weight')

#touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

#IPS
ips = st.selectbox('IPS',['No','Yes'])

#screensize
screen_size = st.number_input('Screen Size')

#resolution

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#Cpu
cpu =  st.selectbox('Cpu',df['Cpu Brand'].unique())

#hdd
hdd = st.selectbox('Hdd(in GB)',[0,128,256,512,1024,2048])

#Ssd
ssd = st.selectbox('SSD(in GB)',[0, 8,128,256,512,1024])

#Gpu
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

#os
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict'):
    #query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (np.sqrt((X_res**2) + (Y_res**2)))/screen_size
    query = np.array([company,type,ram,Weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query.reshape(1,12)
    st.title("The predicted price of laptop is" + str(int(np.exp(pipe.predict(query)[0]))))