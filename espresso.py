import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('expresso_processed.csv')

st.markdown("<h1 style = 'color: #53A8E6; text-align: center; font-size: 60px; font-family:Helvetica'>TELECHURN AFRICA</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #53A8E6; text-align: center; font-family: italic'>BUILT BY FAITH </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)


# #add image
st.image('IMG_8431.JPG',width = 600)

st.markdown("<h2 style = 'color:#53A8E63; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)



# primaryColor="#FF4B71"  
# backgroundColor="#FFBBEEF"
# secondaryBackgroundColor="#F4C8F7"
# textColor="#0C0C0C"
# font="Serif'



st.markdown("This study aims to develop an Espresso machine learning model to predict client churn probability in Mauritania and Senegal. Utilizing over 15 behavior variables, including usage patterns, demographics, and service preferences, the model will analyze historical data to identify key predictors of churn. By leveraging advanced machine learning algorithms such as decision trees and logistic regression, the model seeks to accurately forecast client churn and provide actionable insights for retention strategies. This research addresses the pressing need for telecommunications companies like Espresso to proactively mitigate churn and optimize customer retention efforts in dynamic African markets.")
st.sidebar.image('IMG_8434.JPG', width = 300,caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader('User Input Variables')

sel_cols = ['DATA_VOLUME', 'ON_NET', 'REGULARITY', 'REVENUE', 'FREQUENCE', 'MONTANT',
            'FREQUENCE_RECH', 'ARPU_SEGMENT', 'CHURN']


data_vol = st.sidebar.number_input('DATA_VOLUME', data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
on = st.sidebar.number_input('ON_NET', data['ON_NET'].min(), data['ON_NET'].max())
reg = st.sidebar.number_input('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
rev = st.sidebar.number_input('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
freq = st.sidebar.number_input('FREQUENCE', data['FREQUENCE'].min(), data['FREQUENCE'].max())
mont= st.sidebar.number_input('MONTANT', data['MONTANT'].min(), data['MONTANT'].max())
freq_rec = st.sidebar.number_input('FREQUENCE_RECH', data['FREQUENCE_RECH'].min(), data['FREQUENCE_RECH'].max())
arp = st.sidebar.number_input('ARPU_SEGMENT', data['ARPU_SEGMENT'].min(), data['ARPU_SEGMENT'].max())


#users input
input_var = pd.DataFrame()
input_var['DATA_VOLUME'] = [data_vol]
input_var['ON_NET'] = [on]
input_var['REGULARITY'] = [reg]
input_var['REVENUE'] = [rev]
input_var['FREQUENCE'] = [freq]
input_var['MONTANT'] = [mont]
input_var['FREQUENCE_RECH'] = [freq_rec]
input_var['ARPU_SEGMENT'] = [arp]


st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)


#import the transformers
data_vol = joblib.load('DATA_VOLUME_scaler.pkl')
rev = joblib.load('REVENUE_scaler.pkl')
mont = joblib.load('MONTANT_scaler.pkl')
arp = joblib.load('ARPU_SEGMENT_scaler.pkl')




# transform the users input with the imported encoders
input_var['DATA_VOLUME'] = data_vol.transform(input_var[['DATA_VOLUME']])
input_var['REVENUE'] = rev.transform(input_var[['REVENUE']])
input_var['MONTANT'] = mont.transform(input_var[['MONTANT']])
input_var['ARPU_SEGMENT'] = arp.transform(input_var[['ARPU_SEGMENT']])



# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)


model = joblib.load('MideExpressorModel.pkl')



predict = model.predict(input_var)

if st.button('Check Churn Probability'):
    predicted_churn= model.predict(input_var)
    st.success(f"Your Company's Probability Churn is { predicted_churn[0].round(2)}")
    st.snow()
