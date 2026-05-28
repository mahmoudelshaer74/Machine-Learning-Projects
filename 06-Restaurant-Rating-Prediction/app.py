import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

scaler=joblib.load("scaler.pkl")

st.title("Restaurant Rating Prediction App")
st.caption("This App Help You To Predict a Restaurant Review Class")
st.divider()

Averagecost=st.number_input("Please Enter The Estimated Average Cost For Two",min_value=50,max_value=999999,step=200,value=1000)
TableBooking=st.selectbox("Restaurant Has Table Booking ?",["Yes","No"])
OnlineDelivery=st.selectbox("Restaurant Has Online Delivery ?",["Yes","No"])
PriceRange=st.selectbox("What Is The Price Range (1 : Cheapest ,4 : Most Expensive )",[1,2,3,4])
PredictButton=st.button("Predict The Review !")
st.divider()

model=joblib.load('mlmodel.pkl')
BookingStatus= 1 if TableBooking =="Yes" else 0
DeliverStatus= 1 if OnlineDelivery =="Yes" else 0
Values=[[Averagecost,DeliverStatus,BookingStatus,PriceRange]]
X_Values=np.array(Values)

X=scaler.transform(X_Values)
if PredictButton:
    st.snow()
    prediction=model.predict(X)
    if prediction < 2.5:
       st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4:
        st.write("Good")
    elif prediction < 4.5:
        st.write("very Good")
    else:
        st.write("Excellent")