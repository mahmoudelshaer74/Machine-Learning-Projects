import streamlit as st
import pandas as pd
import pickle

data=pickle.load(open(r"C:\Users\mahmo\Cars_Predictions.sav",'rb'))


st.title("Cars Price Prediction")
st.sidebar.header("Feature Selecting")
st.sidebar.info("Application For Predicting Cars Price")
st.image("https://imageio.forbes.com/specials-images/imageserve/5d35eacaf1176b0008974b54/2020-Chevrolet-Corvette-Stingray/0x0.jpg?format=jpg&crop=4560,2565,x790,y784,safe&width=960")

##----------------------------------------------------------------------------------------------




Manlabel=[16., 12., 17., 43., 27., 45., 35., 31.,  6., 41.,  9.,  3., 21.,
       30., 40., 26., 14., 11., 42., 24., 32.,  2.,  8., 29., 10., 23.,
       20.,  0., 44., 19., 39.,  7., 25.,  4., 33., 47., 15.,  5., 38.,
       18., 34., 22., 28., 36., 46.,  1., 37., 13.]
man=['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
       'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN',
       'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA',
       'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT', 'INFINITI',
       'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ',
       'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR',
       'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC',
       'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION',
       'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
       'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE',
       'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']
man_maping=dict(zip(man,Manlabel))
manu1=st.selectbox("Manufacturer",man)
manf=man_maping[manu1]
#-----------------------------------
m1=['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon',
       'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine',
       'Pickup']
m2=[3, 4, 8, 9, 6, 0, 1, 5, 2, 7]

man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Category",m1)
category=man_maping[manu1]

#---------------------------------------------------------------------

m2=[0, 1]
m1=['Yes', 'No']

man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Leather interior",m1)
Leather=man_maping[manu1]


#---------------------------------------------------------------------
m1=[ 3.5,  3. ,  1.3,  2.5,  2. ,  1.8,  2.4,  4. ,  1.6,  3.3,  2.2,
        4.7,  1.5,  4.4,  1.4,  3.6,  2.3,  5.5,  2.8,  3.2,  3.8,  4.6,
        1.2,  5. ,  1.7,  2.9,  0.5,  1.9,  2.7,  4.8,  5.3,  0.4,  1.1,
        2.1,  0.7,  5.4,  3.7,  1. ,  2.6,  0.8,  0.2,  5.7,  6.7,  6.2,
        3.4,  6.3,  4.3,  4.2,  0. , 20. ,  0.3,  5.9,  5.6,  6. ,  0.6,
        6.8,  4.5,  7.3,  0.1,  3.1,  6.4,  3.9,  0.9,  5.2,  5.8]

Engine=st.selectbox("Engine volume",m1)

#---------------------------------------------------------------------
m1=['Left wheel', 'Right-hand drive']
m2=[1, 0]

man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Wheel",m1)
Wheel=man_maping[manu1]
#-----------------------------------------------------------------------
m1=['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red',
       'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige',
       'Carnelian red', 'Purple', 'Pink']
m2=[ 1, 14, 12,  7,  2, 13, 11,  8,  6, 15,  3,  5,  0,  4, 10,  9]
man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Color",m1)
color=man_maping[manu1]
#-----------------------------------------------------------------------

m1=[12,  8,  2,  0,  4,  6, 10,  3,  1, 16,  5,  7,  9, 11, 14, 15, 13]

Airbags=st.selectbox("Airbags",m1)

#--------------------
m1=['RX 450','Equinox','FIT','E 230 124','RX 450 F SPORT','Prius C aqua']
m2=[890,458,477,485,470,833]
man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Model",m1)
manu2=man_maping[manu1]
#-----------------------------------
Age=st.number_input("Age")
#--------------------------
Levy=st.number_input("Levy")
#--------------------------
Mileage=st.number_input("Mileage")

#-----------------------------------------
m1=['Automatic', 'Tiptronic', 'Variator', 'Manual']
m2=[3, 0, 2, 1]
man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Gear box type",m1)
Gearbox =man_maping[manu1]
#-----------------------------------------
m1=['4x4', 'Front', 'Rear']
m2=[1, 0, 2]
man_maping=dict(zip(m1,m2))
manu1=st.selectbox("Drive wheels",m1)
Drive_wheels =man_maping[manu1]
#------------------------------------------
Cylinders=st.number_input("Cylinders")
#------------------------------------------
m11=['Petrol', 'Hybrid', 'Diesel', 'Plug-in Hybrid', 'LPG', 'CNG']
m22=[4, 2, 1, 5, 3, 0]

man_maping=dict(zip(m11,m22))
manu3=st.selectbox("Fuel type",m11)
Fuel_type=man_maping[manu3]
#---------------------------------

df=pd.DataFrame({"Manufacturer":manf,"Model":manu2,"Category":category,"Leather interior":Leather ,"Fuel type":Fuel_type,"Mileage":Mileage,"Gear box type":Gearbox,
                 "Drive wheels":Drive_wheels,"Wheel":Wheel,"Color":color,"Levy":Levy,"Engine volume":Engine ,"Cylinders":Cylinders,
                 "Airbags":Airbags,"Age":Age},index=[0])

print(df.columns)
print(df)
st.sidebar.divider()
p=st.sidebar.button("Predict Price")
if p:
    pre=data.predict(df)
    st.sidebar.write("Price Is:",pre)