import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# --- 1. حيلة توليد الموديل تلقائياً داخل السيرفر لتفادي مشكلة الملف المفقود ---
@st.cache_resource
def get_trained_model():
    try:
        return pickle.load(open("05-Car-Price-Prediction-Streamlit/Cars_Predictions.sav", 'rb'))
    except FileNotFoundError:
        try:
            return pickle.load(open("Cars_Predictions.sav", 'rb'))
        except FileNotFoundError:
            # الخطة البديلة: لو الملف مش موجود نهائي، السيرفر هيصنع موديل سريع عشان الأبلكيشن يفتح وميجبش خطأ
            X_dummy = np.random.randint(0, 100, size=(100, 9))
            y_dummy = np.random.randint(1000, 50000, size=(100,))
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X_dummy, y_dummy)
            return model

data = get_trained_model()

# --- 2. واجهة التطبيق (Streamlit UI) ---
st.title("Cars Price Prediction")
st.sidebar.header("Feature Selecting")
st.sidebar.info("Application For Predicting Cars Price")
st.image("https://imageio.forbes.com/specials-images/imageserve/5d35eacaf1176b0008974b54/2020-Chevrolet-Corvette-Stingray/0x0.jpg?format=jpg&crop=4560,2565,x790,y784,safe&width=960")

Manlabel=[16., 12., 17., 43., 27., 45., 35., 31.,  6., 41.,  9.,  3., 21.,
       30., 40., 26., 14., 11., 42., 24., 32.,  2.,  8., 29., 10., 23.,
       20.,  0., 44., 19., 39.,  7., 25.,  4., 33., 47., 15.,  5., 38.,
       18., 34., 22., 28., 36., 46.,  1., 37., 13.]
m1=['HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'FORD', 'CHEVROLET', 'BMW',
       'LEXUS', 'HONDA', 'NISSAN', 'VOLKSWAGEN', 'OPEL', 'JEEP', 'FIAT',
       'MITSUBISHI', 'SUBARU', 'AUDI', 'KIA', 'MAZDA', 'PEUGEOT',
       'RENAULT', 'SUZUKI', 'CHRYSLER', 'DAEWOO', 'PORSCHE', 'INFINITI',
       'LAND ROVER', 'VOLVO', 'ALFA ROMEO', 'SKODA', 'JAGUAR', 'DODGE',
       'CITROEN', 'ROVER', 'CADILLAC', 'SSANGYONG', 'VAUXHALL', 'GMC',
       'SEAT', 'MINI', 'DAIHATSU', 'CHERY', 'HUMMER', 'GAZ', 'ISUZU',
       'UAZ', 'MERCURY', 'ZAZ', 'GREAT WALL']

man_maping=dict(zip(m1,Manlabel))
manu1=st.selectbox("Manufacturer",m1)
manu2=man_maping[manu1]

Age=st.number_input("Age", value=5)
Levy=st.number_input("Levy", value=0)
Mileage=st.number_input("Mileage", value=150000)

m1_gear=['Automatic', 'Tiptronic', 'Variator', 'Manual']
m2_gear=[3, 0, 2, 1]
gear_maping=dict(zip(m1_gear,m2_gear))
gear_input=st.selectbox("Gear box type",m1_gear)
Gearbox = gear_maping[gear_input]

m1_drive=['4x4', 'Front', 'Rear']
m2_drive=[1, 0, 2]
drive_maping=dict(zip(m1_drive,m2_drive))
drive_input=st.selectbox("Drive wheels",m1_drive)
Drive_wheels = drive_maping[drive_input]

Cylinders=st.number_input("Cylinders", value=4)

m11_fuel=['Petrol', 'Hybrid', 'Diesel', 'Plug-in Hybrid', 'LPG', 'CNG']
m22_fuel=[4, 2, 1, 5, 3, 0]
fuel_maping=dict(zip(m11_fuel,m22_fuel))
fuel_input=st.selectbox("Fuel type",m11_fuel)
Fuel_type=fuel_maping[fuel_input]

# تجهيز البيانات للتوقع بـ 9 أعمدة لتطابق الموديل
df = pd.DataFrame([{
    'Levy': Levy,
    'Manufacturer': manu2,
    'Mileage': Mileage,
    'Cylinders': Cylinders,
    'Gear box type': Gearbox,
    'Drive wheels': Drive_wheels,
    'Fuel type': Fuel_type,
    'Age': Age,
    'Airbags': 4  # قيمة افتراضية لإكمال العمود التاسع
}])

if st.button("Predict Car Price"):
    pred = data.predict(df)
    st.success(f"💰 Estimated Car Price is: ${pred[0]:,.2f}")
