import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
    

    
    
def main():
    st.set_page_config(
        page_title="GlucoGuard",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state= "auto",
    )
     
    st.title('Diabetes Prediction')
    
    
    
    Glucose=st.slider('Glucose level',min_value=0,max_value=199,value=60,step=1)
    BloodPressure=st.slider('BloodPressure level in mm Hg',min_value=0,max_value=140,value=80,step=1)
    Insulin=st.slider('Insulin level in mu U/ml',min_value=0,max_value=1000,value=400,step=1)
    DiabetesPedigreeFunction=st.slider('Diabetes Pedigree Function value',min_value=0.000, step=0.001, max_value=3.0, value=0.045, format="%3f")
    Age=st.slider('Age of a person',min_value=10,max_value=100,value=21,step=1)
    BMI=st.slider('BMI value',min_value=0.0,max_value=70.0,value=33.3,step=0.01)
    SkinThickness=st.slider('SkinThickness in mm',min_value=0,max_value=100,value=40,step=1)
    Pregnancies=st.slider('Number of Pregnancies',min_value=0,max_value=20,value=4,step=1)
   
    
   
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    if diagnosis != '':
        st.success(diagnosis)  

    st.write("Made with ❤️ by Shivam Pandey")




if __name__ == '__main__':
     main()    
    
    
