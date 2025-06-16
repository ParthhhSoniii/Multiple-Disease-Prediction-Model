import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


# http://localhost:8501/
# python -m streamlit run .\Multiple_Disease_Detection\Project.py


loaded_model = pickle.load(open(r'C:\Users\jayeshsoni\OneDrive\Desktop\Projects\Multiple_Disease_Detection\Multiple_Disease_Detection.sav','rb'))
loaded_symptoms_list = pickle.load(open(r'C:\Users\jayeshsoni\OneDrive\Desktop\Projects\Multiple_Disease_Detection\symptoms_list.pkl','rb'))
loaded_Disease_list = pickle.load(open(r'C:\Users\jayeshsoni\OneDrive\Desktop\Projects\Multiple_Disease_Detection\Disease_list.pkl','rb'))
loaded_le = pickle.load(open(r'C:\Users\jayeshsoni\OneDrive\Desktop\Projects\Multiple_Disease_Detection\LabelEncoder_for_ML_Model.pkl','rb'))
le = LabelEncoder()

loaded_Disease_list = list((loaded_Disease_list).unique())

def runModel(selected_symptoms):
    symptom_code=[]
    get_index = []

    for element in selected_symptoms:
        get_index.append(list(loaded_symptoms_list).index(element))

    for i in range (0,131):   
        if i in get_index:
            symptom_code.append(1)
        else:
            symptom_code.append(0)
    symptom_code = np.asarray(symptom_code)
    symptom_code = symptom_code.reshape(1,-1)

    st.write((f"You have {loaded_le.inverse_transform(loaded_model.predict(symptom_code))}"))
    st.caption("Subscribe to get Medication")
    if st.button("**:grey[Get Subcription]**"):
        pass


with st.sidebar:
    st.sidebar.title('🔍 Navigation Bar')
    selected = option_menu("",
                           ['📝 Home','🧬 Detect Multiple Disease','🧠 Tumor Detection'],
                           icons = ['none','DNA','x-ray'],
                           default_index=0)

print(selected)

if selected == '📝 Home':
    print('Home Executed !')

    st.title("**⚕️Your AI Doctor**")
    st.caption("This is a Demo App built on Streamlit Python Library for Disease Prediction")
    st.markdown(
        f""" 
        **Happy to Help 👨‍⚕️ It's an :blue[AI Doctor] build for you!** 
        
        It could Predict Disease like 
        
        :grey[{(list((items for index, items in enumerate(loaded_Disease_list))))}]

        
        -About 🎨 :rainbow[Frontend]
        Handled completely by **Streamlit** Python Library used by Data Scientist , AI & ML Engineers

        -About 🛠 :rainbow[Backend] 
        Uses **Random Forest Algorithm** and is Trained on a Dataset outsourced from *Kaggle*

        -About 📊 :violet[Dataset]
        Consists of about 131 different Combination of Symptoms along with it's labelled Disease

        """
        )

if selected == '🧬 Detect Multiple Disease':
    print('MDPM Executed !')
    st.title("🧬 Disease Prediction Model")
    st.caption("Select the Symptoms You Experince 🩺 ")

    cols = st.columns(5)  # Adjust number of columns as needed
    selected_symptoms = []

    for i, symptom in enumerate(loaded_symptoms_list):
        with cols[i % 5]:  # Rotate through columns
            if st.checkbox(symptom):
                selected_symptoms.append(symptom)

    st.write("")

    if st.button('**:grey[Get Results]**'):
        runModel(selected_symptoms)
    # selected_symptoms = st.multiselect("Choose Symptoms:", loaded_symptoms_list)
            
if selected == '🧠 Tumor Detection':
    print('Tumor Detection Executed !')

    st.title(f"🧠 Detect Your Brain Tumor")
    file = st.file_uploader("Pick a file")
    st.write(f"*Coming Soon...!* ")
        


    