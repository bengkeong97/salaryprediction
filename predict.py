from json import load
import streamlit as st
import pickle
import numpy as np 


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some infomation to predict the salary""")


# tuple with choices

    countries =(
        "United States",       
        "India",                 
        "United Kingdom",        
        "Germany",               
        "Other",                 
        "Canada",               
        "Brazil"                 
        "France",               
        "Spain",                  
        "Australia",              
        "Netherlands",            
        "Poland",                 
        "Italy",                  
        "Russian Federation",    
        "Sweden",                
        "Turkey",                 
        "Israel",                 
        "Pakistan",               
        "Switzerland",            
        "Mexico",               
        "Ireland",                
        "Norway",                 
        "Ukraine",                
        "Romania"             
        "South Africa",           
        "Czech Republic",        
        "Austria",               
        "Belgium",               
        "Iran",                   
        "Portugal",             
        "Denmark",               
        "Finland",                
        "Argentina",              
        "Hungary",                
        "New Zealand",            
        "Greece",                 
        "Japan",                  
        "Bulgaria",               
        "Bangladesh",             
        "Colombia",               
        "Serbia",                
        "Indonesia",             
        "Philippines",            
        "Nigeria",                
        "Singapore",              
        "Lithuania",              
        "Sri Lanka",              
        "Chile",               
        "China",                 
        "Viet Nam",               
        "Croatia",               
        "Malaysia",               
        "Estonia",                
        "Slovenia",               


    )

    education =(
        'Bachelor’s degree', 
        'Master’s degree', 
        'Less than a Bachelors',
        'Post grad'
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Year of experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok: 
        X = np.array([[country, education , experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")







