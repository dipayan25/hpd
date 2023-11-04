import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



data=pd.read_csv("heartproject//heart.csv")
feed=pd.read_csv("heartproject//feedback.csv")
model=pickle.load(open('heart.sav','rb'))
X,Y=data.drop(['output'],axis=1),data['output']

def diab_pred(input):
    cd=np.asarray(input)
    
    input_reshaped=cd.reshape(1,-1)
    d=model.predict(input_reshaped)
    if(d[0]==1):
        
        st.warning("So sad!!")
        st.error(" Sorry. You have a Heart Diseases, please refer to a  Doctor immediately!")
    
    else:
        
        return("You do not Have a Heart Diseases")
    
def create_account(username, password):
    with open("user_accounts.txt", "a") as file:
        file.write(f"{username}:{password}\n")

def login(username, password):
    with open("user_accounts.txt", "r") as file:
        for line in file:
            stored_username, stored_password = line.strip().split(":")
            if username == stored_username and password == stored_password:
                return True
    return False
def main(): 
    
    st.set_page_config(
    page_title="Heart Diseases Prediction",
    layout='wide',
    initial_sidebar_state="expanded"
    )
    st.title("Welcome to :red[HearDoPred !]")
    st.divider()
    st.write(":yellow[Made with Love by ~ Dipayan]")
    st.header(":blue[Heart Diseases Prediction]")
    st.subheader("YOUR BODY OUR MODEL")
    st.markdown("THANKS FOR CHOOSING &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    st.text("Check Beforehand....")
    i=0
    
    st.image("heartproject//1_6WGnPZ5lkiT2QgK-JP1DFw.png",width=400)
    nav=st.sidebar.radio("Navigation",['Home','Prediction','Feedback','AboutUs','Create Account','Login'])
    if nav=='Home':
        if st.checkbox("Show Table"):
            st.table(data)
        if st.checkbox('Instruction'):
            st.write('''\n
                     Data Dictionary
            1.age: age in years
            2.sex: sex (1 = male, 0 = female)
            3.	cp: chest pain type:
            •	0 = typical angina;
            •	1 = atypical angina;
            •	2 = non-anginal pain;
            •	3 = asymptomatic;
            4.	trestbps: resting blood pressure (in mm/Hg on admission to the hospital);
            5.	chol: serum cholestoral (in mg/dl);
            6.	fbs: fasting blood sugar > 120 mg/dl ? (1 = true, 0 = false);
            7.	restecg: resting electrocardiographic results:
            •	0 = normal;
            •	1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV);
            8.	thalach: maximum heart rate achieved;
            9.	exang: exercise induced angina (1 = yes; 0 = no);
            10.	oldpeak: ST depression induced by exercise relative to rest;
            11.	slope: the slope of the peak exercise ST segment:
            •	0 = upsloping;
            •	1 = flat;
            •	2 = downsloping;
            12.	ca: number of major vessels (0-3) colored by flourosopy;
            13.	thal: 0 = normal; 1 = fixed defect; 2 = reversable defect and the label''')
            

            
    if nav=='Prediction':
        st.header("Know your chances of  having a Heart diseases ")
        p=st.text_input("Enter the Age")
        st.divider()
        a=st.selectbox("Enter the Sex",["Male" , "Female"])
        if a=="Male":
            a=1
        elif a=="Female":
            a=0
        st.divider()
        b=st.selectbox("Enter the Chest Pain type",["typical angina","atypical angina","non-anginal pain","asymptomatic"])
        if b=="typical angina":
            b=0
        elif b=="atypical angina":
            b=1
        elif b=="non-anginal pain":
            b=2
        elif b=="asymptomatic":
            b=3
            
        st.divider()
        c=st.text_input("Enter the Blood Pressure")
        st.divider()
        d=st.text_input("Enter the Serum Cholestrol")
        st.divider()
        e=st.slider("Enter the Fasting Bllod Sugar (1:>120 mm/Hg , 0:<120 mm/dl)",0,1)
        
        st.divider()
        f=st.selectbox("Enter the ECG Results",["Normal","Wave Abnormality"])
        if f=="Normal":
            f=0
        elif f=="Wave Abnormality":
            f=1
        st.divider()
        g=st.text_input("Enter the Maximum Heart Rate")
        st.divider()
        h=st.text_input("Enter the Exercise Induced Angina")
        st.divider()
        i=st.text_input("Enter the ST Depression Induced by Exercise Relative to Rest ")
        st.divider()
        j=st.selectbox("Enter the The Slope of The Peak Exercise ST Segment ",["Unsloping","Flat","Downsloping"])
        if j=="Unsloping":
            j=0
        elif j=="Flat":
            j=1
        elif j=="Downsloping":
            j=2
        st.divider()
        k=st.slider("Enter the Number of Major Vessels Colored by Flourosopy",0,1,2)
        st.divider()
        l=st.slider("Enter the Thallium Stress Test",0,1,2)
        st.write("----")
        diag=''
    
        if st.button("Result",type='primary'):
            diag=diab_pred([p,a,b,c,d,e,f,g,h,i,j,k,l])
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')
            st.write(diag)
            
    if nav=='Feedback':
        
        #v=st.number_input("Rate Out of 10",0,10,step=2)
        v=st.selectbox('Rate Me out of:',["Good","Very Good","Excellent"])
        if st.button('Submit'):
            to_add={"RATE":[v]}
            to_add=pd.DataFrame(to_add)
            to_add.to_csv("heartproject//feedback.csv",mode='a',sep='\t',header=False,index=False)
            st.success("Submitted")
            
    if nav == 'AboutUs': 
        st.success("This project is Made by Dipayan Lodh")
    
    if nav== 'Create Account':
        st.header("Create an Account")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            create_account(new_username, new_password)
            st.success("Account created successfully!")
    if nav=='Login':
        # Login
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.success(f"Logged in as {username}")
                st.write(f"Welcome {username}")
            else:
                st.error("Login failed. Please check your username and password.")


    

if __name__=='__main__':
    main()
