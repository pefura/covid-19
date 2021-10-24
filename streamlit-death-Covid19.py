import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.write("""
# Machine Learning Covid-19 death Prediction App

This app predicts the risk of **Death during Covid-19 ** 

By Pefura-Yone et al
""")

st.header('User Input Parameters(please select patients features here)')


def user_input_features():
    age = st.selectbox('Age(0 - 17 years=1, 18 to 49 years=2, 50-64 years=3,  ≥65 years=4)', ('1','2', '3','4'))
    race = st.selectbox('Ethnicity (White=1, Black=2, Asian=3, Multiple/Other=4, Unknown=5)', ('1', '2', '3','4','5'))
    exposure = st.selectbox('Exposure (Yes=1, No/Unknown=2)', ('1', '2'))
    symptoms = st.selectbox('Symptoms(Asymptomatic=0, Symptomatic=1,Unknown=2)', ('0', '1','2'))
    hospitalisation = st.selectbox('Hospitalisation(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    intensive_care = st.selectbox('Intensive care(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    comorbidities= st.selectbox('Comorbidities(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    data = {'age': age,
            'race': race,
            'exposure': exposure,
            'symptoms': symptoms,
            'hospitalisation':hospitalisation,
            'intensive care':intensive_care,
            'comorbidities':comorbidities}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


st.subheader('User Input parameters confirmation')
st.write(df)

death_covid = pd.read_csv('https://raw.githubusercontent.com/pefura/covid-19/main/covid_19_death.csv', header=0)


# Slectionner les prédicteurs et la variable réponse

y = death_covid['death']
X = death_covid.drop('death', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import OneHotEncoder
encoder= OneHotEncoder()

# Reads in saved classification model
import pickle
LR = pickle.load(open('LR_covid.pkl', 'rb'))

# Prediction logistic regression
prediction = LR.predict(df)
prediction_proba = LR.predict_proba(df)
prediction_proba_percent_1 = prediction_proba * 100
proba = prediction_proba[:, 1]
prediction_proba_percent = proba * 100

st.subheader('Logistic regression Probability of death(%)')
st.write(prediction_proba_percent)

# Prediction catboost
catboost=pickle.load(open('catboost_covid.pkl', 'rb'))
prediction_1 = catboost.predict(df)
prediction_proba_1 = catboost.predict_proba(df)
prediction_proba_percent_1 = prediction_proba_1 * 100
proba_1 = prediction_proba_1[:, 1]
prediction_proba_percent_1 = proba_1 * 100

st.subheader('CatBoost Probability of death(%)')
st.write(prediction_proba_percent_1)
