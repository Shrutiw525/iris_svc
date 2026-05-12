#streamlit for svc
import streamlit as st
st.title("Support Vector Classifier")
st.write("This is a support vector classifier model to predict the class of a flower based on its features.")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("iris_svc.csv")
st.write("The dataset:")
st.dataframe(df.head())
q1=df["SepalWidthCm"].quantile(0.25)
q3=df["SepalWidthCm"].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
df["SepalWidthCm"].clip(lower_bound,upper_bound,inplace=True)
if "Id" in df.columns:
    df=df.drop("Id", axis=1)

x=df.drop("Species",axis=1)
y=df["Species"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.svm import SVC
model=SVC(kernel="rbf",C=1.0,gamma="scale",degree=3,probability=True)
model.fit(x_train,y_train)
#predicting by taking user input
st.write("Enter the features of the flower to predict its class:")
sepal_length=st.number_input("Sepal length (cm)", min_value=1.0, max_value=10.0, value=5.0)
sepal_width=st.number_input("Sepal width (cm)", min_value=1.0, max_value=10.0, value=3.0)
petal_length=st.number_input("Petal length (cm)", min_value=1.0, max_value=10.0, value=1.0)
petal_width=st.number_input("Petal width (cm)", min_value=1.0, max_value=10.0, value=1.0)
input_data=pd.DataFrame(
    [[sepal_length,sepal_width,petal_length,petal_width]],
    columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
)
input_data=input_data.reindex(columns=x.columns)
if st.button("Predict"):
    prediction=model.predict(input_data)
    st.write("The predicted class of the flower is:", prediction[0])
    