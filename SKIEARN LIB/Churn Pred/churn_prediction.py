from webbrowser import open_new

import streamlit as st
import numpy as np
import pandas as pd
import  pickle
import sklearn

from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
ss = StandardScaler()

model = pickle.load(open('Churn_Pred.pkl', 'rb'))
df = pd.read_csv('7 churn.csv')


st.title('Logistic Regression for Churn Prediction')
gender = st.selectbox('Select Gender: ', options=['Male', 'Female'])
SeniorCitizen = st.selectbox('Select CitizenShip: ', options= ['Yes', 'No'])
Partner = st.selectbox("Do you have partner?", options=['Yes','No'])
Dependents	 = st.selectbox("Are you dependents on other?", options=['Yes','No'])
tenure = st.text_input("Enter Your tenure?")
PhoneService = st.selectbox("Do have phone service?",options=['Yes','No'])
MultipleLines = st.selectbox("Do you have mutlilines servics?", options=['Yes','No','no phone service'])
Contract = st.selectbox("Your Contracts?",options=['One year','Two year','Month-to_month'])
TotalCharges = st.text_input("Enter your Total charges?")

# Loading pkl
model = pickle.load(open('Churn_Pred.pkl', 'rb'))

# Using the function

def predictive(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges):
  data = {
      'gender' : [gender],
      'SeniorCitizen' : [SeniorCitizen],
      'Partner' : [Partner],
      'Dependents' : [Dependents],
      'tenure' : [tenure],
      'PhoneService' : [PhoneService],
      'MultipleLines' : [MultipleLines],
      'Contract' : [Contract],
      'TotalCharges' : [TotalCharges]
  }
  df1 = pd.DataFrame(data)
  categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract', 'SeniorCitizen', 'tenure']
  for cols in categorical_cols:
    df1[cols] = le.fit_transform(df1[cols])
# Applying Standarization
  df1 = ss.fit_transform(df1)
  result = model.predict(df1).reshape(1, -1)
  return result

# Tips : 
# For Churn Customers: 
churn_tips = {
   'Tips for Churn_Customers' : [
      "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering."
   ]
}

# For Not-Churn Customers: 

not_churn_tips = {
      'Tips for Not_Churn_Customers' : [
          "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers"
      ]
}

# Create DataFrames.

churn_tips = pd.DataFrame(churn_tips)
not_churn_tips = pd.DataFrame(not_churn_tips)


# Button
if st.button('Predict'):
    result = predictive(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges)
    if result == 1:
        st.title('Churn')
        st.write('10 tips for churn costumers.')
        st.dataframe(churn_tips, height= 300, width= 1000)

    else:
        st.title('Not Churn')
        st.write('10 tips for not churn costumers.')
        st.dataframe(not_churn_tips, height= 300, width= 1000)
