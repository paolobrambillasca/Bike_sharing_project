import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib 
from scipy.stats import norm
import json
import shap
import plotly.graph_objects as go


def configure_theme():
    # Set the page configuration
    st.set_page_config(
        page_title="Bike Sharing Demand Prediction",  # Title of the page
        page_icon="🚲",            # Icon to be displayed in the browser tab
        layout="centered",                   # Layout of the app (wide or centered)
        initial_sidebar_state="expanded" # Initial state of the sidebar (expanded or collapsed)
    )

configure_theme()

#Load the dataset
@st.cache_data()
def load_data():
    data = pd.read_csv('bike-sharing_hourly.csv')
    return data.drop(columns=['instant'])
@st.cache_data()
# Load the trained model
def load_model():
    model = joblib.load('xgb_regressor.pkl')  
    return model


# Load the scores
with open('cv_scores.json', 'r') as f:
    cv_scores = json.load(f)

cv_mean = cv_scores['cv_mean']
cv_std = cv_scores['cv_std']

# Now, cv_mean and cv_std can be used within your Streamlit app

df = load_data()
model = load_model()

#Sidebar
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', ['Home', 'Data Exploration', 'Model Prediction'])

if options == 'Home':
    st.title('Bike Sharing Demand Prediction 🚲')
    st.write('''
             This dashboard provides insights into the bike sharing dataset and allows for interactive data exploration
             and prediction of bike rental demand based on various features. Explore the dataset, understand the
             relationships between different features, and predict demand using the machine learning model developed.
             ''')
    st.image("https://images.ctfassets.net/p6ae3zqfb1e3/2l8ZrEyDeCq8F6oBvda0WP/f92c98aa4dd42818ab413f193945777b/CaBi_Homepage_Hero_2x.png?w=2500&q=60&fm=webp")

elif options == 'Data Exploration':
    st.title('Data Exploration')
    
    # Features vs count with customization
    st.markdown('**Feature against Count**')
    plot_type = st.selectbox("Select plot type:", ['Line', 'Bar'])
    plot_color = st.color_picker("Pick a color for the plot:", '#FF0000')
    default_feature = 'season'  
    feature = st.selectbox('Select feature to plot against count:', df.columns.drop(['cnt', 'registered', 'casual']), index=df.columns.get_loc(default_feature))
    fig, ax = plt.subplots()

    if plot_type == 'Line':
        sns.lineplot(x=df[feature], y=df['cnt'], ax=ax, color=plot_color)
    elif plot_type == 'Bar':
        sns.barplot(x=df[feature], y=df['cnt'], ax=ax, color=plot_color)
    plt.xticks(rotation=45)
    st.pyplot(fig)


    # Corr. Matrix 


    st.markdown('**Correlation Matrix:**')
    color_palette = st.selectbox("Select Color Palette for Correlation Matrix:", ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    numeric_cols = df.select_dtypes(include=np.number).columns
    features_selected = st.multiselect('Select features for the correlation matrix:', numeric_cols, default=list(numeric_cols))
    if features_selected:
        corr_matrix = df[features_selected].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap=color_palette, center=0, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)




    # Summary stats, box plot, and histogram for numeric columns with customization
    st.markdown('**Summary Statistics, Box Plot, and Histogram:**')
    numeric_cols = df.select_dtypes(include=np.number).columns  
    selected_stats_features = st.multiselect('Select features for summary statistics:', numeric_cols, default=list(numeric_cols[:3]))

    boxplot_color = st.color_picker("Pick a color for the box plot:", '#FFFF00')
    hist_color = st.color_picker("Pick a color for the histogram:", '#FF0000')
    show_kde = st.checkbox("Show KDE on histogram", value=True)

    for feature in selected_stats_features:
        st.write(f'Summary statistics for {feature}:')
        st.write(df[feature].describe())
    
        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(y=df[feature], ax=ax, color=boxplot_color)
        st.pyplot(fig)
    
        # Histogram
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=show_kde, ax=ax, color=hist_color)
        st.pyplot(fig)

    
elif options == 'Model Prediction':
    st.markdown('**Model Prediction**')
    # Feature Importances
    feature_names = ['Season', 'Year', 'Hour', 'Holiday', 'Weekday', 'Weather Situation', 'Feels Like Temperature', 'Humidity', 'Windspeed']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    sns.barplot(x=np.array(feature_names)[indices], y=importances[indices], palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Input fields to collect features
    season = st.selectbox('Season', options=[1, 2, 3, 4], format_func=lambda x: {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}.get(x))
    yr = st.selectbox('Year', options=[0, 1], format_func=lambda x: {0: '2011', 1: '2012'}.get(x))
    hr = st.slider('Hour of the day', 0, 23)
    holiday = st.selectbox('Holiday', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}.get(x))  
    weekday = st.selectbox('Day of the Week', options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}.get(x))
    weathersit = st.selectbox('Weather Situation', options=[1, 2, 3, 4], format_func=lambda x: {1: 'Clear', 2: 'Mist + Cloudy', 3: 'Light Snow', 4: 'Heavy Rain'}.get(x))
    atemp = st.slider('Feels Like Temperature', float(df.atemp.min()), float(df.atemp.max()), float(df.atemp.mean()))
    hum = st.slider('Humidity', float(df.hum.min()), float(df.hum.max()), float(df.hum.mean()))
    windspeed = st.slider('Windspeed', float(df.windspeed.min()), float(df.windspeed.max()), float(df.windspeed.mean()))

    if st.button('Predict Count'):
        input_features = np.array([[season, yr, hr, holiday, weekday, weathersit, atemp, hum, windspeed]])
        prediction = model.predict(input_features)[0]  # Assuming model.predict returns an array

        # Calculating a 95% confidence interval around the prediction
        ci_lower = prediction - 1.96 * cv_std
        ci_upper = prediction + 1.96 * cv_std

        st.write(f'Predicted count of bike rentals: {prediction:.2f}')
        st.write(f'95% confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}]')

        # Plotting the normal distribution for the confidence interval
        fig, ax = plt.subplots()
        x = np.linspace(ci_lower, ci_upper, 1000)
        y = norm.pdf(x, prediction, cv_std)
        ax.plot(x, y, 'r-', lw=2)
        ax.fill_between(x, y, 0, alpha=0.3, color='red')
        ax.set_title('Normal Distribution around the Prediction')
        st.pyplot(fig)

        # SHAP explanation and Waterfall chart adjustment
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(input_features)
        shap_values = shap_values[0]  # Assuming it's a single prediction

        # Making sure to use the correct expected_value if it's an array
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):  # SHAP can return a numpy array for base_value
            base_value = base_value[0]  # Assuming we're dealing with a single output model

        # Preparing waterfall chart data
        total_effect = np.sum(shap_values)
        predicted_value = base_value + total_effect

        formatted_shap_values = [f"{value:.2f}" for value in shap_values]
        formatted_base_value = f"{base_value:.2f}"
        formatted_predicted_value = f"{predicted_value:.2f}"

        fig = go.Figure(go.Waterfall(
            name="Model Prediction",
            orientation="v",
            measure=["absolute", *["relative"] * len(feature_names), "total"],
            x=["Base Value", *feature_names, "Predicted Value"],
            textposition="outside",
            text=[formatted_base_value, *formatted_shap_values, formatted_predicted_value],
            y=[base_value, *shap_values, predicted_value],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(title="Waterfall Chart of SHAP Values")
        st.plotly_chart(fig, use_container_width=True)
