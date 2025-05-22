import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: App Title and Description
st.title("Inventory Forecasting and Recommendation System")
st.write("""
This app forecasts future demand for alcohol types and brands at different bars, providing inventory recommendations based on forecasted demand and a safety stock factor.
""")

# Step 2: Upload Dataset
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file (Google Sheets export link compatible)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Step 3: Data Preprocessing
    st.header("Step 2: Data Preprocessing")
    data['Date Time Served'] = pd.to_datetime(data['Date Time Served'], errors='coerce')
    data = data.dropna(subset=['Date Time Served'])
    data['Date'] = data['Date Time Served'].dt.date

    # Display raw data
    if st.checkbox("Show raw data"):
        st.write(data.head())

    daily_consumption = data.groupby(['Date', 'Bar Name', 'Alcohol Type', 'Brand Name'])['Consumed (ml)'].sum().reset_index()

    # Step 4: Bar, Alcohol, and Brand Selection
    st.header("Step 3: Forecast Demand and Calculate Par Level")
    bar_name = st.selectbox("Select Bar", daily_consumption['Bar Name'].unique())
    alcohol_types = daily_consumption[daily_consumption['Bar Name'] == bar_name]['Alcohol Type'].unique()
    alcohol_type = st.selectbox("Select Alcohol Type", alcohol_types)
    brand_names = daily_consumption[(daily_consumption['Bar Name'] == bar_name) & 
                                    (daily_consumption['Alcohol Type'] == alcohol_type)]['Brand Name'].unique()
    brand_name = st.selectbox("Select Brand Name", brand_names)
    days_ahead = st.slider("Forecast Days Ahead", min_value=7, max_value=60, value=30)

    def forecast_demand(bar_name, alcohol_type, brand_name, days_ahead=30):
        df = daily_consumption[(daily_consumption['Bar Name'] == bar_name) &
                               (daily_consumption['Alcohol Type'] == alcohol_type) &
                               (daily_consumption['Brand Name'] == brand_name)]

        if df.empty:
            st.warning(f"No data available for {brand_name} ({alcohol_type}) at {bar_name}")
            return None

        prophet_df = df.rename(columns={'Date': 'ds', 'Consumed (ml)': 'y'})[['ds', 'y']]
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def calculate_par_level(forecast_df, safety_factor=1.5):
        mean_forecast = forecast_df['yhat'].mean()
        std_forecast = forecast_df['yhat'].std()
        par_level = mean_forecast + safety_factor * std_forecast
        return max(round(par_level, 2), 0)

    # Forecast demand and calculate par level
    forecast_df = forecast_demand(bar_name, alcohol_type, brand_name, days_ahead)
    if forecast_df is not None:
        st.write(f"### Forecast for {brand_name} ({alcohol_type}) at {bar_name}")
        st.line_chart(forecast_df[['ds', 'yhat']].set_index('ds'))

        par_level = calculate_par_level(forecast_df)
        st.write(f"**Recommended Par Level:** {par_level} ml")

        # Plot forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
        ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='lightblue', alpha=0.5)
        ax.set_title(f'Forecast for {brand_name} ({alcohol_type}) at {bar_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumed (ml)')
        ax.legend()
        st.pyplot(fig)

    # Step 5: Generate Recommendations Table
    st.header("Step 4: Recommendations for Bar")
    def generate_recommendations(bar_name, days_ahead=30):
        alcohol_types = daily_consumption[daily_consumption['Bar Name'] == bar_name]['Alcohol Type'].unique()
        recommendations = []

        for alc in alcohol_types:
            brands = daily_consumption[(daily_consumption['Bar Name'] == bar_name) & 
                                       (daily_consumption['Alcohol Type'] == alc)]['Brand Name'].unique()
            for brand in brands:
                forecast_df = forecast_demand(bar_name, alc, brand, days_ahead)
                if forecast_df is not None:
                    par = calculate_par_level(forecast_df)
                    recommendations.append({'Alcohol Type': alc, 'Brand Name': brand, 'Recommended Par Level (ml)': par})

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values(by='Recommended Par Level (ml)', ascending=False).reset_index(drop=True)
        return rec_df

    recommendations_df = generate_recommendations(bar_name, days_ahead)
    st.write(f"### Recommendations for {bar_name}")
    st.dataframe(recommendations_df)
else:
    st.info("Please upload a dataset to begin.")
