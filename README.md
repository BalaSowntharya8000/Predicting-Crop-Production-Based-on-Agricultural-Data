# Predicting Crop Production Based on Agricultural Data

## Problem Statement
Agriculture is a key contributor to the economy, and accurately predicting crop production is essential for improving planning and decision-making. This project aims to develop a regression model that forecasts crop production (in tons) based on agricultural factors such as area harvested (in hectares), yield (in kg/ha), and the year, for various crops grown in a specific region.

### Project Overview
This regression-focused machine learning project forecasts crop production for different crops across regions and years. It supports data-driven decisions in agricultural planning and supply chain optimization.

Domain        : Agriculture
Goal          : Predict total crop production
Input Features: Area harvested(hectares), Yield(kg/ha), Year
Output        : Crop production (in tons)

By analyzing historical data for various crops in a specific region, the model supports better planning and decision-making in agriculture.

### Business Use Cases
**1) Food Security and Planning    :** Help governments and NGOs plan supplies
**2)Agricultural Policy Development:** Inform agricultural policies and subsidies
**3) Supply Chain Optimization     :** Optimize storage, transport, and market logistics
**4) Market Price Forecasting      :** Predict prices and selling time
**5) Precision Farming             :** Select ideal crops based on productivity
**6) Agro-Technology Solutions     :** Fuel intelligent agri applications

### Technologies Used
- **Python**        : Data handling and logic implementation
- **Pandas**        : Data manipulation and analysis
- **Plotly**        : Interactive visualizations
- **Streamlit**     : Web app framework for real-time data interaction (Dashboard and frontend web application)
- **Openpyxl**      : For reading/writing Excel data
- **NumPy**         : Numerical operations
- **Seaborn**       : Statistical data visualizations (Optional libraries for advanced visualization)
- **Matplotlib**    : Statis Visualization
- **Scikit-learn**  : Machine learning algorithms
- **VS Code**       : Development environment

### Set Up a Virtual Environment:
Set up a virtual environment to manage project dependencies separately from the global Python environment.

Ensure that have Python 3.10 or higher installed. 

**Create a virtual environment** inside the folder
python -m venv env

**Activate the environment**
     - On Windows:
         - .\env\Scripts\activate

### Installation Instructions
**Install the necessary packages**

To run this project, install the following libraries (via pip if not already installed): 
**pip install streamlit pandas plotly openpyxl numpy** 

OR (Individually)

To install the essential libraries for the project, run the following commands in the terminal:

- **pip install streamlit** - Streamlit library for building the web app
- **pip install pandas** - Pandas for data manipulation and handling dataframes
- **pip install plotly** - Plotly for creating interactive visualizations
- **pip install openpyxl** - Openpyxl to read/write Excel files
- **pip install numpy** - NumPy for numerical operations (useful for data manipulation)
- **pip install seaborn** - Statistical data visualizations (built on top of matplotlib)
- **pip install matplotlib** - Creating static plots like line graphs, bar charts, etc

### Code File Structure

- crop_production.ipynb               - Jupyter Notebook with data analysis and modeling
- crop_production_dashboard.py        - Streamlit app frontend
- machine_learning_concepts_explained - Folder with explanations
- Project Structure Guide             - Documentation on folder organization

### Data Sources
Access the datasets used in this project from the following links:

FAOSTAT_data - https://docs.google.com/spreadsheets/d/1rxG8FoHzFL_0S6FH6fljti1I3uPGDyihMj82_6HE4CU/edit?usp=sharing

### Dataset Explanation
1) Domain Code & Domain:
     Domain Code: Identifier for the data domain (e.g., QCL for crops and livestock).
     Domain     : The specific area of focus, such as "Crops and livestock products."
2) Area Code (M49) & Area:
     Area Code (M49): Numerical code representing countries or regions (e.g., "4" for Afghanistan).
     Area           : Name of the country or region (e.g., Afghanistan).
3) Element Code & Element:
     Element Code: Numeric code for the measured parameter (e.g., 5312 for area harvested).
     Element     : Description of the parameter (e.g., Area harvested, Yield, or Production).
4) Item Code (CPC) & Item:
     Item Code (CPC): Classification code for the crop/product (e.g., 1371 for Almonds, in shell).
     Item           : The name of the crop/product (e.g., Almonds, in shell).
5) Year Code & Year:
     Year Code: Numerical representation of the year.
     Year     : The calendar year for the recorded data.
6) Unit & Value:
     Unit : Unit of measurement (e.g., ha for hectares, kg/ha for yield, t(tons) for production).
     Value: The quantitative measure for the element and crop (e.g., 29203 hectares harvested).
7) Flag & Flag Description:
     Flag            : Coded indication of the data source or nature (e.g., "A").
     Flag Description: Explanation of the flag (e.g., Official figure).

This dataset enables analysis of agricultural patterns, including area harvested, crop yield, and production by region and year.

## Approach

### 1. Data Cleaning and Preprocessing
- Handled missing data and standardized column metrics.
- Filtered relevant columns for focused analysis.

### 2. Exploratory Data Analysis (EDA)

#### Analyze Crop Distribution
- **Crop Types**: Studied the distribution of the `Item` column to identify the most and least cultivated crops across regions.
- **Geographical Distribution**: Explored the `Area` column to understand which regions focus on specific crops or have high agricultural activity.

#### Temporal Analysis
- **Yearly Trends**: Analyzed the `Year` column to detect trends in Area harvested, Yield, and Production over time.
- **Growth Analysis**: Investigated if certain crops or regions showed increasing or decreasing trends in yield or production.

#### Environmental Relationships
- Although explicit environmental data was absent, inferred relationships between Area harvested and Yield to examine the impact of resource utilization on productivity.

#### Input-Output Relationships
- Studied correlations between Area harvested, Yield, and Production to understand the relationship between land usage and productivity.

#### Comparative Analysis
- **Across Crops**: Compared yields (`Yield`) of different crops (`Item`) to identify high-yield vs. low-yield crops.
- **Across Regions**: Compared production (`Production`) across regions (`Area`) to identify highly productive areas.

#### Productivity Analysis
- Examined variations in `Yield` to identify efficient crops and regions.
- Calculated productivity ratios: `Production / Area harvested` to verify and complement yield calculations.

#### Outliers and Anomalies
- Identified anomalies in `Yield` or `Production`, such as unusually high or low values.
- Correlated them with potential external factors like agricultural policies or environmental changes.

### 3. Task: Crop Production Prediction
- **Target Variable**: `Production` (in tons)
- **Use Case**: Predict total output of a specific crop for a given region and year.
  - Answers: *"What will the total production of a specific crop be for a given region and year?"*

### Feature Engineering

Feature engineering involved creating new insights from existing data to enhance analysis and modeling. 
Key steps included:
- Deriving productivity ratio (`Production / Area harvested`) to measure crop efficiency.
- Aggregating data by region, year, and crop to support meaningful comparisons.
- Preparing categorical fields (`Area`, `Year`, `Item`) for further analysis and modeling.

### Model Development

We developed regression models to predict crop production (in tons) based on year-wise data.

#### Features Used:
- **Input Features** : Year
- **Target Variable**: Production (tons)

#### Modeling Steps:
- Filtered data based on selected crop and region.
- Used `Year` as the input feature to train the model.
- Split data into training and testing sets for offline evaluation.
- Applied the following regression models:
  - **Linear Regression**
  - **Ridge Regression**
  - **Random Forest Regressor**
- Trained models using historical crop production data.
- Deployed a model selection dropdown in the Streamlit UI to let users choose between models.
- Generated future predictions for user-selected years.

#### Notes:
- **Linear Regression**: Best for simple linear trends.
- **Ridge Regression**: Linear model with regularization to prevent overfitting.
- **Random Forest**: Captures complex, non-linear patterns and is robust to noise.

## Model Evaluation

To evaluate the performance of each regression model, we used the following metrics:

### Evaluation Metrics:
- **Mean Absolute Error (MAE)**  
  Measures the average magnitude of errors in a set of predictions, without considering their direction.

- **Mean Squared Error (MSE)**  
  Measures the average squared difference between actual and predicted values. More sensitive to outliers than MAE.

- **Root Mean Squared Error (RMSE)**  
  Square root of MSE. Represents error in the same units as the target variable.

- **R² Score (Coefficient of Determination)**  
  Indicates how well the model explains the variance in the target variable. Ranges from 0 to 1 (higher is better).

### Insights:
- **Random Forest** is expected to perform well in capturing non-linear trends.
- **Ridge Regression** helps prevent overfitting in linear models through regularization.
- **Linear Regression** is best suited for simple linear trends.

### Streamlit App Features  
This application enables data-driven decision-making for agricultural planning through visualization, prediction, and actionable recommendations.

- **Home Page            :** Personalized greetings and project overview with visual context.  
- **Data Exploration     :** Interactive preview of cleaned data with filtering options and visual insights.  
- **Trend Analysis       :** Visualize production trends across years, crops, and countries using Plotly charts.  
- **Modeling & Prediction:** Train and evaluate Linear, Ridge, and Random Forest models to predict future crop yields with RMSE and R² metrics.  
- **Top Production per Country:** Discover top 5 crops by production volume for each selected country.  
- **Actionable Insights  :** Analyze yield and production efficiency, receive human-readable recommendations, and download CSV summaries for planning use.

  ![image](https://github.com/user-attachments/assets/5f36a10c-6099-455d-ad4f-858bf0b87e25)
  

###Author Bala Sowntharya Bala Subramanian
