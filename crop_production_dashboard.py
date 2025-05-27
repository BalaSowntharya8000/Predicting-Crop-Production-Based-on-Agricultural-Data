#Import Required Libraries
import streamlit as st                     #For building the web app interface
import pandas as pd                        #For data manipulation
import numpy as np                         #For numerical operations and array handling
import matplotlib.pyplot as plt            #For visualizations
import seaborn as sns                      #For advanced visualizations
from datetime import datetime              #To fetch the current time for greeting
import plotly.express as px                #For interactive charts

#Commands Used
#import streamlit as st           : Imports Streamlit library for UI development
#import pandas as pd              : Imports pandas for data manipulation and analysis
#import matplotlib.pyplot as plt  : Imports matplotlib for basic plotting
#import seaborn as sns            : Imports seaborn for statistical plots
#from datetime import datetime    : Imports datetime to fetch the current time

#Summary: This block sets up all the essential libraries needed for data processing, visualization, and time-based greetings.

#📊Data Loading

#Define a cached function to load data
#Caches the output of the data-loading function to avoid re-reading the file on every rerun, improving app speed and performance
@st.cache_data    
def load_data():
    file_path = r"C:\Users\Bala Sowntharya\Documents\Crop_Production_Prediction_Project\FAOSTAT_data.xlsx"  #Local Excel file path
    df = pd.read_excel(file_path)  #Load Excel data into pandas DataFrame
    return df

df = load_data()  #Call the cached data loading function

#Commands Used
# @st.cache_data          : Caches the function output to speed up app performance
# pd.read_excel(file_path): Reads the Excel file into a DataFrame
# r"..."                  : Raw string literal to handle Windows file paths correctly

#Summary: Loads the historical crop production dataset using a cached function for faster performance on repeated runs


#🧭 Sidebar Navigation Setup
#🧭 Sidebar Navigation Setup
st.sidebar.title("📌 Navigation")  # Sidebar title for better UI

# Define all pages
pages = ["🏠 Home", 
         "🔍 Data Exploration", 
         "📈 Trend Analysis", 
         "🧠 Modeling & Prediction", 
         "📊 Top Production per Country",
         "🌱 Actionable Insights"]

# Sidebar radio button with default selection set to "📊 Top Production per Country"
page = st.sidebar.radio("Go to", pages, 
                        index=4,
                        help="Use this menu to navigate between different analysis and prediction pages.")
 #Radio button for page navigation

#Commands Used
#st.sidebar.title(...)  : Sets a title for the sidebar
#st.sidebar.radio(...)  : Creates a radio button menu for navigation

#Summary: Adds a sidebar with navigation options to switch between different pages of the app


#🏠 Home Page Setup (Main Interface)
if page == "🏠 Home":                                #Home page code
    #⏰ Dynamic Greeting Based on Time
    current_hour = datetime.now().hour
    if current_hour < 12:
        st.write("🌞 Good Morning!")
    elif 12 <= current_hour < 18:
        st.write("☀️ Good Afternoon!")
    else:
        st.write("🌙 Good Evening!")

#🎯 Title
    st.title("🌾 Predicting Crop Production")   #Main title for home page     

    #🧾 Short Project Intro
    st.markdown("""
    Welcome to the **Predicting Crop Production Dashboard**.

    This dashboard helps analyze crop production trends and predict future yields using historical FAOSTAT data.

    Navigate through the pages to explore visualizations, train models, and generate insights!
    """)

    #📖 How to Use This Dashboard
    with st.expander("📖 How to Use This Dashboard"):
         st.markdown("""
         Use the **sidebar** to navigate across different pages            
          - 📊 **Data Exploration**          : Dive into raw data and trends using charts  
          - 📈 **Trend Analysis**            : Understand production patterns by region, crop, and year            
          - 🔧 **Modeling & Prediction**     : Train models to predict future production  
          - 📊 **Top Production per Country**: Analyze which crops dominate production in each country                      
          - 💡 **Actionable Insights**       : Interpret results and key factors affecting yield
    """)
#This page includes greetings, an introduction, and usage instructions based on time and dashboard goals.

#Commands Used
#Streamlit functions
#st.title(...)             : Adds a title to the main page
#st.write(...)             : Displays dynamic text (greetings)
#st.markdown(...)          : Renders formatted text
#st.subheader(...)         : Adds a subheading

#DateTime Functions
#datetime.now().hour       : Gets the current hour of the day

#Summary: This is the landing page
#It shows a time-based greeting, project intro, and a how-to-use guide
  #It's only shown when "Home" is selected


#📊 Data Exploration Page
elif page == "🔍 Data Exploration":         #Switch to 'Data Exploration' page

    st.title("📊 Data Exploration")         #Adds page title to the Streamlit app

    #📂 Data Preview
    st.subheader("📄 Dataset Preview")      #Adds subheading for dataset preview

    #Standardize column names
    df.columns = df.columns.str.strip().str.lower()  #Cleans column names by stripping spaces and converting to lowercase
    df.drop_duplicates(inplace=True)                 #Removes duplicate rows permanently

    st.dataframe(df.head())                          #Shows first 5 rows of unique, cleaned dataset

    # 🔍 Filter Section
    st.subheader("🔍 Filter the Data")              #Adds subheading for filtering section

    #Ensure correct numeric format for year and value
    df['year'] = pd.to_numeric(df['year'], errors='coerce')    #Converts 'year' column to numeric, replaces invalid parsing with NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')  #Converts 'value' column to numeric, replaces invalid parsing with NaN

    #Remove non-ASCII characters from crop names
    df['item'] = df['item'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))  #Special glyphs in crop names

    # 🔽 Dropdown filters
    crop_filter = st.selectbox("Select Crop", options=["All"] + sorted(df['item'].dropna().unique().tolist()))  
    #Creates a dropdown to filter by crop

    country_filter = st.selectbox("Select Country", options=["All"] + sorted(df['area'].dropna().unique().tolist()))  
    #Creates a dropdown to filter by country

    year_filter = st.selectbox("Select Year", options=["All"] + sorted(df['year'].dropna().unique().astype(str).tolist()))  
    #Creates a dropdown to filter by year

    #Apply Filters
    filtered_df = df.copy()  #Create a separate copy for filtering

    if crop_filter != "All":  
        filtered_df = filtered_df[filtered_df['item'] == crop_filter]       #Filter by crop

    if country_filter != "All":  
        filtered_df = filtered_df[filtered_df['area'] == country_filter]    #Filter by country

    if year_filter != "All":  
        filtered_df = filtered_df[filtered_df['year'] == int(year_filter)]  #Filter by year


    filtered_df.drop_duplicates(inplace=True)                               #Remove duplicates again from filtered data

    # 📋 Display Filtered Data
    st.subheader("📋 Filtered Data")  #Subheading for filtered data section
    st.dataframe(filtered_df)         #Show filtered results

    # 💾 CSV Download Option
    st.download_button("⬇️ Download Filtered Data as CSV", filtered_df.to_csv(index=False), file_name="filtered_crop_data.csv")  
    #Download button to export filtered data

    # 📈 Visualization
    st.subheader("📈 Crop Production Trend")   #Subheading for the graph section

    if not filtered_df.empty:                   #Proceed only if data exists
        import matplotlib.pyplot as plt         #Import for plotting
        import seaborn as sns                   #Import for styled plots

        #Fix font issues for missing glyphs
        plt.rcParams['font.family'] = 'DejaVu Sans'  #Use glyph-safe font to avoid warnings

        #Set darkgrid style for dark theme compatibility
        sns.set_style("darkgrid")  

        #Optional Top N Crop Selector
        #Conditional Check for Crop Filter
        if crop_filter == "All":                          #Checks if the user has selected "All" crops in the filter
            top_n = st.slider("Select Top N Crops by Average Production", min_value=1, max_value=15, value=5)  
            #Slider to choose how many top crops to show
            top_crops = (                                 
                filtered_df.groupby("item")["value"]      #Group filtered data by 'item' (crop)
                .mean()                                   #Calculate mean production value for each crop
                .sort_values(ascending=False)             #Sort crops by descending average production
                .head(top_n)                              #Select the top N crops based on slider input
                .index.tolist()                           #Convert the crop names to a list
            )
            plot_df = filtered_df[filtered_df["item"].isin(top_crops)]  #Filter top N crops
        else:
            plot_df = filtered_df.copy()

        plot_df = plot_df.sort_values(by="year")                    #Sort data for clean plotting

        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')  #White background for better contrast

        #Plotting logic
        if crop_filter == "All":
            sns.lineplot(data=plot_df, x='year', y='value', hue='item', marker='o', palette='tab10', ax=ax)  #Multi-crop plot
        else:
            sns.lineplot(data=plot_df, x='year', y='value', marker='o', color='blue', ax=ax)  #Single-crop plot

        ax.set_title("Crop Production Over Time", color='black')  #Chart title
        ax.set_ylabel("Production (in tonnes)", color='black')    #Y-axis label
        ax.set_xlabel("Year", color='black')                      #X-axis label
        ax.tick_params(colors='black')                            #Axis tick color
        fig.patch.set_facecolor('white')                          #Full figure white background

        #Fix Year as integer on X-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        #Optional: Round Y-axis values (e.g., 12,000 instead of 12000.53)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

        #Legend fixes for readability
        if crop_filter == "All":
            ax.legend(title="Crop Type", bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)  #Move legend below the graph
        else:
            ax.legend().set_visible(False)  #Hide legend for single crop

        st.pyplot(fig)  #Render the plot in Streamlit
    else:
        st.warning("No data to display. Please adjust your filters.")  #Message for empty filtered result

#Data Exploration Page – Description
#This page allows users to explore crop production trends interactively
#It allows users to preview, filter, and visualize crop production data by country, crop, and year

#Key Features Implemented

#Data Cleaning
  #Standardized column names and removed duplicates for consistency
#Interactive Filters
  #Users can filter the dataset by Crop, Country, and Year using dropdowns
#Filtered Data Display
  #Shows the filtered dataset in a clean table format below the filters
#Trend Visualization
  #Generates a line chart showing crop production trends over the years
  #Handles both single crop and multiple crops with dynamic legends
#Axis Formatting Fixes
  #Ensured Year is displayed as an integer (e.g., 2019 instead of 2019.0)
  #Optionally rounded Production to whole numbers with commas (e.g., 12,000)
#Legend Visibility & Positioning
  #Hid the legend when a single crop is selected
  #(Optional) Could reposition the legend or label the crop below the graph if it clutters

#Code Refinement
#Column Formatting           Standardized column names by stripping whitespaces and converting to lowercase
#Data Type Conversion        Converted 'year' and 'value' columns to numeric using pd.to_numeric()
#Filter Controls             Added dropdown filters for Crop, Country, and Year
#Data Duplication            Removed duplicate rows after filtering for clean data display
#Empty Dataset Handling      Added warning when no data is available after filters
#Plot Enhancements           Sorted data before plotting and applied seaborn styling
#Axis Formatting             Cleaned year/value axes by converting decimals to integers
#Legend Overlap Issue        Moved crop type labels below the graph to avoid overlap
#Matplotlib Styling          Set white background, dark grid theme, and improved label visibility
#Glyph Errors (Advanced)     Resolved missing glyphs warning by using 'DejaVu Sans' font to support special characters
  
#Commands Used
#Pandas
#df.copy()	                        Creates a copy of the DataFrame
#df['item'], df['area'], df['year']	Access specific columns
#df.columns.str.strip().str.lower()	Cleans column names
#df.drop_duplicates()	              Removes duplicate rows
#pd.to_numeric()	                  Converts strings to numbers (handles errors)
#df.dropna()	                      Removes missing values
#df.sort_values(by='year')	        Sorts rows by year
#.unique()	                        Extracts unique values
#.tolist()	                        Converts Series to list
#.astype(str)	                      Converts values to string for display
#df[df['col'] == val]	              Filters data by condition
#errors='coerce'                    Converts invalid values to NaN during type conversion
#groupby("item")["value"]           Groups data by crop type for analysis

#Streamlit
#st.title()	                        Adds a page title
#st.subheader()	                    Adds a subheading
#st.dataframe()	                    Displays a data table
#st.selectbox()	                    Creates dropdown menus
#st.pyplot()	                      Displays matplotlib plots
#st.warning()	                      Shows warning messages

#Seaborn
#sns.set_style()                    Sets the plot style
#sns.lineplot()	                    Creates a line plot with hue (categorical coloring)

#Matplotlib
#plt.subplots()	                    Initializes a plot figure and axes
#ax.set_title()	                    Sets the plot title
#ax.set_xlabel()                    Sets the x-axis label
#ax.set_ylabel()	                  Sets the y-axis label
#ax.tick_params()                   Customizes tick appearance (like color)
#fig.patch.set_facecolor()          Sets the background color of the plot
#ax.legend()                        Displays or hides the plot legend

#Python Core
#if not filtered_df.empty	Checks if the filtered DataFrame is not empty
#if crop_filter != "All"	Conditional logic for filtering

#📈 Trend Analysis
elif page == "📈 Trend Analysis":
    # 📌 Page Title and Instructions
    st.title("📈 Trend Analysis: Crop Production by Region, Crop Type & Year")
    st.markdown("""
    This page helps you analyze historical crop production trends. 
    You can explore trends across different regions, crop types, and years to inform future planning.
    """)

    #Step 1: Efficiently Load & Filter Data
    #Load and filter data (cached)
    @st.cache_data                         
    def load_filtered_production_data():
        df = pd.read_excel(r"C:\Users\Bala Sowntharya\Documents\Crop_Production_Prediction_Project\FAOSTAT_data.xlsx")
        df = df[df['Element'] == 'Production']  # Keep only 'Production' rows
        df = df.dropna(subset=['Area', 'Item', 'Year', 'Value'])  # Remove missing values
        return df

    df = load_filtered_production_data() #Returns DataFrame for visualization

    #Step 2: Year-wise Total Crop Production Line Chart
    st.subheader("📅 Total Crop Production Over the Years")
    #Group data by Year and sum total production values
    yearly = df.groupby("Year")["Value"].sum().reset_index()
    #Create a line chart using Plotly to show total production over years
    fig_yearly = px.line(             
        yearly,                       #DataFrame with Year and summed Value
        x="Year",                     #X-axis : Year
        y="Value",                    #Y-axis : Total Production
        title="Year-wise Total Crop Production (tons)", #Chart Title
        labels={"Value": "Total Production (tons)", "Year": "Year"}, #Axis Labels
        markers=True                  #Add data point markers
    )
    #Display the Plotly chart in Streamlit
    st.plotly_chart(fig_yearly, use_container_width=True)

    #Step 3: Region-wise Production Over Time
    st.subheader("🌍 Region-wise Crop Production Trend")
    #Get unique region names from the 'Area' column
    region_options = df["Area"].unique()
    #Dropdown to let user select a region
    selected_region = st.selectbox("Select a Region", region_options)
    #Filter data for the selected region and group by Year, then sum production
    region_df = df[df["Area"] == selected_region].groupby(["Year"])["Value"].sum().reset_index()
    
    #Create a line chart for region-wise crop production trend
    fig_region = px.line(      
        region_df,               #Filtered DataFrame
        x="Year",                #X-axis: Year
        y="Value",               #Y-axis: Production
        title=f"Total Production in {selected_region} Over Time",  #Dynamic chart title
        labels={"Value": "Production (tons)", "Year": "Year"},     #Axis Labels
        markers=True             #Show data point markers
    )
    #Display the chart in Streamlit
    st.plotly_chart(fig_region, use_container_width=True)

    #Step 4: Crop-wise Production Trend
    st.subheader("🌾 Crop-wise Production Trend")
    #Get top 20 most common crops using value_counts and extract their names
    crop_options = df["Item"].value_counts().head(20).index.tolist()  #Most common crops
    #Dropdown to let user select one crop from the top 20
    selected_crop = st.selectbox("Select a Crop", crop_options)
    #Filter data for the selected crop and group by Year, then sum production
    crop_df = df[df["Item"] == selected_crop].groupby(["Year"])["Value"].sum().reset_index()
    
    #Create a line chart for crop-wise production trend
    fig_crop = px.line(
        crop_df,            #Filtered DataFrame for selected crop
        x="Year",           #X-axis: Year
        y="Value",          #Y-axis: Production value
        title=f"Production Trend for {selected_crop}",          #Dynamic chart title
        labels={"Value": "Production (tons)", "Year": "Year"},  #Axis Labels
        markers=True        #Show data point markers
    )
    #Display the chart in Streamlit
    st.plotly_chart(fig_crop, use_container_width=True)

    #Optional: Combined Crop vs Region Filter
    st.subheader("📊 Compare Crop Production by Region")
    #Dropdown to select a crop for regional comparison (uses key to avoid conflicts)
    selected_crop2 = st.selectbox("Select a Crop to Compare Across Regions", crop_options, key="compare_crop")
    #"Select a Crop to Compare Across Regions",  #Label for the dropdown
    #crop_options,                               #Use the same top 20 crops list
    #key="compare_crop"                          #Unique key to differentiate from other dropdowns

    #Filter data for selected crop, group by Year and Region, sum production  
    compare_df = df[df["Item"] == selected_crop2].groupby(["Year", "Area"])["Value"].sum().reset_index()
    
    #Create a multi-line plot showing region-wise production trend for selected crop
    fig_compare = px.line(
        compare_df,             #Filtered DataFrame
        x="Year",               #X-axis: Year
        y="Value",              #Y-axis: Production value
        color="Area",           #Different color for each region
        title=f"{selected_crop2} Production Across Regions Over Years",          #Dynamic chart title
        labels={"Value": "Production (tons)", "Year": "Year", "Area": "Region"}  #Axis Labels
    )
    #Display the comparison chart in Streamlit
    st.plotly_chart(fig_compare, use_container_width=True)

#Trend Analysis Page – Description
#This page helps users explore and visualize crop production trends over time
#It supports viewing data year-wise, region-wise, and crop-wise, including comparative analysis of crop production across different regions.

#Key Features Implemented
#📅 Year-wise Trend
  #Aggregates total production for each year and displays a line chart
  #Helps visualize overall production changes over time

#🌍 Region-wise Filter
  #Users can select a region (country) to view its yearly crop production trend
  #Interactive line chart updates based on region selected

#🌾 Crop-wise Filter
  #Filters production trends by individual crop items
  #Displays historical trend of selected crop over the years

#🧮 Comparative Crop Analysis Across Regions
  #Users can compare a single crop’s performance across multiple regions
  #Multi-line chart shows each region’s data with distinct color legends

#📊 Interactive Charts with Plotly
  #Hover tooltips, zoom, and legend interactivity enabled
  #Graphs auto-resize to container width for better layout on all screens

#Code Refinement
#Data Caching	         Used @st.cache_data to improve performance and avoid reloading data repeatedly
#NaN Handling	         Dropped missing data rows to ensure clean analysis
#Standard Filtering	   Enabled dropdown filters for Area (Region) and Item (Crop)
#Grouping	             Used groupby() to prepare data by Year and Region/Crop
#Sorted Data	         Grouped and sorted data to improve visual flow in line charts
#Responsive Plots	     Set use_container_width=True for consistent UI scaling
#Optimized Dropdowns	 Limited crop options to top 20 frequent crops to simplify selection
#Dynamic Chart Titles	 Chart headings update based on selection for better clarity

#Commands Used
#Pandas
#pd.read_excel()	                  Load Excel file into DataFrame
#df.dropna()	                      Remove rows with missing values
#df[df["Element"] == "Production"]	Filter only production rows
#groupby(["Year"])["Value"].sum()	  Aggregate production over time
#.unique()	                        Get unique values for filters
#.head(20)	                        Get top 20 crops for dropdown
#.reset_index()	                    Flatten grouped data

#Plotly Express
#px.line()	                        Create line chart
#color="Area"	                      Differentiate regions in multi-line chart
#markers=True	                      Show markers on data points
#labels={}	                        Add axis labels
#use_container_width=True	          Full-width chart rendering

#Streamlit
#st.title()	                        Set page title
#st.subheader()	                    Section headers
#st.markdown()	                    Show markdown descriptions
##st.plotly_chart()	                Display Plotly charts
#@st.cache_data	                    Cache data to speed up loading

# 🌱 Modeling & Prediction Page
elif page == "🧠 Modeling & Prediction":

    #Step 1: 📚 Import Required Libraries
    import io                                           #For temporary memory file (used for download)
    import pandas as pd                                 #For data manipulation
    import numpy as np                                  #For numerical operations
    from sklearn.linear_model import LinearRegression   #Linear regression model
    from sklearn.linear_model import Ridge              #Ridge regression model (L2 regularization)
    from sklearn.ensemble import RandomForestRegressor  #Tree-based ensemble model
    from sklearn.metrics import mean_squared_error, mean_absolute_error  #Evaluation metrics
    import matplotlib.pyplot as plt                     #For visualizing data
    import streamlit as st                              #For web app interface

    #Step 2: 🧾 Title for the Page
    st.title("📈 Crop Production Modeling & Prediction")

    #Step 3: Data Preparation Section
    st.subheader("Data Preparation")                    #Section header
    st.write("Ensuring data is complete, cleaned, and ready for modeling.")

    df_model = df.copy()                                         #Work with a copy of original data
    df_model.columns = df_model.columns.str.strip().str.lower()  #Standardize column names

    #Input Validation
    if 'year' not in df_model.columns:           #Checks if a required column exists in the DataFrame
        st.error("The data does not contain a 'year' column. Please check your data source.") #Displays an error message in the Streamlit interface
        st.stop()                                #Stops the script from executing further in Streamlit
  
    #Data Type Conversion
    df_model['year'] = pd.to_numeric(df_model['year'], errors='coerce')
    df_model['value'] = pd.to_numeric(df_model['value'], errors='coerce')
    #pd.to_numeric(errors='coerce') – Converts strings to numeric and replaces invalid parsing with NaN

    #Column Formatting
    df_model['item'] = df_model['item'].str.strip().str.lower()
    df_model['area'] = df_model['area'].str.strip().str.lower()
    #.str.strip() – Removes leading/trailing whitespace
    #.str.lower() – Converts text to lowercase for consistency

    #Missing Values Handling
    df_model.dropna(subset=['year', 'value', 'item', 'area'], inplace=True) #Drops rows where any of the specified columns have missing values
    
    #Data Preview
    st.write("Cleaned and ready data preview:") #Displays a text message on the Streamlit app indicating the data preview section
    st.dataframe(df_model.head()) #Shows the first few rows of the cleaned DataFrame df_model in an interactive table

    #Step 4: 🎯 Feature Selection from User
    st.subheader("Select Features for Prediction")  #Adds a subheading in the Streamlit UI to denote the feature selection section
    
    #Unique Feature Extraction
    crops   = sorted(df_model['item'].unique()) #Extracts unique crop names from the 'item' column and sorts them alphabetically
    regions = sorted(df_model['area'].unique()) #Extracts unique region names from the 'area' column and sorts them alphabetically
    
    #Dropdown Filters
    selected_crop   = st.selectbox("Select Crop", options=crops) #Creates a dropdown menu for users to select a crop from the list of unique crops
    selected_region = st.selectbox("Select Region", options=regions) #Creates a dropdown menu for users to select a region from the list of unique regions
    
    #Slider for Year Selection
    selected_year = st.slider(                      #Adds a slider allowing users to pick a year for prediction
        "Select Year for Prediction",
        min_value=int(df_model['year'].min()),      #Slider range starts from the minimum year in the data
        max_value=int(df_model['year'].max()) + 5,  #Maximum slider value extends 5 years beyond the latest year in the data
        value=int(df_model['year'].max()) + 1       #Default selected year is set to one year after the latest data year
    )
    #Model Selection
    model_type = st.selectbox("Select Regression Model", ["Linear Regression", "Ridge Regression", "Random Forest"])
    #st.selectbox() → Dropdown menu for selecting regression model

    #Step 5: Modeling Logic
    #Data Filtering for Modeling
    model_data = df_model[
        (df_model['item'] == selected_crop) &
        (df_model['area'] == selected_region)
    ]
    #df[...] → Filters a DataFrame based on conditions
    #==      → Compares values in each row
    #&       → Logical AND to combine conditions

    #Data Sufficiency Check
    if len(model_data) < 5:           #len() → Returns the number of rows in the DataFrame
        st.warning("Not enough data points to build a model. Try another crop or region.") #Displays a warning message in Streamlit
    #Training Info Display
    else:
        st.write(f"Using {len(model_data)} records for model training.")   #Displays formatted text and variables in the Streamlit app
        #Feature (X) and Target (y) Separation
        X = model_data[['year']]
        y = model_data['value']
       #df[['column']] → Selects a column as a DataFrame (not Series), used for model input features
       #df['column']   → Selects a single column as a Series, used for target variable 

        # Conditional model initialization
                #Model Initialization and Training
        if model_type == "Linear Regression":
            model = LinearRegression()  #LinearRegression() → Initializes a simple linear model
        #Linear Regression – Simple method that finds the best straight line to predict the outcome
        #Used when there's a linear relationship between year and production
        
        elif model_type == "Ridge Regression":
            model = Ridge()  #Ridge() → Adds penalty for large coefficients to improve generalization
        #Ridge Regression – Linear model with L2 regularization to prevent overfitting
        #Useful when data has multicollinearity or noise
        
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            #RandomForestRegressor() → Builds 100 trees and combines their predictions (bagging approach)
        #Random Forest – Ensemble model that builds multiple decision trees and averages their output
        #Captures non-linear patterns, robust to outliers, suitable for complex agricultural trends
        
        model.fit(X, y)  #Trains the selected model using year (X) and value (y)

        #Prediction for Selected Year
        pred_value = model.predict(np.array([[selected_year]]))[0]
        
        #model.predict()    → Makes a prediction using the trained model
        #np.array([[...]])) → Converts a scalar value into a 2D NumPy array required by the model
        #[0]                → Extracts the scalar prediction value from the array
      

#Note: Why All Models May Give Similar Predictions

#Linear Regression, Ridge Regression, and Random Forest may produce similar results in this case because:
  #1. Only one input feature (year) is used, so the relationship is simple and nearly linear
  #2. Ridge adds regularization, but has minimal effect when there's no multicollinearity or overfitting
  #3. Random Forest works best with many features and complex patterns; with a single numeric feature, it builds simple splits
      #that mimic linear trends — hence, predictions remain similar to linear models
#This behavior is expected and confirms the data has a clean, predictable time-based trend

        #Step 6: 📊 Display Prediction

        #Add Subheader to the Streamlit Page
        st.subheader(f"📊 Prediction Result ({model_type})")  #Adds a subheading in the Streamlit interface for better structure and readability
        
        #Write Text Description for the Prediction Output
        st.write(f"Predicted production for **{selected_crop.title()}** in **{selected_region.title()}** for year **{selected_year}**:")
        #st.write() → Renders formatted markdown-like text in Streamlit
        #f""        → Python f-string for embedding variables
        #.title()   → Capitalizes the first letter of each word (used for cleaner display of crop and region names)
        #**text**   → Markdown-style bold formatting supported by st.write()

        #Display the Predicted Production Metric
        st.metric(label="Predicted Production (tonnes)", value=f"{pred_value:,.2f}")

        #st.metric() → Displays a labeled metric with a numeric value in Streamlit
        #label= → Sets the title for the metric card
        #value=f"{pred_value:,.2f}" → Formats the prediction value:
          #:,.2f → Adds comma as a thousand separator and rounds to 2 decimal places (e.g., 12,345.68)

        #Step 7: 📉 Model Performance
        #Generate Predictions for Training Data
        y_pred = model.predict(X)             #Predicts output using the trained Linear Regression model
        
        #Calculate Performance Metrics
        #Mean Squared Error (MSE)
          #Measures the average squared difference between actual and predicted values — smaller values mean better predictions
        #Mean Absolute Error (MAE)
          #Measures the average absolute difference between actual and predicted values — shows how far off predictions are on average
        #R-squared (R²)
          #Shows how well the model explains the variation in data — closer to 1 means a better fit

        mse = mean_squared_error(y, y_pred)   #Calculates the average of squared prediction errors
        mae = mean_absolute_error(y, y_pred)  #Calculates the average of absolute prediction errors
        r2 = model.score(X, y)                #Returns the R-squared value (coefficient of determination)

        #Quick Metric Descriptions:
        #MSE (Mean Squared Error) : Penalizes large errors more; lower is better
        #MAE (Mean Absolute Error): Average of how wrong predictions are; lower is better
        #R² (R-squared): How well the model explains the variance in the data; ranges from 0 to 1 (higher is better)
        
        #Display Metrics in Streamlit
        st.subheader(f"📉 Model Performance Metrics ({model_type})")      #Adds a section title in the app for clarity
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")  #Prints formatted text in the app
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}") #Displays MAE with 2 decimal formatting
        st.write(f"R-squared (R2): {r2:.2f}")             #Displays R² value with 2 decimal formatting

        #Step 8: 📈 Plotting Trend
        #Prepare Data for Plotting
        st.subheader("📈 Production Trend")              #Adds a subheader in the app to label the section
        
        #Extracts only year and value columns from the filtered dataset and creates a copy for plotting
        plot_df = model_data[['year', 'value']].copy()
        #Creates a new row for the predicted value for the selected future year
        pred_row = pd.DataFrame({'year': [selected_year], 'value': [pred_value]})
        #Concatenates the prediction row with historical data and sorts it by year
        plot_df = pd.concat([plot_df, pred_row], ignore_index=True).sort_values('year')

        #Add Flags for Prediction Highlighting

        #Adds a boolean column to flag the prediction year (used to highlight the red point)
        plot_df['is_prediction'] = plot_df['year'] == selected_year
        #Ensures year is rounded to integer format for consistent display on the x-axis
        plot_df['year_display'] = plot_df['year'].round(0).astype(int)

        #Create and Customize Matplotlib Plot
        fig, ax = plt.subplots()       #Initializes a Matplotlib figure and axes object

        #Plots a line chart with historical + predicted values using circular markers
        ax.plot(plot_df['year_display'], plot_df['value'], marker='o', label='Historical + Prediction')
        #Highlights the prediction point in red with a larger size and keeps it on top layer (zorder=5)
        ax.scatter(plot_df[plot_df['is_prediction']]['year_display'],
                   plot_df[plot_df['is_prediction']]['value'],
                   color='red', label='Prediction', s=100, zorder=5)
        
        #Labeling and Layout
        ax.set_xlabel("Year")                 #Sets the x-axis label
        ax.set_ylabel("Production (tonnes)")  #Sets the y-axis label
        ax.set_title(f"Production Trend for {selected_crop.title()} in {selected_region.title()}") #Adds a dynamic title using the selected crop and region
        ax.legend()                           #Displays the legend to distinguish between historical and predicted data
        ax.grid(True)                         #Enables grid lines for better readability
        #Display the Plot in Streamlit
        st.pyplot(fig)                        #Renders the Matplotlib figure directly in the Streamlit app

        #Step 9: 💾 Download Section
        #Section Header
        st.subheader("💾 Download Data")     #Adds a subheader in the app to label the download section

        #Add description column for prediction values
        plot_df['note'] = plot_df['is_prediction'].apply(
            lambda x: "Predicted value - may reflect mid-year estimate" if x else ""
        ) #Creates a new column note that adds a description only for predicted rows using a lambda function

        #Drop helper columns
        #Clean Data Before Export
        export_df = plot_df.drop(columns=['is_prediction', 'year_display']) #Removes helper columns used for plotting before exporting to CSV

        #💾 Prepare in-memory CSV buffer
        csv_buffer = io.StringIO()                 #Creates an in-memory string buffer to hold CSV data
        export_df.to_csv(csv_buffer, index=False)  #Writes the cleaned dataframe into the CSV format inside the buffer, excluding row indexes
        csv_bytes = csv_buffer.getvalue().encode() #Converts the CSV string into bytes for download

        #⬇️ Download button
        st.download_button(
            label="Download Data as CSV",
            data=csv_bytes,
            file_name=f"crop_prediction_{selected_crop}_{selected_region}_{selected_year}.csv",
            mime="text/csv"
        )
      #Adds a button in Streamlit to allow users to download the CSV file with a dynamic filename including crop, region, and year

      #Optional: Include a note in the dataframe or export that float years mean mid-year predictions
      #plot_df['note'] = plot_df['year'].apply(lambda x: "Predicted - mid-year estimate" if x == selected_year else "")


#Modeling & Prediction Page Description
   #This page enables users to build, evaluate, and use predictive models for crop production based on historical data
   #It features an interactive interface for selecting input variables and displays predicted crop yields alongside 
    #key model performance metrics for informed decision-making

#Key Features Implemented

#Data Cleaning
  #Standardized column names by stripping whitespaces and converting to lowercase
  #Converted 'year' and 'value' columns to numeric types and handled missing values
  #Removed rows with incomplete data to ensure model accuracy
#Interactive Filters
  #Added dropdown menus to select Crop and Region for focused analysis
  #Added slider to select prediction year, allowing future projections
#Data Validation
  #Included warnings for insufficient data points to build reliable models
#Modeling & Prediction
  #Built a linear regression model to predict crop production based on historical year-value data
  #Provided predicted production values for user-selected years
#Model Performance Metrics
  #Calculated and displayed Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) values
#Visualization Enhancements
  #Plotted production trend including historical data and predicted values
  #Highlighted predicted points distinctly for clarity
  #Formatted year axis to integer values for cleaner visuals
#Data Export
  #Enabled CSV download of historical and predicted data with descriptive notes for predictions


#Code Refinements
#Data Filtering           Filtered dataset by selected crop and region for model training
#Data Sufficiency Check   Added minimum data points check to ensure model reliability
#Feature-Target Split     Separated input feature (year) and target variable (production value)
#Model Initialization     Used sklearn LinearRegression for simplicity and interpretability
#Model Training           Trained model only if sufficient data is available
#Prediction Logic         Predicted production value for user-selected future year
#Dynamic Display          Used f-string and markdown for clean, readable output
#Performance Metrics      Calculated MSE, MAE, and R² to evaluate model accuracy
#Visualization            Plotted historical + predicted data with clear labeling and highlighted prediction point
#Download Feature         Enabled CSV download of combined historical and prediction data
#Data Cleaning            Dropped helper columns before CSV export for clean output
#UI Feedback              Warnings and info messages guide users on data sufficiency and results


#Commands Used
#Pandas
#df.copy()                          Creates a copy of the DataFrame
#df.columns.str.strip().str.lower() Cleans and standardizes column names
#pd.to_numeric()                    Converts columns to numeric, handling errors with 'coerce'
#df.dropna()                        Removes rows with missing values
#df['col']                          Access specific columns in DataFrame
#df['col'].unique()                 Extracts unique values from a column
#df[df['col'] == val]               Filters rows based on a condition
#pd.DataFrame()                     Creates a new DataFrame (e.g., for predicted rows)
#pd.concat()                        Concatenates multiple DataFrames vertically
#df.sort_values(by='col')           Sorts DataFrame rows by a column

#Scikit-learn
#LinearRegression()                 Initializes linear regression model
#model.fit(X, y)                    Fits the model to feature and target variables
#model.predict(X)                   Predicts target values using the fitted model
#mean_squared_error(y, y_pred)      Calculates MSE for model evaluation
#mean_absolute_error(y, y_pred)     Calculates MAE for model evaluation

#Matplotlib
#plt.subplots()                     Creates figure and axes objects for plotting
#ax.plot()                          Plots line and markers for data visualization
#ax.scatter()                       Highlights specific data points (e.g., predictions)
#ax.set_xlabel(), ax.set_ylabel()   Sets axis labels
#ax.set_title()                     Sets plot title
#ax.legend()                        Displays legend on plot
#ax.grid()                          Adds grid lines to the plot

#Streamlit
#st.title()                         Adds a main title to the page
#st.subheader()                     Adds section subheadings
#st.write()                         Displays text or data frames
#st.dataframe()                     Displays DataFrame tables interactively
#st.selectbox()                     Creates dropdown selection boxes
#st.slider()                        Creates a slider input widget
#st.warning()                       Shows warning messages for users
#st.metric()                        Displays key numeric values prominently
#st.pyplot()                        Renders matplotlib figures in the app
#st.download_button()               Enables downloading data files (e.g., CSV)

#Python I/O
#io.StringIO()	                    In-memory text stream for CSV
#df.to_csv(buffer)	                Write DataFrame to CSV in memory
#buffer.getvalue().encode()	        Convert to byte format for download

#📊 Top Production per Country
elif page == "📊 Top Production per Country":

    #Page Title and Instructions
    st.title("📊 Top 5 Crop Production by Country")
    #Display instructions for the user
    st.markdown("""
    Select a country and optionally a year to view the top 5 crops by production volume.
    This helps identify the most important crops contributing to the country's total production.
    """)

    #Step 1: Load & Filter Data Efficiently (cached)
    @st.cache_data                   #Cache the function to improve performance on repeated runs
    def load_production_data():
        #Define the local Excel file path
        file_path = r"C:\Users\Bala Sowntharya\Documents\Crop_Production_Prediction_Project\FAOSTAT_data.xlsx"
        
        #Load Excel data into pandas DataFrame
        df = pd.read_excel(file_path)
        
        #Keep only production rows
        df = df[df['Element'] == 'Production']
        
        #Drop rows with missing Area, Item, Value, and Year
        df = df.dropna(subset=['Area', 'Item', 'Value', 'Year'])
        
        #Return the cleaned DataFrame
        return df

    #Load cleaned production data
    df = load_production_data()

    #Step 2: Dropdown for selecting a country
    countries = df['Area'].unique()            #Get unique country names from the 'Area' column
    selected_country = st.selectbox("Select a Country", countries) #Display dropdown in Streamlit to select a country

    #Optional Year Filter (add “All Years” option)
    #Create a dropdown for optional year selection (with "All Years" as default)
    years = sorted(df['Year'].unique())        #Get sorted list of unique years from the data
    selected_year = st.selectbox("Select a Year", options=["All Years"] + [str(y) for y in years])  #Add "All Years" + year options to dropdown

    #Step 3: Filter data for the selected country and year (if specified)
    if selected_year == "All Years":           #If user chooses "All Years", filter only by country
        country_data = df[df['Area'] == selected_country]
    else:                                      #If a specific year is selected, filter by both country and year
        country_data = df[(df['Area'] == selected_country) & (df['Year'] == int(selected_year))]

    #Step 4: Aggregate production by crop and get top 5 crops
    top_crops = country_data.groupby('Item')['Value'].sum().reset_index()  #Group by crop type and sum production values
    top_crops = top_crops.sort_values(by='Value', ascending=False).head(5) #Sort crops by highest production and select top 5

    #Step 5: Display top 5 crops in a bar chart using Plotly Express
    fig_top_crops = px.bar(   #Create a bar chart using Plotly
        top_crops,            #Data source: aggregated top crops
        x='Item',             #X-axis: crop names
        y='Value',            #Y-axis: production values
        title=f"Top 5 Crop Production in {selected_country}" + (f" ({selected_year})" if selected_year != "All Years" else ""),  #Dynamic title with country and year
        labels={'Item': 'Crop', 'Value': 'Production (tons)'},  #Axis labels for clarity
        text='Value'          #Display production values on the bars
    )
    fig_top_crops.update_traces(texttemplate='%{text:.2s}', textposition='outside') #Format the value labels and place them outside the bars
    fig_top_crops.update_layout(yaxis=dict(title='Production (tons)'), xaxis=dict(title='Crop')) #Customize chart layout
        
    #Step 6: Show the chart in Streamlit
    st.plotly_chart(fig_top_crops, use_container_width=True)  #Display the Plotly bar chart in the Streamlit app using full container width for better visibility

#Page Description
#This page lets users select a country to view its top 5 crops by total production. 
#The data is filtered for production values, aggregated by crop, and visualized using an interactive bar chart for quick insights into key agricultural outputs.

#Key Features Implemented
# User Interaction
  #Dropdown menu allows users to select a specific country for crop production insights
#Data Aggregation
  #Grouped data by crop ('Item') and aggregated total production ('Value') for selected country
  #Sorted and extracted top 5 crops contributing most to national production volume
#Insights Visualization
  #Used Plotly Express to create an interactive and visually clear bar chart of top crops
  #Displayed crop-wise production with labeled bars for immediate interpretation
#Instructional Design
  #Included page title and markdown guidance to explain purpose and usage clearly

#Code Refinement
#Data Caching             Used @st.cache_data to load Excel only once, improving performance
#NaN Handling             Dropped rows with missing Area, Item, or Value to maintain data quality
#Country Filtering        Extracted data specific to the selected country for relevance
#Data Grouping            Applied groupby() and sum() to compute total production by crop
#Data Sorting             Sorted production values descendingly to highlight top contributors
#Top-N Selection          Limited results to top 5 crops for clarity and focus
#Bar Chart Customization  Added text labels, axis titles, and used Plotly’s interactive bar chart
#Responsive Chart         Set use_container_width=True to ensure chart adapts to layout
#Dynamic Titles           Chart title reflects selected country for contextual relevance
#Code Modularity          Encapsulated data load in a function for reusability and clean structure


#Commands Used
#Pandas
#pd.read_excel()                  Load Excel file into DataFrame
#df.dropna()                     Remove rows with missing values in critical columns
#df[df['Element'] == 'Production']  Filter rows to keep only production data
#df['Area'].unique()             Get unique list of countries for dropdown
#groupby('Item')['Value'].sum()  Aggregate total production by crop
#sort_values(by='Value', ascending=False).head(5)  Sort and select top 5 crops by production
#reset_index()                   Reset DataFrame index after grouping for plotting
#df['col'].unique()        # Retrieves unique values from a column
#sorted(df['col'])         # Sorts column values (usually for dropdowns)
#df.groupby()              # Groups data by a column
#df['col'].sum()           # Aggregates values in a column (sum in this case)
#df.reset_index()          # Resets DataFrame index after groupby
#df.sort_values()          # Sorts DataFrame based on one or more columns
#df.head()                 # Returns the top N rows of a DataFrame

#Plotly Express
#px.bar()                       Create bar chart for top crops by production
#update_traces(texttemplate='%{text:.2s}', textposition='outside')  Show data labels on bars
#update_layout()                Customize axis titles and font sizes for clarity

#Streamlit
#st.title()                    Set page title
#st.markdown()                 Display descriptions and instructions in markdown
#st.selectbox()                Dropdown menu for selecting country
#st.plotly_chart()             Render Plotly charts in Streamlit app
#@st.cache_data                Cache data loading to improve performance and avoid reloads

#Actionable Insights
elif page == "🌱 Actionable Insights":
    st.title("🌱 Actionable Insights for Resource Allocation & Agricultural Planning")

    #📁 Step 1: Define a cached function to load and process the Excel data
    #Cache the function so data is only loaded once unless the file changes
    @st.cache_data
    #Define function to load and process Excel data
    def load_and_process_data(path): 
      #Set the full file path to the Excel dataset
      file_path = r"C:\Users\Bala Sowntharya\Documents\Crop_Production_Prediction_Project\FAOSTAT_data.xlsx"
      #Read the Excel file into a DataFrame using pandas
      df = pd.read_excel(file_path)
      return df
      
    #🔍 Step 2: Filter Relevant Elements (Area harvested, Production, Yield)
    #Define the list of key agricultural elements to retain
    required_elements = ['Area harvested', 'Production', 'Yield']
    #Filter the dataset to only include rows with selected elements
    df_filtered = df[df['Element'].isin(required_elements)]
    # Drop rows with missing values in key columns
    df_filtered = df_filtered.dropna(subset=['Area', 'Item', 'Year', 'Element', 'Value'])

    #Step 3: Pivot Table to Wide Format
    #Create a pivot table to reshape data from long to wide format
    df_pivot = df_filtered.pivot_table(
        index=['Area', 'Item', 'Year'],        #Grouping keys: Region, Crop, and Year
        columns='Element',                     #Spread 'Element' values into separate columns
        values='Value',                        #Use 'Value' as cell content
        aggfunc='mean'                         #If duplicates exist, compute their mean
    ).reset_index()                            #Flatten the index for a clean DataFrame

    #Step 4: Rename Columns for Clarity and Consistency
    df_pivot.rename(columns={
        'Area harvested': 'Area_ha',           #Rename harvested area to 'Area_ha'
        'Production': 'Production_tonnes',     #production to 'Production_tonnes'
        'Yield': 'Yield_kg_per_ha'             #Rename yield to 'Yield_kg_per_ha'
    }, inplace=True) 

    #Drop rows with any missing values in key metrics
    df_pivot.dropna(subset=['Area_ha', 'Production_tonnes', 'Yield_kg_per_ha'], inplace=True) #Remove rows with missing values in key metrics to ensure clean data
    df_pivot.drop_duplicates(inplace=True)

    #Step 5: Calculate new efficiency metrics for further analysis
    df_pivot['Yield_tonnes_per_ha'] = df_pivot['Yield_kg_per_ha'] / 1000   #Convert yield to tonnes per hectare
    df_pivot['Yield_Efficiency'] = df_pivot['Yield_tonnes_per_ha'] / df_pivot['Area_ha']  #Yield efficiency calculation
    df_pivot['Production_Efficiency'] = df_pivot['Yield_tonnes_per_ha'] / df_pivot['Production_tonnes']  #Production efficiency

    #📋 Step 6: Display Processed Data Snapshot
    st.subheader("📈 Processed Data Snapshot")
    st.dataframe(df_pivot)

    #📊 Step 7: Group Data for Recommendations
    # Group data by Area and Item, calculating average Yield and Production Efficiency
    summary = df_pivot.groupby(['Area', 'Item'])[['Yield_Efficiency', 'Production_Efficiency']].mean().reset_index()
    
    #Clean the summary first
    summary_clean = summary.dropna(subset=['Yield_Efficiency', 'Production_Efficiency'])

    #Replace inf/-inf with the string "Infinity" in summary
    summary['Yield_Efficiency'] = summary['Yield_Efficiency'].replace([np.inf, -np.inf], 'Infinity')

    #Ensure the entire column is treated as string for display in summary
    summary['Yield_Efficiency'] = summary['Yield_Efficiency'].astype(str)

    #Replace inf/-inf with 'Infinity' and convert to string in summary_clean to fix blanks in Streamlit
    summary_clean['Yield_Efficiency'] = summary_clean['Yield_Efficiency'].replace([np.inf, -np.inf], 'Infinity').astype(str)
    summary_clean['Production_Efficiency'] = summary_clean['Production_Efficiency'].replace([np.inf, -np.inf], 'Infinity').astype(str)

    #Top 5 by Yield Efficiency
    #top_yield_eff = summary.sort_values(by='Yield_Efficiency', ascending=False).head(5)  # Sort by Yield Efficiency descending, select top 5 regions
    #st.markdown("### 🌾 Top 5 Regions by Yield Efficiency")
    #st.table(top_yield_eff)  # Display top 5 yield efficiency table

    #Top 5 by Production Efficiency (proxy for water efficiency)
    top_prod_eff = summary_clean.sort_values(by='Production_Efficiency', ascending=False).head(5)  # Sort by Production Efficiency descending, select top 5 regions (water efficiency proxy)
    
    #Display the cleaned table
    st.markdown("### 💧 Top 5 Regions by Production Efficiency")
    st.dataframe(top_prod_eff, width=800)  # Display top 5 production efficiency table

    #📅 Step 8: Year-wise Crop Production Table and Graph
    st.subheader("📅 Year-wise Crop Production Summary") #Display subheader for Year-wise Crop Production Summary section

    # --> Group data by year and sum production
    #Filter original df for 'Production' element, group by Year, sum total production
    yearly_production = df[df['Element'] == 'Production'].groupby('Year')['Value'].sum().reset_index()
    #Rename the 'Value' column to 'Total_Production' for clarity
    yearly_production.rename(columns={'Value': 'Total_Production'}, inplace=True)

    # --> Show the table
    #Show the yearly production data as a styled dataframe (2 decimal places)
    st.dataframe(yearly_production.style.format({"Total_Production": "{:.2f}"}), use_container_width=True)

    # --> Plot graph using Plotly
    import plotly.express as px
    fig = px.line(                 #Plot the yearly production line chart
        yearly_production, 
        x='Year', 
        y='Total_Production',
        title="Year-wise Total Crop Production",
        labels={'Total_Production': 'Production (tons)', 'Year': 'Year'},
        markers=True
    )
    #Display the Plotly chart in Streamlit with container width enabled
    st.plotly_chart(fig, use_container_width=True)

    # 🧾 Step 9: Human-Readable Actionable Recommendations
    #Display subheader for actionable insights section
    st.subheader("📝 Human-Readable Insights")
    #Header for Yield Efficiency recommendations
    #st.markdown("#### 🔼 Based on Yield Efficiency")
    #Loop over top yield efficiency rows and display recommendations
    #for _, row in top_yield_eff.iterrows():
        #st.markdown(f"- **{row['Area']}** should consider allocating more land to **{row['Item']}** for high yield efficiency.")
    #Header for Production Efficiency recommendations (water/resource use)
    st.markdown("#### 🔼 Based on Production Efficiency (proxy for water use)")
    #Loop over top production efficiency rows and display recommendations
    for _, row in top_prod_eff.iterrows():
        st.markdown(f"- **{row['Area']}** should focus on growing **{row['Item']}** to maximize water/resource use.")

    # 📥 Step 10: Downloadable Efficiency Summary
    st.subheader("📤 Download Efficiency Summary CSV") #Display subheader for downloadable CSV summary
    #Create a download button to export the efficiency summary dataframe as CSV
    st.download_button(
        label="Download CSV Summary",
        data=summary.to_csv(index=False),
        file_name="agri_efficiency_summary.csv",
        mime='text/csv'
    )

    #Decision-Support System     : Helps plan land and water allocation based on data
    #Multi-Metric Recommendations: Uses both yield and production efficiency
    #Human-Readable Insights     : Offers text recommendations planners can understand
    #Downloadable Summary        : Enables offline use of insights in meetings or reports
    #Scalable Design             : Easily add cost, fertilizer, climate, or pest data later

#Insights Page Description
  #This page provides actionable insights by summarizing crop production efficiencies
  #It highlights top-performing crops and regions based on yield and production efficiency metrics
  #Users receive clear recommendations to optimize crop allocation and resource usage

#Key Features Implemented
#Data Aggregation
  #Grouped data by Area and Crop to calculate average Yield and Production Efficiencies
  #Sorted and extracted top performers for yield efficiency and production efficiency
#Insights Presentation
  #Displayed human-readable suggestions highlighting which crops and regions show highest efficiencies
  #Used markdown formatting for clarity and emphasis on key insights
#Data Export
  #Enabled CSV download of summarized efficiency metrics for offline review and sharing

#Code Refinement
#Data Caching             Used @st.cache_data to avoid redundant data loading and speed up app
#NaN Handling             Dropped rows with missing values for accurate insights
#Interactive Filters      Added dropdowns for Area (Region) and Item (Crop) filtering
#Data Grouping            Applied groupby() on Year and Region/Crop for summarized metrics
#Data Sorting             Sorted grouped data by Year to ensure correct timeline in charts
#Responsive Visualization Set use_container_width=True for line charts to maintain layout consistency
#Dropdown Optimization    Limited item options to top frequent crops for user convenience
#Dynamic Titles           Updated chart titles dynamically based on user selections for context clarity
#Infinity Handling        Replaced infinite values in yield efficiency to prevent blanks or confusion
#Table Display            Fixed Streamlit table issues to show proper values instead of blanks or infinity
#Infinity Handling        Replaced infinite values in yield efficiency to prevent blanks or confusion
#Data Cleaning            Cleaned inf and NaN values consistently for accurate sorting and display
#Table Display            Fixed Streamlit table issues to show proper values instead of blanks or infinity


#Commands Used
#Pandas
#pd.read_excel()	         Reads an Excel file into a DataFrame
#df.dropna()	             Removes rows with missing values
#df.pivot_table()	         Reshapes DataFrame from long to wide format using a pivot
#df.rename()	             Renames columns
#df.drop_duplicates()	     Removes duplicate rows
#df['col'] = ...	         Creates or modifies DataFrame columns
#df.groupby()              Groups data for aggregation
#df.mean()                 Calculates mean values for grouped data
#df.replace()	             Replaces values in DataFrame
#df.astype()	             Changes data type of columns
#df.sort_values()          Sorts DataFrame by specified columns
#df.iterrows()             Iterates through DataFrame rows
#df.to_csv()               Converts DataFrame to CSV format

#Streamlit
#st.subheader()            Adds subsection headings in the app
#st.markdown()             Displays markdown text for rich formatting
#st.download_button()      Creates a download button for files like CSVs
#st.cache_data()	         Caches data loading and processing function
#st.title()	               Sets a main title for the page
#st.subheader()	           Adds a subsection heading
#st.dataframe()	           Displays a scrollable DataFrame in the app
#st.markdown()	           Adds formatted text using Markdown
#st.download_button()	     Adds a button to download data (like a CSV)
#st.plotly_chart()	       Displays a Plotly chart in the app

#Plotly Express
#px.line()	               Creates a line chart