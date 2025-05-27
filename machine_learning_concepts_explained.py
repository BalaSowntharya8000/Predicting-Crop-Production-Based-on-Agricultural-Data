#Machine Learning
#Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data
#and make predictions or decisions without being explicitly programmed for every task.
   #Instead of using hard-coded rules, ML algorithms improve automatically through experience by analyzing historical data.
   #It is used in many fields like healthcare, finance, and especially agriculture to drive data-based decision-making.

#Importance of Machine Learning in Agriculture
#Machine Learning is becoming increasingly important in agriculture because it allows for smarter, data-driven practices.

#It helps forecast crop yields by analyzing trends in past production and environmental conditions.
#It detects patterns in crop health, growth cycles, and harvesting times.
#It improves resource planning by predicting water, fertilizer, and labor needs.
#It enhances policy-making by providing insights into food supply and regional productivity.

#In this project:
#We use Machine Learning to predict future crop production using historical agricultural data, including:
# - Year
# - Area Harvested
# - Yield
#This helps farmers, planners, and researchers anticipate future output and optimize strategies.

#Summary of Workflow
#1. Clean the Data        → Fix missing values, types, and formatting
#2. Explore the Data      → Understand trends in crops and regions
#3. Choose Input Features → E.g., use year to predict production
#4. Select Model          → Try Linear, Ridge, and Random Forest
#5. Train the Model       → model.fit(X, y)
#6. Predict               → model.predict(new_year)
#7. Evaluate              → Check MSE, MAE, and R2
#8. Visualize             → Show trend and forecast in a chart
#9. Export                → Download prediction as CSV

##1) Data Cleaning & Preprocessing
#Definition: Prepare raw data for analysis by removing inconsistencies, converting types, and formatting text.

#Purpose: Ensure the dataset is accurate and standardized before modeling.

#What We Did:
# - Removed duplicates using df.drop_duplicates()
# - Converted 'year' and 'value' columns to numeric using pd.to_numeric()
# - Cleaned text data (e.g., crop names) using .str.strip() and .str.lower()
# - Handled missing values using df.dropna()

#Why Important: Models need clean data to learn accurately and avoid errors during training.

##2) Exploratory Data Analysis (EDA)
#Definition: Analyze data distributions, patterns, and anomalies visually and statistically.

#Purpose: Understand relationships between features like Area, Yield, and Production.

#What We Did:
# - Identified top crops and regions using value_counts() and groupby()
# - Tracked production trends using line plots (sns.lineplot, px.line)
# - Detected anomalies in crop productivity

#Why Important: Helps form hypotheses, guides feature selection, and uncovers insights before modeling.

##3) Feature Selection
#Definition: Choosing input variables (features) that influence the target (production). 

#Purpose: Use only meaningful inputs for model accuracy.

#What We Did:
# - Selected only "year" as the independent variable (X)
# - Used "production value" as the dependent target (y)

#Why Important: Reduces model complexity and enhances interpretability.

##4) Train-Test Split
#Definition:
#Train-Test Split is the process of dividing data into:
# - Training Set → used to teach the model patterns from past data
# - Test Set     → used to check how well the model works on new/unseen data

#Why We Used:
#To avoid overfitting and to ensure the model performs well on future data, not just on what it was trained on.

#What We Did:
#In this project, some crop-region combinations had limited data, so we used all available data for training.
#But we showed a warning if there were fewer than 5 records — to prevent overfitting risk.

#What is Overfitting? (Important Concept)
#Overfitting happens when a model learns the training data too well, including its noise and random variations.
#This leads to poor predictions on new data.

#In simple terms:
# - If a model is too accurate on the training set but performs poorly on future data → It is overfitting.

#Example in This Project:
#Imagine you train a model to predict crop production using only a few years of data (e.g., 3-4 years).
#The model might perfectly "fit" those years, but its prediction for the next year could be totally wrong.
#This is because it learned the noise or small fluctuations instead of the general trend.

#How We Prevented It:
# - We ensured there are **at least 5 records** before training a model (checked using len(model_data))
# - We used simpler models like **Linear Regression** for cases with fewer data points
# - In larger datasets, we would use **Train-Test Split** to validate how the model performs on unseen data

#When to Use:
#Always use Train-Test Split when you have enough data.
#It helps test whether your model is truly learning or just memorizing.

##5) Model Selection
#Definition:
#Model selection means choosing which machine learning algorithm is best suited for your data and problem type.

#Why We Used:
#Different algorithms have strengths. Some are simple (Linear), some handle noise well (Ridge), and some capture complex trends (Random Forest).
#We tried all three to compare and understand which one fits crop prediction best.

#What We Did:
# - Linear Regression      → Best for straight-line trends (easy to explain)
# - Ridge Regression       → Adds regularization to reduce overfitting
# - Random Forest Regressor→ Tree-based method that handles non-linear patterns and outliers

#When to Use:
# - Linear       : When the trend is mostly straight and the data is clean.
# - Ridge        : When you have multicollinearity or too many features.
# - Random Forest: When the relationship is complex and not strictly linear.

#Linear Regression
#Fits a straight line: y = mx + b
#Best used when the relationship between input and output is linear.

#Ridge Regression
#Similar to Linear Regression, but adds a penalty for large coefficients (L2 regularization)
#Helps avoid overfitting when features are noisy.

#Random Forest
#Builds multiple decision trees and averages their outputs (ensemble)
#Good for non-linear trends and resistant to overfitting.

#Caching in Streamlit
#@st.cache_data: Caches function outputs to avoid redundant processing
#Boosts speed and performance

##6) Model Training
#Definition:
#Training is the process where the model "learns" from historical data by finding patterns between input (e.g., year) and output (e.g., production).

#Why We Used:
#To make the model intelligent - so it can predict future values by understanding past trends.

#What We Did:
#We used the .fit() function to train the model:
#model.fit(X, y)

#Here, X = year and y = production
#The model learns the relationship: "How does the year affect crop production?"

#When to Use:
#Training must be done before predictions. It’s the first step in making your model useful.

##7) Prediction
#Definition:
#Prediction is when the model applies what it has learned to forecast the value for a new or future input.

#Why We Used:
#To answer real-world questions like:
# "What will be the production in 2026 for wheat in India?"

#What We Did:
# - Took user input (e.g., year = 2026)
# - Converted it into the format model understands: np.array([[2026]])
# - Called model.predict() to get the output

#Example:
# pred_value = model.predict(np.array([[selected_year]]))[0]

#When to Use:
#After training; Used anytime you want the model to estimate a result based on input features.

##8) Model Evaluation
#Definition:
#Model evaluation means measuring how good or bad the model is at making predictions.
#We use numeric metrics to calculate error (difference between actual and predicted values).

#Why We Used:
#To decide:
# - Is the model accurate enough to be trusted?
# - Which machine learning algorithm performs best for predicting crop production?
# - Can we rely on this model for future decision-making?

#What We Did in This Project:
#After the model predicted values (based on year → production), we compared them with actual values using 3 popular regression metrics:

#Used 3 standard metrics:
# 1. Mean Squared Error (MSE)
# 2. Mean Absolute Error (MAE)
# 3. R-squared (R² Score)

#Code Used:
# mse = mean_squared_error(y, y_pred)
# mae = mean_absolute_error(y, y_pred)
# r2 = model.score(X, y)

#Metric Meaning (Beginner Friendly):

#MSE (Mean Squared Error):
#   - Measures the squared difference between real and predicted production values.
#   - Large errors are penalized more due to squaring.
#   - Best when you want to reduce the impact of big mistakes.
#   - Lower MSE means better model performance.

#MAE (Mean Absolute Error):
#   - Measures the average difference between real and predicted values (ignores sign).
#   - Simple to understand: "On average, how far off is the prediction?"
#   - Lower MAE means predictions are closer to real values.
#   - For example: MAE = 2,000 → model is off by 2,000 tons on average.

#R² Score (R-squared):
#   - Ranges from 0 to 1
#   - 1 = Perfect prediction, 0 = Model does not explain the variation at all.
#   - Tells us how well the input (year) explains the variation in crop production.
#   - In this project, R² helps us understand if year is a strong predictor.

#Example in Our Project:
#Let’s say we predicted wheat production in India for 2026:
# - MSE = 3,240,000 → Squared error across all years is 3.24 million
# - MAE = 1,800     → On average, model was off by 1,800 tons
# - R² = 0.92       → 92% of variation in production is explained by year → Very good model

#When to Use:
#Model evaluation is used **after prediction** and **before accepting results**.
#It tells you whether your model is good enough to use in real-world scenarios.

##9) Visualization of Predictions 
#Definition:
#Visualization of predictions means creating a graph or chart that shows:
# - Actual production data over time (historical)
# - Predicted value for a future year (forecast)

#Why We Used:
# - To help users **visually verify the prediction**
# - To show how well the model follows the historical trend
# - Because visual results are more understandable for decision-makers, especially in agriculture

#What We Did in This Project:
#We used **Matplotlib** to create a line plot that includes:
# - A line connecting all historical production values by year
# - A **red dot** to highlight the predicted production for the selected future year
# - Axes labeled for clarity, with year formatted as integer and production with commas

#Example:
#Let’s say we predicted wheat production for the year **2026** in India.
# - The historical data from past years (e.g., 2010 to 2025) is plotted as a line.
# - The prediction for 2026 (a future year selected by the user) is shown as a red dot on the graph.
# - This helps visualize how the model extends the trend into the future.

#Note:
#The prediction for future years like 2026 is based entirely on historical trends.
#No actual data exists for 2026 — this is a **forecasted value**, not observed.


#Code Used 
# fig, ax = plt.subplots()
# ax.plot(years, production, marker='o')              → Line chart for actual values
# ax.scatter(pred_year, pred_value, color='red')      → Red dot for prediction
# ax.set_title(), ax.set_xlabel(), ax.set_ylabel()    → Adds labels and title
# st.pyplot(fig)                                      → Renders the plot in Streamlit

#When to Use:
#Always visualize predictions after modeling.
#It makes your model more transparent, easy to explain, and suitable for presentations or real-world planning.

#Graph
#This graph is part of the **Modeling & Prediction** page in our Streamlit dashboard, allowing users to:
# - See the trend of selected crop in a specific region
# - Identify how the forecast compares to historical performance
# - Download the result including both table and graph data

##10) CSV Export (Download Prediction)
#Definition:
#CSV Export is the process of allowing users to download the model’s output — including historical data and the predicted value — as a .csv (comma-separated values) file.

#Why We Used:
# - To let users (farmers, researchers, planners) save the prediction results
# - For offline analysis, record-keeping, or sharing reports
# - Especially useful in agriculture where decision-makers often rely on spreadsheets

#What We Did in This Project:
#After generating the prediction and combining it with historical data:
# - We created a new DataFrame (plot_df) that includes:
#     - All previous years' crop production data
#     - The predicted production for the selected future year
#     - A note/flag indicating which row is a prediction
# - We cleaned up temporary/helper columns
# - We used `io.StringIO()` to hold the CSV content in memory (no need to save on disk)
# - We added a **Streamlit download button** to let users save the file

#Example:
#If a user selects:
#   Crop   = Wheat
#   Region = India
#   Year   = 2026

#The downloaded CSV file will include:
# - Year, Value, Note
# - Historical production (e.g., 2015 to 2025)
# - 2026 with the predicted production and a note like: "Predicted value - may reflect mid-year estimate"

#Code Used:
# csv_buffer = io.StringIO()
# export_df.to_csv(csv_buffer, index=False)
# csv_bytes = csv_buffer.getvalue().encode()
# st.download_button(label="Download", data=csv_bytes, file_name="prediction.csv")

#When to Use:
# - After prediction is complete and visualized
# - When users want to save results for reference or use in external tools like Excel, Power BI, etc.

# - The filename is dynamic, including crop name, region, and year (e.g., `crop_prediction_wheat_india_2026.csv`)
# - Notes are included in the exported file to clearly mark predicted values