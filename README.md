As part of my internship with Oasis Infobyte, I was entrusted with three key machine learning projects, each addressing unique business needs across different domains. These projects provided an excellent opportunity to apply advanced data science techniques to real-world problems, enhancing my practical expertise in the field. The tasks I completed were as follows:

## TASK 1 - IRIS FLOWER CLASSIFICATION
This project implements a machine learning model to classify iris flowers into three species: It will compare the setosa (Iris-setosa), versicolor (Iris-versicolor) and virginica (Iris-virginica). The dataset used is the famous Iris dataset, which consists of 150 samples with four features: being the four variables of the study which included sepal length, sepal width, petal length and petal width. The target here is to design a classifier which can correctly determine the type of iris flowers based on these characteristics.

### Features

Sepal Length: Length of the sepal in centimeters.

Sepal Width: Width of the sepal in centimeters.

Petal Length: Length of the petal in centimeters.

Petal Width: Width of the petal in centimeters.

### Project Structure

Data Preprocessing: Cleaning and normalizing the dataset.

Model Training: Using algorithms like Naive Bayes, SVM , Random Forest , Logistic Regression, K-Nearest Neighbors, or Decision Trees to classify the iris species.

Model Evaluation: Assessing model accuracy and performance using metrics such as accuracy.

Web Interface: A simple web interface using Gradio to allow users to input flower measurements and get species predictions in real-time.

### Best performing model

The best performing model is K-Nearest Neighbours with precision 1.0

## Task 2 - CAR PRICE PREDICTION WITH MACHINE LEARNING

This project aims to predict the price of a car based on various features such as car name, year ,present price, Driven kms, Fuel Type, Selling type, Transmission , Owner By leveraging historical data, we train a machine learning model that can estimate the selling price of used cars with high accuracy. The project is designed to help users, car dealers, and potential buyers estimate the fair market value of cars based on their specifications.

### Features

Year: Manufacturing year of the car.

Present Price : Present price of car.

Driven kms : Kms for which car is driven

Fuel Type: Type of fuel the car uses (e.g., Petrol, Diesel, CNG).

Selling type : Sold by dealer or individual owner.

Transmission: Type of transmission (Manual/Automatic).

Owner Type: The number of previous owners.

### Project Structure

Data Collection: A dataset of used car listings, including features that influence the price.

Data Preprocessing: Handling missing values, outliers, and encoding categorical variables.

Model Training: Building machine learning models such as linearRegression , Ridge, DecisionTreeRregressor , RandomForestRegressor, GradientBoostingRegressor , AdaBoostRegressor, BaggingRegressor , ExtraTreesRegressor .

Model Evaluation: Evaluating the model using metrics like Mean Absolute Error (MAE), and R-squared.

Prediction Interface: A simple interface using Gradio for users to input car details and get price predictions.

### Best performing model

![image](https://github.com/user-attachments/assets/b64e91fb-7aa6-4b4f-818d-70792f433d9d)

The best performing model is ExtraRegressionTree with lowest Mean Absolute Error and highest R2score.

## TASK 3 - GMAIL SPAM DETECTION USING MACHINE LEARNING

This project deals with using a model that is learned from training data for the purposes of categorizing incoming emails as spam, not spam (ham). By applying methods of natural language processing (NLP) the project reads the content of the messages and calculates the probabilities of them being spam. The presented model can be used to enhance the email filtering systems and save people from receiving unwanted or even potentially dangerous messages.

### Features

Email : The content of the email, including subject and body text.

Label : Classifies an email as either spam or not spam.

### Project Structure

Data Preprocessing: Tokenizing, cleaning, and vectorizing email data (removing punctuation, converting text to lowercase, removing stopwords).

Feature Extraction: Using methods such as TF-IDF (Term Frequency-Inverse Document Frequency) to convert email text into numerical features.

Model Training: Training machine learning models like GaussianNB, MultinomialNB, BernoulliNB, and Voting Classifier to classify emails.

Model Evaluation: Assessing model performance using accuracy, precision.

Web Interface: A simple web app using streamlit where users can type an email, and the model will predict whether it is spam or not.

### Best performing model

![image](https://github.com/user-attachments/assets/55e42b62-4fd6-4e13-bca6-4e44bb4adb57)

The best performing model is Bernaulii Naive Bayes.

### Technologies Used

Python

Scikit-learn for model building

Gradio and Streamlit for web interface

Pandas and NumPy for data manipulation

Matplotlib or Seaborn for data visualization

NLTK for Natural language processing.
