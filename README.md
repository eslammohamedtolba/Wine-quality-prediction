# Wine-quality-prediction
This project aims to predict the quality of red wine based on various features using a machine learning model.
The dataset used in this project is 'winequality-red.csv'.

## Dataset
The dataset used in this project is 'winequality-red.csv', included in the repository.
It contains various features related to red wine, such as acidity levels, alcohol content, and more, along with a quality rating.

## Prerequisites
Before running the code, make sure you have the following Python libraries installed:
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- numpy

## Model Overview
- The dataset is loaded and explored to gain insights into its structure and the relationships between different features and wine quality.
- Data cleaning is performed to check for missing values or inconsistencies in the dataset.
- An exploratory data analysis is conducted to understand the distribution of wine quality ratings and the correlations between features.
- The dataset is split into input (X) and label (Y) data, where the 'quality' column is transformed into a binary label, classifying wines as either high quality (quality >= 7) or low quality (quality < 7).
- A Random Forest Classifier model is trained on the dataset to predict wine quality.
- The model is evaluated for its accuracy on both the training and test datasets.
- The model achieves an accuracy of 89% on the test data, indicating its effectiveness in predicting wine quality.

## Making Predictions
The trained model can be used to make predictions for new input data. 
You can provide a set of wine features, and the model will classify the wine as high or low quality based on these features.

## Contribution
We welcome contributions from the open-source community to improve and enhance this project. If you'd like to contribute, here's how you can get involved:
- Fork this repository to your GitHub account.
- Make your desired changes or additions to the project.
- Submit a pull request, explaining the changes you've made and why they are valuable.
- Your contributions will be reviewed, and if accepted, they will be merged into the main project.
