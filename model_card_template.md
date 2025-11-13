# Model Card 

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Income Classification Model
Algorithm used: RandomForestClassifier (Scikit-learn)
Framework - scikit-learn, FastAPI for deployment
Date - November 2025
Version: v1.0
Files: model.pkl,encoder.pkl, lb.pkl

The model uses demographic and employment data to predict whether an individual makes over 50k a year. 

## Intended Use
Primary Uses: Predict a persons income bracket (whether they make under or above 50k).
Intended use: To learn how to FastAPI for model deployment

## Training Data
Dataset: UCI Adult Census Dataset
Features used: age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, native-country
Label: salary (>50K or <=50K)
Data preprocessing: Missing values handled by dropping or imputing,Categorical features one-hot encoded, Label binarized

## Evaluation Data
The evaluation was performed on the test dataset using 20% of the total data. 

## Metrics
The metrics are as follow: precision - .7353, recall - .6378, f1-score - .6831. 
These evaluate how well the model is able to distingush between income classes. 


## Ethical Considerations
Since this dataset reflects historic socioeconomic inbalances, additional audits should be used to find biases. We should also avoid using for decision making like hiring or credit scoring. 
## Caveats and Recommendations
Since this dataset is from 1994, retraining on a newer dataset would be beneficial. It would also be helpful to include safeguards against biases and insuring fairness.  