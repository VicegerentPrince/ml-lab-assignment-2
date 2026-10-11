# ML Assignment 2: End-to-End Machine Learning System

Muhammad Muneeb Shahzad (FA24-BSE-085)

Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus

## Task 2: Heart Disease Prediction System

Live App: https://heart-disease-fa24-bse-085.streamlit.app

Predicts if a patient has heart disease or not. It uses 4 input attributes (ChestPain, ExerciseAngina, Vessels, Thal) and a Support Vector Classifier. Accuracy on the testing data is 0.75.

Folder: `task2-heart-disease`

| File | What it is |
|:--|:--|
| `Heart Disease Prediction - FA24-BSE-085.ipynb` | Notebook with all the steps |
| `Heart Disease Prediction - FA24-BSE-085.html` and `.pdf` | Saved copies of the notebook |
| `processed.cleveland.data` | Dataset downloaded from UCI |
| `sample-data.csv` | Pre-processed data made by the notebook |
| `svc_trained_model.pkl` | Trained model |
| `app.py` | Streamlit web app |
| `requirements.txt` | Libraries needed |

## Task 1: Titanic Passenger Survival Prediction System

Live App: https://titanic-survival-fa24-bse-085.streamlit.app

The Titanic code given by the instructor. I ran it on my machine and deployed the trained model.

Folder: `task1-titanic`

## How to Run

Open a terminal inside one of the task folders.

```
pip install -r requirements.txt
jupyter notebook
streamlit run app.py
```
