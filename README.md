Temps de Réponse de la Brigade des Pompiers de Londres
==============================

This data science project is about learning and predicting the response duration of the fire brigade in the greater London area.
It is based on the data provided by data.london.gov.uk from the 01.01.2009 to the 30.05.2024.

## Prerequisites
To use the project efficiently, follow these steps:

### Step 0: Download the Datasets
Download the datasets and save them in the correct folders based on the architecture shown below.
- [London Fire Brigade Incident Records - London Datastore](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)
- [London Fire Brigade Mobilisation Records - London Datastore](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)
- [Download UK Postcodes with Latitude and Longitude](https://www.freemaptools.com/download-uk-postcode-lat-lng.htm)

### Step 1: Install the Requirements
Install the necessary dependencies by running: `pip install -r requirements.txt`

### Step 2: first run these notebooks in this order to merge the datasets and provide with a featured cleaned csv file called ML_data.csv
- 01_importation_fusion.ipynb
- 02_weather.ipynb
- 03_featuring.ipynb

### Step 3: two possibilities
- A => More DETAILED : continue running the following notebooks to show the differents models with a detailed processing included : 
   - 04_regression.ipynb
   - 05_classification.ipynb
   - 06_deep_learning.ipynb
- B => More EFFICIENT : go to the src folder and run the main notebook, this one will use the following scripts to run the wanted models.
   - preprocessing.py
   - regression.py
   - classification.py
   - deep_learning.py
   - utils.py

### Step 4: optionnal
- use the exploration.ipynb & dataviz.ipynb notebooks to run the different illustrations that can be found in the report.
- run the comand : `streamlit run streamlit_app_cv.py` to show the app


Project Architecture
------------
    ├── LICENSE
    ├── README.md           <- the top-level README for developers using this project.
    ├── data                <- not on Github (only in .gitignore)
    │   ├── processed       <- the final, canonical data sets for modeling.
    │   │   └── dataviz.csv
    │   │   └── ML_data.csv
    │   │ 
    │   └── raw             <- the original, immutable data dump.
    │   │   └── Incident
    │   │   │   └── LFB Incident data from 2009 - 2017.csv
    │   │   │   └── LFB Incident data from 2018 onwards.csv.xlsx
    │   │   │ 
    │   │   └── Mobilisation
    │   │   │   └── LFB Incident data from 2009 - 2017.csv
    │   │   │   └── LFB Incident data from 2018 onwards.csv.xlsx
    │   │   │   └── LFB Incident data from 2018 onwards.csv.xlsx
    │   │   │ 
    │   │   └── merged_data.csv
    │   │ 
    │   └── external        <- External data, added to broaden the dataset or to help plot
    │       └── ukpostcodes.csv
    │       └── weather.csv
    │
    ├── models                <-  Where models are saved to be loaded again if needed
    │
    ├── notebooks             <- Jupyter notebooks. Naming convention is a number and also exploration & dataviz
    │   ├── __init__.py        <- Makes src a Python module
    │   ├── utils.py                            <- Scripts that contains useful functions
    │   ├── 01_importation_fusion.ipynb         <- Scripts to turn imported data into a single dataset
    │   ├── 02_weather.ipynb                    <- Scripts to import weather data and save it int the project
    │   ├── 03_featuring.ipynb                  <- Scripts to do the feature selection and prepare the dataset to be processed for the machine learning   
    │   ├── 04_regression.ipynb                 <- Scripts to run regression models
    │   ├── 05_classification.ipynb             <- Scripts to run classification models
    │   ├── 06_deep_learning.ipynb              <- Scripts to run deep_learning models
    │   ├── KF_dataviz.ipynb                    <- Scripts to run data visualization (by Keyvan)
    │   ├── KF_exploration.ipynb                <- Scripts to run data exploration 
    │   └── SL_dataviz.ipynb                    <- Scripts to run data visualization (by Suzanne)
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting and in streamlit
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── environement.yml   <- Environment export generated by `conda env export > environment.yml`  
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── utils.py                    <- Scripts that contains useful functions
    │   ├── main.py                     <- Scripts to use to run and save the different models directly (regression / classification / deep learning)
    │   ├── read_models.py              <- Scripts to use to load the different models from the models folder 
    │   ├── preprocessing.py            <- Scripts that contains the necessary functions to do the preprocessing
    │   ├── regression.py               <- Scripts that contains the necessary functions to run regression models
    │   ├── classification.py           <- Scripts that contains the necessary functions to run classification models
    │   ├── deep_learning.py            <- Scripts that contains the necessary functions to run deep learning models
    │   ├── viz.py                      <- Scripts that contains the necessary functions to display figures in streamlit    
    │   └── streamlit_app_cv.py         <- Scripts to use to run the streamlit app
    │  
--------
