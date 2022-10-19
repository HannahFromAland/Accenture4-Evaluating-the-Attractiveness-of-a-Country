# Accenture4: Evaluating the Attractiveness of a Country
2022 Fall Data Science Capstone Project repository for Team Accenture 4

## Info

#### Project Objective 

Business investment is a major pillar of economic growth. New business investment decision is fraught with many risks and other factors. New investment depends on the factors pertinent to policies and regulations, technology, and risk factors ( policy risk, economic risk etc.). New investment decision also depends on the industry sector/vertical. In this capstone project we will focus on two major sectors: Manufacturing and service.

Two major industries covered in this capstone project:

1.	Manufacturing
2.	Service

A time series forecasting (univariate / multivariate) model can be built to find the manufacturing and service trend in a particular country.

Manufacturing and Service, individually depends on many socio-economic factors including risk factors.

A country wise comparative assessment can be done to assess the better investment destination for a sector (Manufacturing/Service) than the others.

#### 👩🏻‍💻 Authors:
- Di Mu (dm3686)
- Freddy Wong (ww2615)
- Hanlin Yan (hy2654)
- Jace Yang (jy3174) <mark style="background-color:#c3dbfc;">Team Captain</mark>
- Yuan Heng (yh3416)

#### 🧑🏻‍💼 Sponsor/Mentor:
- Paritosh Pramanik from Accenture

#### 🧑🏻‍🏫 CA:
- Aayush Verma
- Jessica Rodriguez

#### 🧑🏻‍🏫 instructor:
- Sining Chen

## Repository Structure

**Data Preparation:**

- Inside the `Data/raw` is the original data we targeted, which were manually downloaded from the worldbank through databank.

- `Data/1_Raw_Data_Cleaning.ipynb` processed and consolidate all the raw data, and output `Data/cleaned/Manufacturing_all.xlsx` and `Data/cleaned/Service_all.xlsx`

- `Data/2_Variable_Selection.ipynb` further filtered out some *bad* predictors base on data quality and their relation to target variable in terms of evidence & intuition, then output to `Data/cleaned/Service.xlsx` and `Data/cleaned/Manufacturing.xlsx`

- `Data/3_Filling_Missingness.ipynb` will conduct NA imputation techniques, assess them, and generate output into the `Data/clean/Manufacturing_filled` and `Data/clean/Service_filled` to be used by the modeling part.

**Modeling:**
- `Model/ARIMA` is our first baseline univariate method.
- `Model/LSTM` allows both univariate and multivariate methods.
- `Model/VAR` is multivariate.
- `Model/prophet` is univariate.
- `Model/Bayes`: TBD