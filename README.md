# CSE_151A

## Dataset link: https://www.kaggle.com/datasets/nikhil1e9/traffic-violations

## Jupyter Notebook link - [https://colab.research.google.com/drive/158S7DPr74nzn6cxtxhOb1CG0ZGZ0D4N_?usp=sharing](https://colab.research.google.com/github/devPach4545/CSE_151A/blob/main/CSE_151A_Project.ipynb#scrollTo=hjlk0wceV7xT)

Dhaivat Pachchigar

Rushil Mohandas

Christophian Austin Sulaiman

Vincent Vo

Eyaad

Peeyush Jha

Hoang Le

Feiyang Jiang

Christopherian Austin Sulaiman

Uyen Le

## How will we do Data Preprocessing.
- After a rigourous data cleaning and plotting, we would be starting data preprocesing. We have already completed data imputation step and we found out that we have missing values for State', 'Year', 'Make', 'Model', 'Color', 'Driver.City', and 'DL.State.' Therefore, we imputated the above columns in our dataset using random sampling and taking median. Now, we need to encode and normalize our data. It is important for our Machine learning model because we have so many Yes/no and non comparable features such as name of car companies. We also have description that we want to encode. We will encode Yes/No fearues into binary 0/1, and the for the rest we will use one-hot encoding. After that we will normalize our data. 
