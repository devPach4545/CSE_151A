# CSE_151A

## Dataset link: https://www.kaggle.com/datasets/nikhil1e9/traffic-violations

## Jupyter Notebook link - [https://colab.research.google.com/drive/158S7DPr74nzn6cxtxhOb1CG0ZGZ0D4N_?usp=sharing](https://colab.research.google.com/github/devPach4545/CSE_151A/blob/main/CSE_151A_Project.ipynb#scrollTo=hjlk0wceV7xT)

## Group Members 
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

## Data Preprocessing Plan

After a rigorous data cleaning and exploratory data analysis, we move on to the data preprocessing stage. The dataset has undergone data imputation to address missing values in columns such as 'State', 'Year', 'Make', 'Model', 'Color', 'Driver.City', and 'DL.State'. The imputation was performed using random sampling and taking the median of the respective columns.

### Handling Categorical Variables

To prepare the data for machine learning models, encoding is required, particularly for categorical variables. The dataset contains various categorical features such as Yes/No indicators and non-comparable features like the names of car companies. We'll apply the following strategies for encoding:

1. **Binary Encoding for Yes/No Features:**
   - We will convert Yes/No features into binary representations (0 for No, 1 for Yes) to make them compatible with machine learning algorithms.

2. **One-Hot Encoding for Non-Comparable Categorical Features:**
   - For non-comparable categorical features, such as car company names and other descriptive attributes, we will use one-hot encoding. This technique creates binary columns for each category and indicates the presence of the category with a 1.

### Normalization

Normalization is crucial for ensuring that features with different scales do not unduly influence the machine learning model. We will apply normalization to scale numerical features to a standard range, typically between 0 and 1. This step enhances the performance and convergence of machine learning algorithms.

**Summary of Data Preprocessing Steps:**

1. Imputation: Missing values in 'State', 'Year', 'Make', 'Model', 'Color', 'Driver.City', and 'DL.State' were addressed using random sampling and median imputation.

2. Encoding:
   - Binary Encoding: Yes/No features will be converted to binary representations (0 or 1) This includes things such as driver being male/female, whether or not there was a personal injury, and whether or not there was property damage, to name a few.
   - One-Hot Encoding: Non-comparable categorical features will undergo one-hot encoding.

3. Normalization: Numerical features will be normalized to a standard range for optimal model performance.

By completing these preprocessing steps, the dataset will be ready for training machine learning models, ensuring that it is appropriately formatted and scaled for accurate and reliable predictions.

### Result

***For model 1, logistic regression classifier:***

**From the classification report for training data:**

The precision value for class 0 is 0.99 while the precision value for class 1 is 0.57

The recall value for class 0 is 1 while the value for class 1 is only 0.23

The F1-score value for class 0 is 0.99 while the F1-score value for class 1 is 0.33 

The accuracy of the training data for this model is 0.99

The MSE for training data is 0.011009786476868327

From the confusion matrix: the value for true negative is 44345, the value of false positive is 89, the value for false negative is 406 and the value for true positive is 120

**From the classification report for testing data:**

The precision value for class 0 is 0.99 while the precision value for class 1 is 0.52

The recall value for class 0 is 1 while the value for class 1 is only 0.23 

The F1-score value for class 0 is 0.99 while the F1-score value for class 1 is 0.31

The accuracy of the training data for this model is 0.99

The MSE for testing data is 0.010533807829181495

From the confusion matrix: the value for true negative is 13868, the value for false positive is 31, the value for false negative is 117 and the value for true positive is 34

**From the classification report for validation data:**

The precision value for class 0 is 0.99 while the precision value for class 1 is 0.54

The recall value for class 0 is 1 while the value for class 1 is only 0.24 

The F1-score value for class 0 is 0.99 while the F1-score value for class 1 is 0.34 

The accuracy of the training data for this model is 0.99

The MSE for validation data is 0.010943060498220641

From the confusion matrix: the value for true negative is 13868, the value of false positive is 31, the value for false negative is 117 and the value for true positive is 34

The model will toward the underfitting in the fitting graph.


**For model 2, neural network:**

The value for training loss is 0.0408 while the training accuracy is 0.9893

The value for testing loss is 0.0412 while the testing accuracy is 0.99

The difference in loss is 0.0004 and the difference in accuracy is 0.0006

The best value for val_loss is 0.04218614101409912

**From the classification report for training data:**

The precision value for class 0 is 0.99 while the precision value for class 1 is 0.77

The recall value for class 0 is 1 while the value for class 1 is only 0.13

The F1-score value for class 0 is 0.99 while the F1-score value for class 1 is 0.22

The accuracy of the training data for this model is 0.99

From the confusion matrix: the value for true negative is 13868, the value of false positive is 31, the value for false negative is 117 and the value for true positive is 34

The model will toward the underfitting in the fitting graph.


**For Model 3: Linear SVM**

**From the classification report for training data:**

The precision value for class 0 is 0.92 while the precision value for class 1 is 0.83

The recall value for class 0 is 0.81 while the value for class 1 is only 0.93 

The F1-score value for class 0 is 0.86 while the F1-score value for class 1 is 0.88 

The accuracy of the training data for this model is 0.87

The MSE for training data is 0.1303

From the confusion matrix: the value for true negative is 36018, the value of false positive is 8416, the value for false negative is 3165 and the value for true positive is 41269

**From the classification report for testing data:**

The precision value for class 0 is 1 while the precision value for class 1 is 0.05
The recall value for class 0 is 0.81 while the value for class 1 is only 0.80

The F1-score value for class 0 is 0.90 while the F1-score value for class 1 is 0.09 

The accuracy of the training data for this model is 0.81

The MSE for testing data is 0.1875

From the confusion matrix: the value for true negative is 11291, the value of false positive is 2608, the value for false negative is 26 and the value for true positive is 125

**From the classification report for validation data:**

The precision value for class 0 is 1 while the precision value for class 1 is 0.05
The recall value for class 0 is 0.81 while the value for class 1 is only 0.80

The F1-score value for class 0 is 0.90 while the F1-score value for class 1 is 0.09 

The accuracy of the training data for this model is 0.81

The model will move toward the underfitting in the fitting graph.

