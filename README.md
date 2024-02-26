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

### What are the next 2 models you are thinking of and why?

#### 1. Neural Networks:

Applicability to the Dataset:
A Neural Network can be advantageous in capturing complex relationships within our dataset. It can handle non-linearities in the data and adapt to the interactions between features such as description of violation, seatbelt usage, injury occurrence, vehicle details, state, year, make, gender, etc.
The flexibility of neural networks allows it to model intricate patterns, making it suitable for a diverse dataset like traffic violations.


#### 2. Decision Trees:

Applicability to the Dataset:
Decision Trees are inherently interpretable, which can be valuable in understanding and explaining the decisions made by the model. This is particularly important in domains like traffic violations where transparency is crucial.
Decision Trees can handle both categorical and numerical features, making them suitable for datasets with diverse types of information.

In summary, a Neural Network can capture complex relationships in our dataset, while Decision Trees, with their interpretability and ability to handle diverse features, could provide insights into the factors influencing traffic violations. Experimenting with both models and fine-tuning their parameters will help determine the most effective approach for your task.



