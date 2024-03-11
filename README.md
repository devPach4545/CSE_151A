# CSE_151A

## Dataset link: https://www.kaggle.com/datasets/nikhil1e9/traffic-violations

## Jupyter Notebook link - https://colab.research.google.com/github/devPach4545/CSE_151A/blob/main/CSE_151A_Project.ipynb

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

### Where does your model fit in the fitting graph

From the classification report for both training and testing data, the precision and recall for predicting class 1 is low for training and testing data. We chose to focus on the precision and recall for class 1 (person injured) as the metrics for measurement to avoid bias because the dataset has a lot of instances for class 0 but not a lot for class 1. According to the report, the error is similarly high for both testing and training test, indicating relatively poor performance. So the  model will towards to the left in the underfitting part of the fitting graph.

![image](https://github.com/devPach4545/CSE_151A/assets/146885173/10a80f10-12f3-4553-a83d-bee3e13b5487)


### What are the next 2 models you are thinking of and why?

#### 1. Neural Networks:

A Neural Network can be advantageous in capturing complex relationships within our dataset. It can handle non-linearities in the data and adapt to the interactions between features such as description of violation, seatbelt usage, injury occurrence, vehicle details, state, year, make, gender, etc.
The scalability of neural networks enables them to adapt to the complexity of the dataset. By adjusting the number of hidden layers and neurons, this model can accommodate diverse accident scenarios. This adaptability is particularly beneficial when dealing with a vast range of conditions and circumstances that may lead to personal injuries in car accidents. Training a neural network on a comprehensive set of accidents enhances its ability to generalize and make accurate predictions across various scenarios. This is crucial for creating a robust model capable of handling the complexities inherent in predicting personal injuries.

#### 2. Decision Trees:

Decision Trees are inherently interpretable, which can be valuable in understanding and explaining the decisions made by the model. This is particularly important in domains like traffic violations where transparency is crucial.
Employing ensemble methods, such as Random Forests or XGBoost, with Decision Trees enhances predictive performance. By building multiple trees and combining their predictions, these ensemble methods provide robustness against overfitting and improve overall model accuracy.
Decision Trees are valuable for exploratory analysis, aiding in the identification of the most influential features in predicting personal injuries. This feature importance analysis can guide interventions and policies, focusing efforts on mitigating the key factors contributing to injury outcomes.


### Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

After training a logistic regression model on our dataset, we've concluded that it's a fundamental step towards understanding its predictive capabilities. Logistic regression is well-suited for binary classification problems, but to truly bring out its effectiveness, we applied key performance metrics such as accuracy, precision, recall, and the F1 score to the predictions made on the test set. These metrics gave us a comprehensive view of how accurately the model predicts outcomes, balancing the rate of true positives against false positives and negatives. However, we realize that we did not create a confusion matrix which would make this all so much easier. This is a crucial next step, to understand the types of errors our model might be making. Additionally according to the report, both the training and testing sets exhibit similarly high error rates, signaling relatively weak performance. This suggests that the model leans towards the left side, or the underfitting portion, of the model fitting graph. To improve this we could make the model more complex by increasing the model complexity such as utilizing neural networks.

To enhance the model's performance, we could consider several strategies. Firstly, implementing cross-validation, like K-Fold or Repeated K-Fold, could provide a more robust assessment of the model's performance by training it on different subsets of the data. This would help us understand how well the model generalizes to new, unseen data. Secondly, tuning the hyperparameters of the logistic regression model could significantly improve its performance. Lastly, if logistic regression seems too simplistic for our dataset's complexity, we are open to exploring more advanced models like decision trees, random forests, or even neural networks, which could potentially offer better accuracy and insights.

### Conclusion section (Milestone 4 and model 2): What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?
- After running neural network we got the accuracy of 0.98, precisionn of 0.531, and, recall of 0.225. It seems that our model has a lot of true negatives (13869) with compare to true positives only 34. Neural network did not provide a significant improvemnet over logistic regression because our model 1 got the accuracy of 0.98 as well. Both of our models identified same number of true negatives. We are thinking about oversampling the data to get rid of class imbalance.

### Our Third model:-
- First, we thought we would run Decision trees, but after carefull consideration from all group members, we have decided to go with SVM. Since, they are good for binary classification and we also have binary classification problem. 



