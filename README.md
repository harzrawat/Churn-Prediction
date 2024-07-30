# Churn Prediction
---

**Churn Prediction**: It is the prediction of probability of customer likeliness of not using the product or taking the service in future. It’s significant to reduce Customer Aquisition Cost (CAG) for a service provider. 

> ### Generally: 

**Logic**:  
For example, if we want to predict the customer churn in the next 3 months. In this case the data for the last 6 months will be taken and divided into 2 sections divided equally in 3 months each.  

***Let A***: previous 3 months table 

***B***: Later 3 months table 
Training Data: In training data, we will have all the columns of a customer data table from the previous 3 months and create a new feature called Churn with values “0” for “not churn” and “1” for “churn”. 

- Customer ID available in table B will get churn value as 0 in table A and 1 if not available. 

- Model is trained on data table A , where churn column is the target column 

> ### Challenge here:
As here we don’t have any existing data to create churn feature. 




## Steps :  

> ### I. `Synthetic data generation` 

- ### Useful packages I tried :             

    1.Faker 

    2.Scikit-learn 

    3.PyOD 

    4.ctGAN 

    5.Mimesis   
```python
from mimesis import Address, Finance, Datetime, Person 
```
- ### Columns required: 

    | Demography | Channel Valid | Transaction |
    |---------------|---------------|------------|
    | Customer ID | Email Valid |Transaction Date |
    | Age |SMS|Purchase |
    | City |Whatsapp |CLTV|
    |Country|Push notification |RFM |
    |Gender |-----------|Product Category|

- ### Columns created at first are : 

```python
Index(['CustomerID', 'Gender', 'Date_of_Birth', 'Email Valid', 'SMS Valid', 'WhatsApp Valid', 'Notification Valid', 'City', 'Country', 'Product Category', 'Transaction Date', 'Purchase'], dtype='object') 
```

 

 

> ### II. `Adding Noise` 

- ### Ways used to add noise : 

    - Adding random none 
    - Creating anomalies to the numeric data columns 

> ### III. `Data Profiling`         

- Finding for null value columns 
- Knowing the datatypes and valuecount (frequency) 
- Finding Anomalies in data 

> ### IV. `Data Preprocessing` 

- Removing data anomalies  
- Null value imputation
- Encoding of categorical and numerical data 

> ### V. `EDA (Exploratory Data Analysis) `

- ### Using
    ```python 
        import Pandas, Matplotlib, Seaborn, Ydata_profiling
    ```

- *****Finding correlation between columns by plotting graphs***** 

> ### VI. `Feature Engineering` 

- Feature Scaling 
- Feature Creation 
- Feature Selection 
- Missing value imputation 

    - **Using**

        - *Simple Imputer* 
            - Gender 
        - *KNN*
            - Age (had correlation with Purchase) 
        - *Remove*
            - Any column where Purchase is NULL 

ML model 

ML training and evaluation 

 
