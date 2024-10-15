Deep Learning Challenge 

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Challenge Overview](#challenge-overview)
- [Prerequisites](#prerequisites)
  - [Required Tools](#required-tools)
  - [Windows Installation Process](#windows-installation-process)
  - [Repository Setup:](#repository-setup)
- [Repository Structure](#repository-structure)
- [Challenge Instructions](#challenge-instructions)
- [Feature Selection Code Example](#feature-selection-code-example)
- [Analysis](#analysis)
- [Acknowledgements](#acknowledgements)
- [Author](#author)


# Challenge Overview
The Alphabet Soup Charity challenge aims to build a deep learning model that predicts the success of applicants for funding. Alphabet Soup has provided historical data, and our goal is to preprocess this dataset, build a binary classifier using neural networks, and optimize the model for higher predictive accuracy.

This challenge involves preprocessing the data, building and training the neural network, optimizing the model for accuracy, and evaluating its performance. The final deliverables include the trained models saved as ```.h5``` files.


# Prerequisites

For the Deep Learning Challenge, ensure you complete the following requirements:

## Required Tools 
- Install Visual Studio Code and Python on your machine 
- Install the Pandas, Scikit-learn, HDF5, and TensorFlow & Keras libraries
- Create a account on Google Colab

## Windows Installation Process
- Open your terminal or command prompt and run the following commands:

  ``` 
     pip install pandas
     pip install tensorFlow 
     pip install scikit-learn
     pip install h5py
   ```

## Repository Setup:
  - Create a new repository called ```deep-learning-challenge``` on GitHub with a README file
  - Clone the repository to your local machine:   
  ```git clone https://github.com/your-username/deep-learning-challenge.git``` 
  - Navigate into the repository folder and add the starter file to the folder
    - Create two copies of the starter code notebook and call them ```AlphabetSoupCharity_Model``` and ```AlphabetSoupCharity_Optimization```
  - Push the changes to your GitHub repository

```
git add .
git commit -m "Pushing updated notebook"
git push origin main
```


# Repository Structure
```
├── deep-learning-challenge/
│   ├── AlphabetSoupCharity.ipynb
│   ├── AlphabetSoupCharity_Optimization.ipynb
│   ├── AlphabetSoupCharity.h5
│   ├── AlphabetSoupCharity_Optimization.h5
│   ├── README.md
│   └── .gitignore
               
```


# Challenge Instructions 

Step 1: Preprocess the data, drop unnecessary columns, and encode categorical variables.

Step 2: Build and compile the neural network model. Train it using the preprocessed data.

Step 3: Optimize the model by adjusting its structure, adding more neurons, hidden layers, or experimenting with different activation functions. Aim for an accuracy higher than 75%.

Step 4: Write a report explaining the performance of the model and discuss any optimization attempts.

Step 5: Push all changes to your GitHub repository.


# Feature Selection Code Example

```VS Code
# Split our preprocessed data into our features and target arrays
y = numeric_df['IS_SUCCESSFUL'].values
X = numeric_df.drop(columns=['IS_SUCCESSFUL']).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


# Analysis

Overview of the Analysis:
- The purpose of this analysis is to create a neural network model that can predict whether an organization will be successful or not when effectively using the funds given by Alphabet Soup. This is a classification mdoel to predict the success of the organization considering severa features like application type, requested funding amount, etc. The goal is to optimize the baseline model to obtain an accuracy of 75% or higher.

Data Preprocessing:
- The target variable is ```IS_SUCCESSFUL```. 
- The feature variables for the first model are ```APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, NCOME_AMT, SPECIAL_CONSIDERATIONS```, and ```ASK_AMT```.
- The removes variables are ```EIN`` and ```NAME``` since they do not contribute to the predictions. 

Compiling, Training, and Evaluating the Model:
- There are 64 input features after preprocessing and scaling the data with three layers. The first hidden layer has 128 neurons and second hidden layer has 64 neurons, both layers has the ReLU activation function. The output layer has one neuron with the sigmoid activation function for binary classification. The ReLU function was best to effectively mitigating the gradient problem. 
- The initial model achieved an accuracy of 72%, which is a good starting point but has room for improvement. However, to improve the model, a third hiddel layer was added to increase the models complexity with 32 neurons and the ReLU function. The features were changed  slightly and the number of epochs were reduced to 50. In conclusion, the model did not improve much, there was a slight improvement. 

Summary:
Overall, the neural network model did not achieve an accuracy of 75% or higher, which did not meet the performance threshold. However, to further improve the model, other optimizations can include exploring other model such as decision trees or random forests to capture more complex non-linear relationships and effectively handle categorical data. Additionally, understanding the feature importance or correlation to the target variable can help with effective feature sections. 


# Acknowledgements

I want to mention the following individuals and resources for their assistance and support throughout this assignment: 
- Xpert Learning Assistant and ChatGPT
- Class Activities 


# Author

For any questions or feedback, please contact:
- Name: Gursimran Kaur (Simran)
- Email: kaursimran081999@gmail.com
