# Importing Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML libraries
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('../../data/spotify.csv')

# we must first extract our target variable
#extract the target
y=df['playlist_genre']

#drop the target to extract features
x=df[['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
      'liveness', 'valence', 'tempo', 'duration_ms', 'mode']]

#check the shape
print(f"shape of target {y.shape},\n shape of features {x.shape}")

#split the dataset into Train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

#split the training size into train and eval
x_train,x_eval,y_train,y_eval=train_test_split(x_train,y_train,test_size=0.2,random_state=123)

#check and store numeric columns
num_cols=list(set(x.select_dtypes('number')))
#Create num_pipeline
num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
#create a columntransformer
preprocessor = ColumnTransformer([('numeric', num_pipe, num_cols)])
#Create a full pipeline / end2end pipline
end2end=Pipeline([('Preprocessor',preprocessor),('model',None)])
end2end

#Create A function to Evaluate the models
def evaluate(actual,predicted,model_name):
    PrecisionScore = precision_score(actual,predicted, average='weighted')
    RecallScore = recall_score(actual,predicted, average='weighted')
    F1_score = f1_score(actual,predicted, average='weighted')
    Accuracy = accuracy_score(actual,predicted)

    result={'Model':model_name, 'Precision_Score':PrecisionScore,'Recall_Score':RecallScore,
            'F1_Score':F1_score,'Accuracy':Accuracy}
  
    return result

#This variable will hold the list of dictionaries of the results of the different models
dict_list=[]

models = {'Logistic Regression': LogisticRegression(),
         'K-Nearest Neighbors': KNeighborsClassifier(),
         'Decision Tree': DecisionTreeClassifier(),
         'Support Vector Machine (Linear Kernel)': LinearSVC(),
         'Support Vector Machine (RBF Kernel)': SVC(),
         'Neural Network': MLPClassifier(),
         'Random Forest': RandomForestClassifier(),
         'Gradient Boosting':GradientBoostingClassifier()
         }


#Train all the models Using a for loop

for model_name , model in models.items():

    #fit data to the pipeline
    end2end_pipeline=Pipeline([('Preprocessor',preprocessor),('model',model)])
    end2end_pipeline.fit(x_train,y_train)
    #make predictions
    y_pred= end2end_pipeline.predict(x_eval)

    #evaluate the model using the evaluate function
    eval=evaluate(y_eval,y_pred,model_name)
    dict_list.append(eval)


#Put models results in a dataframe
df_results=pd.DataFrame(dict_list)

#sort the results by F1 score
df_results.sort_values(by='F1_Score',ascending=False,inplace=True,ignore_index=True)
#display results
print(f'[Info] Results of the different models:\n{df_results}')

# Set up the parameter grid for each model
random_forest_params = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [3, 4, 5, 6, 7],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

gradient_boosting_params = {
    'model__learning_rate': [0.1, 0.01, 0.001],
    'model__n_estimators': [100, 200, 300],
    'model__subsample': [0.5, 0.7, 1.0],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

neural_network_params = {
    'model__hidden_layer_sizes': [(100,), (100, 50)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001]
}

# Perform hyperparameter tuning with cross-validation for each model
models = {
    'Random Forest': (RandomForestClassifier(), random_forest_params),
    'Gradient Boosting': (GradientBoostingClassifier(), gradient_boosting_params),
    'Neural Network': (MLPClassifier(), neural_network_params)
}

# Perform hyperparameter tuning with cross-validation for each model
best_score=[]
for model_name, (model, params) in models.items():
    #make a pipeline 
    pipe=Pipeline([('Preprocessor',preprocessor),('model',model)])
    random_search = RandomizedSearchCV(pipe, params, n_iter=10, cv=5, scoring=['accuracy', 'f1_macro'], refit='f1_macro' , n_jobs=-1)
    #fit randomsearch
    random_search.fit(x_train, y_train)

    #Print Best parameters
    print(f"Best parameters for {model_name}:")
    print(random_search.best_params_)
    

    #Put scores in a dict
    scores={'model':model_name,'F1_score':random_search.cv_results_['mean_test_f1_macro'][random_search.best_index_],'Accuracy':random_search.cv_results_['mean_test_accuracy'][random_search.best_index_]}
    best_score.append(scores)

#Put scores in a dataframe
print('=========================================')
scores_df=pd.DataFrame(best_score)

print(f'[Info] Best scores for each model:\n{scores_df}')
