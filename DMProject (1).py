
# coding: utf-8

# ## Books Recommendation System

# 

# In[1]:

import pandas as pd
import csv
import numpy as np
from pandas import Series, DataFrame


# 

# In[2]:

path = "\Users\HP\Desktop\Books.csv"


# In[3]:

data = pd.read_csv(path, names=['UserID', 'BookID', 'Ratings'])


# In[4]:

len(data)


# In[5]:

users = data['UserID'].unique()
len(users)


# In[6]:

books = data['BookID'].unique()
len(books)


# In[7]:

df = pd.DataFrame(data)
df.head()


# In[8]:

matrix = df.pivot(columns='BookID', index='UserID', values='Ratings').fillna(0)


# In[9]:

matrixcopy = df.pivot(columns='BookID', index='UserID', values='Ratings').fillna(0)


# In[10]:

matrixcopy['userid'] = matrixcopy.index


# In[11]:

USER = matrixcopy['userid'].unique()
new_df = pd.DataFrame(USER)
for i in range(0,26496):
    USER[i] = i


# In[12]:

matrix.index = USER
matrix.index.name = 'userid'


# In[13]:

matrix


# In[14]:

matrix[matrix['2007770'] > 0]['2007770']


# In[15]:

BOOKS = df['BookID']
df1 = pd.DataFrame(BOOKS)
df['userid'] = df.index


# In[16]:

df1.head()


# In[17]:

M = matrix.as_matrix()
user_ratings_mean = np.mean(M, axis = 1)
M_demeaned = M - user_ratings_mean.reshape(-1, 1)


# In[18]:

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(M_demeaned, k = 50)


# In[19]:

sigma = np.diag(sigma)


# In[20]:

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = matrix.columns)


# In[21]:

preds_df.head()


# In[22]:

def recommend_books(predictions_df, userID, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID 
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data 
    user_data = original_ratings_df[original_ratings_df.userid == (userID)]
    user_full = (user_data.merge(df1, how = 'left', left_on = 'BookID', right_on = 'userid'))

    print ('User {0} has already rated {1} books.'.format(userID, user_full.shape[0]))
    print ('Recommending the highest {0} predicted ratings books not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating books that the user hasn't bought yet.
    recommendations = (df1[~df1['BookID'].isin(user_full['BookID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'BookID',
               right_on = 'BookID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# In[23]:

already_rated, predictions = recommend_books(preds_df, 28, df, 2)


# In[ ]:

already_rated.head()


# In[ ]:

predictions


# In[ ]:



