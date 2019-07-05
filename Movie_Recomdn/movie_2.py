# Get the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Importing and Preprocessing

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()

# Data Transformation

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

# Building the correlation matrix

corrMatrix = userRatings.corr()
corrMatrix.head()

# Analyzing the Correlation Matrix

corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()

# Data Preprocessing

myRatings = userRatings.loc[0].dropna()
myRatings

# Writing a simulation to retrive similar movies

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))

# Analyzing the Results of our Recommendation System

simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)
filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)