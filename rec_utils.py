import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from collections import defaultdict

def get_user_input(titles):
    count = 0 
    num = 0
    ratings = []
    unique_title = titles#data.Title.unique()
    while count < 30:
        while True:
            try:
                response = int(input('Rate: {} (On a scale form 1-5, or 0 if unseen)'.format(unique_title[num])))
                if response < 0 or response > 5:
                    raise ValueError
                else:
                    if response > 0:
                        ratings.append(([response, '{}' .format(unique_title[num])]))
                        count += 1
                    break
        
            except ValueError:
                prompt = " "
        num += 1
        
    ind_ratings_df = pd.DataFrame(ratings, columns = ['Rating', 'Title'])
    return ind_ratings_df

def get_id(x, data):
    t_id = data.query('Title == "{}"' .format(x))['MovieID'].sample(1).values
    t_id = int(t_id)
    return t_id

def clean_df(ratings_df, data):
    ratings_df['MovieID'] = ratings_df['Title'].apply(lambda x: get_id(x, data))
    ratings_df['UserID'] = 9999
    ratings_df.drop('Title', axis = 1, inplace = True)
    return ratings_df

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def get_recommendations(top_picks, data):
    recommendations = []
    for t in top_picks[9999]:
        t_df = data.query('MovieID == {}'.format(t[0]))
        sample = t_df[['Title']].sample(1).values
        recommendations.append(sample[0][0])
    return recommendations