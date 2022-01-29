import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from collections import defaultdict
import rec_utils

def main():
    data = pd.read_csv('movies.csv')
    df = data[['UserID', 'MovieID', 'Rating']]
    
    ind_ratings = rec_utils.get_user_input(data.Title.unique())
    ind_ratings = rec_utils.clean_df(ind_ratings, data)
    df = df.append(ind_ratings, ignore_index = True, sort = True)
    
    reader = Reader(rating_scale =(1,5))
    new_data = Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)
    
    trainset = new_data.build_full_trainset()
    algo = SVD(n_factors = 100, n_epochs = 20, lr_all = 0.005, reg_all = 0.02)
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    
    top_n = rec_utils.get_top_n(predictions)
    
    recs = rec_utils.get_recommendations(top_n, data)
    print('Results:')
    for r in recs:
        print(r)
        
main()