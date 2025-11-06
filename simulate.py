import numpy as np
import pandas as pd
from typing import Optional


def get_simulated_scores(abilities_df,
                         item_params_df,
                         num_lives: int,
                         num_sub_levels: int,
                         item_scoring_df = None,
                         seed: Optional[int] = None):

    """
    abilities_df: columns ['participant_id','theta']
    item_params_df: columns ['item_id','a','b','c']
    item_scoring_df: DataFrame with columns ['item_id', 'item_score']
    
    returns 
    simulated_results_df: columns ['AccountId','Score']

    """
    rng = np.random.default_rng(seed)

    simulated_results = []

    for _, person in abilities_df.iterrows():
        participant_id = person["participant_id"]
        theta = float(person["theta"])

        remaining_lives = num_lives
        total_score = 0
        
        #Iterate over questions
        for _, row in item_params_df.iterrows():
            #Iterate over sub-levels
            for _sub in range(num_sub_levels):
                is_correct = False
                #Retry untill sublevel is successful
                while (not is_correct) and (remaining_lives > 0):
                    item_id = int(row["item_id"])
                    a = float(row["a"])
                    b = float(row["b"])
                    c = float(row["c"]) if "c" in row and pd.notna(row["c"]) else 0.0

                    is_correct = simulate_question(theta, a, b, c,rng)

                    if not is_correct:
                        score = 0
                        remaining_lives -=1
                    else:
                        score = get_question_score(item_id, item_scoring_df) if item_scoring_df else 1
                    
                    total_score += score

                    if remaining_lives <= 0:
                            break
            if remaining_lives <= 0:
                break  

        simulated_results.append({"AccountId": participant_id,
                        "Score": total_score})  
    return pd.DataFrame(simulated_results)
          

def simulate_question(theta,a,b,c,rng):

    is_correct = False
   
    if c == None:
        c = 0
    
    p = c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))
    if rng.random() < p:
        is_correct = True
    else:
        is_correct = False
    
    return is_correct

def get_question_score(item_id,item_scoring_df):
    row = item_scoring_df.loc[item_scoring_df["item_id"] == item_id, "item_score"]

    if len(row) == 0:
        return 1   # default score if missing

    return row.iloc[0]