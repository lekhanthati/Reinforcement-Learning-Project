import pandas as pd
import numpy as np
from d3rlpy.dataset import MDPDataset
import d3rlpy
from d3rlpy.metrics import InitialStateValueEstimationEvaluator

def rl_model(df):


    df1 = df.copy()
    df2 = df.copy()

    # Actions 
    df1['Action'] = [1 for i in range(len(df))]
    df2['Action'] = [0 for i in range(len(df))]

    df3  = pd.concat([df1, df2], ignore_index=True).sample(frac=1).reset_index(drop=True)

    def reward(data):
        if data['Action'] == 0:
            return 0
        else:
            if data.loan_status == 0:
                return float(data['loan_amnt'] * data['int_rate'] * 0.01)
            else:
                return - float(data['loan_amnt'])

    # Rewards 
    df3.Reward = df3.apply(reward,axis = 1)


    # Creating the MDP dataset

    states = df3.drop(columns=['Action', 'loan_status'],axis = 1).values
    actions = df3.Action.values
    rewards = df3.Reward.values
    terminals = np.ones(len(df3), dtype=bool)


    dataset = MDPDataset(
        observations=states,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )


    # setup algorithm
    dqn = d3rlpy.algos.DQNConfig().create()

    # start offline training
    dqn.fit(
    dataset,
    n_steps=100000, # This is just an example, for the final report the model is trained for multiple steps
    n_steps_per_epoch=10000,

    )

    evaluator = InitialStateValueEstimationEvaluator()
    epv = evaluator(dqn, dataset)

    return epv
