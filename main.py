import pandas as pd
from Data_prep import data_preprocessing
from DL_model import dl_model
from RL_model import rl_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv('accepted_2007_to_2018Q4.csv')

    cleaned_df = data_preprocessing(df)

    # DL part
    X = cleaned_df.drop("loan_status", axis=1)  
    Y = cleaned_df["loan_status"]   

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=38)


    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    dl_metrics = dl_model(x_train, y_train, x_test, y_test)

    # RL part

    EPV = rl_model(cleaned_df) 

    print(f'The F1 score and the AUC of the deep learning model are {dl_metrics["F1"]} and {dl_metrics["AUC"]}')
    print(f'The EPV of the RL model is {EPV}')
    

if __name__ == '__main__':
    main()


