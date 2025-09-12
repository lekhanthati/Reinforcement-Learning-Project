import pandas as pd 
import numpy as np

def data_preprocessing(df):
    #removing the columns whose null values are greater than half of the dataset

    df = df.dropna(axis=1, thresh=1130350)

    df = df.drop(['id','url','policy_code'],axis=1)

    # droping the columns where there is high no.of unique categorical values 

    df = df.drop(['emp_title','purpose','title','zip_code','addr_state'],axis=1)

    ## cleaning all columns 

    # loan_amnt
    # no null values in the column

    # funded_amnt
    # no null values in the column

    # term
    # droping the null values 
    df = df.dropna(subset=["term"])
    # converted into right data format
    df['term'] = df['term'].str.replace(" months", "").astype(int)

    # int_rate
    # no null values in the column

    # installment
    # no null values in the column

    # grade
    # no null values 
    # label encoding the data 
    map2 = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7}
    df['grade'] = df['grade'].map(map2)

    # sub_grade
    # no null values 
    # label encoding the data 
    subgrades = ['A1','A2','A3','A4','A5',
                'B1','B2','B3','B4','B5',
                'C1','C2','C3','C4','C5',
                'D1','D2','D3','D4','D5',
                'E1','E2','E3','E4','E5',
                'F1','F2','F3','F4','F5',
                'G1','G2','G3','G4','G5']


    map3 = {sg: i+1 for i, sg in enumerate(subgrades)}
    df['sub_grade'] = df['sub_grade'].map(map3)

    # emp_length
    # impute 
    def clean_emp_length(val):
        if pd.isnull(val):
            return np.nan
        if val == '< 1 year':
            return 0
        if val == '10+ years':
            return 10
        return int(val.split()[0])  

    df['emp_length'] = df['emp_length'].apply(clean_emp_length)
    df['emp_length'].fillna(df['emp_length'].median(), inplace=True)

    # home_ownership
    # no null values 
    # one hot encoding
    df['home_ownership'] = df['home_ownership'].replace(['ANY','NONE','OTHER'], 'OTHER')

    # annual_inc
    # no null values

    # verification_status
    # no null values 
    # one hot encoding 

    # Cleaning loan_status
    map1 = {
        'Fully Paid': 0,               
        'Current': None,                
        'In Grace Period': None,      
        'Late (16-30 days)': 1,   
        'Late (31-120 days)': 1,  
        'Charged Off': 1,    
        'Default': 1    
    }

    df = df[df['loan_status'].isin(['Fully Paid', 'Late (16-30 days)', 'Late (31-120 days)', 'Charged Off', 'Default'])]

    df['loan_status'] = df['loan_status'].map(map1)

    # pymnt_plan
    # no null values 
    # droping the columns due to high imbalance in data 
    df = df.drop(['pymnt_plan'],axis=1)

    # delinq_2yrs
    # no null values 

    # earliest_cr_line
    # no null values
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    current_date = pd.to_datetime("today")
    df['earliest_cr_line'] = (current_date - df['earliest_cr_line']).dt.days / 365
    df['earliest_cr_line'] = df['earliest_cr_line'].round(2)

    # fico_range_low
    # no null values 

    # inq_last_6mths
    # dropint the null rows 
    df = df.dropna(subset=["inq_last_6mths"])

    # open_acc
    # no null values 

    # pub_rec
    # no null values 

    # revol_bal
    # no null values

    # revol_util
    # Droping the null values 
    df = df.dropna(subset=["revol_util"])

    # total_acc
    # no null values 

    # initial_list_status
    # no null values 
    # one hot encoding 

    # out_prncp, recoveries, collection_recovery_fee
    # no null values 

    # total_pymnt, total_rec_prncp, total_pymnt_inv, total_rec_int,total_rec_late_fee
    # no null values 

    # last_pymnt_d and issue_d
    # droping the null values 
    df = df.dropna(subset=["last_pymnt_d"])
    # converting the last payment date and issue date into months 
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')

    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['last_pymnt_d'] = (df['last_pymnt_d'].dt.year - df['issue_d'].dt.year) * 12 + (df['last_pymnt_d'].dt.month - df['issue_d'].dt.month)

    df['last_pymnt_d'] = df['last_pymnt_d'].fillna(0).round(2)

    # droping the issue_d columns
    df = df.drop(['issue_d'],axis=1)

    # last_pymnt_amnt
    # no null values 

    # last_credit_pull_d
    # droping the null values 
    df = df.dropna(subset=["last_credit_pull_d"])

    # converting the data into useful format
    df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%Y')
    df['last_credit_pull_year'] = df['last_credit_pull_d'].dt.year
    df['last_credit_pull_month'] = df['last_credit_pull_d'].dt.month
    df = df.drop(['last_credit_pull_d'],axis=1)

    # last_fico_range_high, last_fico_range_low
    # no null values 

    # collections_12_mths_ex_med
    # droping the null values 
    df = df.dropna(subset=["collections_12_mths_ex_med"])

    # acc_now_delinq
    # no null values 

    # tot_coll_amt
    # impute
    df['tot_coll_amt'].fillna(df['tot_coll_amt'].median(), inplace=True)
    df['tot_coll_amt'].fillna(0, inplace=True)

    # adding a missing flag 
    df['tot_coll_amt_missing'] = df['tot_coll_amt'].isnull().astype(int)

    # tot_cur_bal
    # impute
    df['tot_cur_bal'].fillna(df['tot_cur_bal'].median(), inplace=True)
    df['tot_cur_bal_missing'] = df['tot_cur_bal'].isnull().astype(int)

    # open_acc_6m, open_act_il,open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il,il_util, inq_fi, total_cu_tl
    # inq_last_12m
    # Droping the columns due to high null values 
    df = df.drop(['open_acc_6m','open_act_il','open_il_12m', 'open_il_24m',
                'mths_since_rcnt_il','total_bal_il','il_util','inq_fi',
                'total_cu_tl','inq_last_12m'],axis=1)

    # total_rev_hi_lim, acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util
    # impute 
    df['total_rev_hi_lim'].fillna(df['total_rev_hi_lim'].median(), inplace=True)
    df['total_rev_hi_lim_missing'] = df['total_rev_hi_lim'].isnull().astype(int)

    df['acc_open_past_24mths'].fillna(df['acc_open_past_24mths'].median(), inplace=True)
    df['acc_open_past_24mths_missing'] = df['acc_open_past_24mths'].isnull().astype(int)

    df['avg_cur_bal'].fillna(df['avg_cur_bal'].median(), inplace=True)
    df['avg_cur_bal_missing'] = df['avg_cur_bal'].isnull().astype(int)

    df['bc_open_to_buy'].fillna(df['bc_open_to_buy'].median(), inplace=True)
    df['bc_open_to_buy_missing'] = df['bc_open_to_buy'].isnull().astype(int)

    df['bc_util'].fillna(df['bc_util'].median(), inplace=True)
    df['bc_util_missing'] = df['bc_util'].isnull().astype(int)

    # chargeoff_within_12_mths, delinq_amnt
    # no null values 

    # mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, mths_since_recent_bc
    # mths_since_recent_inq, num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl, num_bc_sats
    # impute
    df['mo_sin_old_il_acct'].fillna(df['mo_sin_old_il_acct'].median(), inplace=True)
    df['mo_sin_old_il_acct_missing'] = df['mo_sin_old_il_acct'].isnull().astype(int)

    df['mo_sin_old_rev_tl_op'].fillna(df['mo_sin_old_rev_tl_op'].median(), inplace=True)
    df['mo_sin_old_rev_tl_op_missing'] = df['mo_sin_old_rev_tl_op'].isnull().astype(int)

    df['mo_sin_rcnt_rev_tl_op'].fillna(df['mo_sin_rcnt_rev_tl_op'].median(), inplace=True)
    df['mo_sin_rcnt_rev_tl_op_missing'] = df['mo_sin_rcnt_rev_tl_op'].isnull().astype(int)

    df['mo_sin_rcnt_tl'].fillna(df['mo_sin_rcnt_tl'].median(), inplace=True)
    df['mo_sin_rcnt_tl_missing'] = df['mo_sin_rcnt_tl'].isnull().astype(int)

    df['mort_acc'].fillna(df['mort_acc'].median(), inplace=True)
    df['mort_acc_missing'] = df['mort_acc'].isnull().astype(int)

    df['mths_since_recent_bc'].fillna(df['mths_since_recent_bc'].median(), inplace=True)
    df['mths_since_recent_bc_missing'] = df['mths_since_recent_bc'].isnull().astype(int)

    df['mths_since_recent_inq'].fillna(df['mths_since_recent_inq'].median(), inplace=True)
    df['mths_since_recent_inq_missing'] = df['mths_since_recent_inq'].isnull().astype(int)

    df['num_accts_ever_120_pd'].fillna(df['num_accts_ever_120_pd'].median(), inplace=True)
    df['num_accts_ever_120_pd_missing'] = df['num_accts_ever_120_pd'].isnull().astype(int)

    df['num_actv_bc_tl'].fillna(df['num_actv_bc_tl'].median(), inplace=True)
    df['num_actv_bc_tl_missing'] = df['num_actv_bc_tl'].isnull().astype(int)

    df['num_actv_rev_tl'].fillna(df['num_actv_rev_tl'].median(), inplace=True)
    df['num_actv_rev_tl_missing'] = df['num_actv_rev_tl'].isnull().astype(int)

    df['num_bc_sats'].fillna(df['num_bc_sats'].median(), inplace=True)
    df['num_bc_sats_missing'] = df['num_bc_sats'].isnull().astype(int)

    # hardship_flag
    # removing the columns due to high imbalance in data 
    df = df.drop(['hardship_flag'],axis=1)

    # debt_settlement_flag
    # no null values 
    # label encoding 
    df['debt_settlement_flag'] = df['debt_settlement_flag'].map({'Y': 1, 'N': 0})

    # months_diff
    # no null values

    for i in list(df.columns[64 : 82]):
        df[i].fillna(df[i].median(), inplace=True)
        df[f'{i}_missing'] = df[i].isnull().astype(int)


    # one hot encoding categorical values 

    cat_cols = list(df.select_dtypes(include=['object']).columns)
    
   
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)


    # converting the boolean into int 

    bool_cols = list(df.select_dtypes(include=['bool']).columns)
    df[bool_cols] = df[bool_cols].astype(int)

    # final check for null values 

    df = df.dropna(subset=["dti"])

    # droping the columns with more than 50% null values 
    df_cleaned = df.drop(df.columns[df.isnull().any()].tolist(),axis=1)

    return df_cleaned