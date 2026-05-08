import pandas as pd

def preprocess(df):
    
    df = df.copy()
    
    # Drop unnecessary columns
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1, errors='ignore')
    
    # Feature Engineering
    df['AgeGroup'] = pd.cut(df['Age'],
                           bins=[18,30,45,60,100],
                           labels=['Young','Adult','MidAge','Senior'])
    
    df['TenureGroup'] = pd.cut(df['Tenure'],
                              bins=[0,3,6,10],
                              labels=['New','Mid','Loyal'])
    
    # Encoding
    df = pd.get_dummies(df, drop_first=True)
    
    return df