from operator import index
import pandas as pd
import copy

df = pd.read_csv('Label/002.csv')

for i in range(len(df)):
    temp = df.loc[i].tolist()
    del(temp[0])
    del(temp[0])
    # print("1",temp)
    # x1 = temp[1]
    # x2 = temp[4]
    # x3 = temp[7]
    for x in range(2):
        for y in range(2):
            if (temp[3*y]<temp[3*(y+1)]):
                temp1 = temp[3*y:3*(y+1)]
                temp[3*y:3*(y+1)] = temp[3*(y+1):3*(y+2)]
                temp[3*(y+1):3*(y+2)] = temp1
    
    df.loc[i,'x1'] = temp[0]
    df.loc[i,'y1'] = temp[1]
    df.loc[i,'r1'] = temp[2]
    df.loc[i,'x2'] = temp[3]
    df.loc[i,'y2'] = temp[4]
    df.loc[i,'r2'] = temp[5]
    df.loc[i,'x3'] = temp[6]
    df.loc[i,'y3'] = temp[7]
    df.loc[i,'r3'] = temp[8]

print(df)
df.to_csv('Label/sorted.csv', index=False)