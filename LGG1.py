from pandas import read_csv
from pandas import set_option
import pandas as pd
from sklearn.model_selection import train_test_split

# Import dataset
filename = 'spambase.csv'

names = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will',
         'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money',
         'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999',
         'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(',
         '[', '!', '$', '#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
         'spam']
full_data = read_csv(filename, names=names)
print("--------Processing, Please Wait--------------")
# Compute the lub,glb and averages, which will be the interval boundary
lub=full_data.min()
glb=full_data.max()
avg0=full_data.loc[full_data['spam']==0,:].mean()
avg1=full_data.loc[full_data['spam']==1,:].mean()

# Compare to ensure the interval boundary
midlub={}
midglb={}
for f in names:
        if avg0[f]>= avg1[f]:
            midglb[f]=avg0[f]
            midlub[f]=avg1[f]
        else:
            midglb[f] = avg1[f]
            midlub[f] = avg0[f]

# Data pre-processing
for f in names:
    bin=[lub[f],midlub[f],midglb[f],glb[f]]
    if f != 'spam':
        seq=pd.cut(full_data[f],bin,labels=['1','2','3'],include_lowest=True)
        full_data[f]=seq
set_option('display.max_columns',58)
# print(full_data)

# Divide into Training and Testing datasets
validation_size=0.4
seed=7
Train,Test = train_test_split(full_data,test_size=validation_size, random_state=seed)

# Algorithm 4.1
def LGGSet(D):
    x = D.iloc[0, :]
    H = x
    i = 1

    while (len(D)- i) != 0:
        x = D.iloc[i, :]
        H=LGGConj(H,x)
        i = i + 1
    return H

# Algorithm 4.2
def LGGConj(x, y):
    z={}
    z[0]={}
    z[1]={}
    # To judge whether x is the first entry, if not, the len will equal to 2
    if len(x)==2:
        x=x.T
        z[0]=x[0]
        z[1]=x[1]
        if y['spam']==0:
            for f in names:
                if y[f] not in z[0][f]:
                    z[0][f].append(y[f])
                else:
                    z[0][f] = z[0][f]

        else:
            for f in names:
                if y[f] not in z[1][f]:
                    z[1][f].append(y[f])
                else:
                    z[1][f] = z[1][f]

    # If it is the first entry, put the initial data as the first temporary hypothesis
    else:
        if y['spam'] == 0 and x['spam'] == 0:
            z[0] = TempCombine(x, y)
        if y['spam'] == 1 and x['spam'] == 1:
            z[1] = TempCombine(x, y)
        if y['spam'] == 1 and x['spam'] == 0:
            z[1] = TempCombine(y,y)
            z[0] = TempCombine(x,x)
        if y['spam'] == 0 and x['spam'] == 1:
            z[1] = TempCombine(x,x)
            z[0] = TempCombine(y,y)

    df = pd.DataFrame(z, index=names, columns=(0, 1))
    z=df.T
    return z

def TempCombine(x,y):
    L= {}
    for f in names:
        if x[f] == y[f]:
            L[f] = [x[f]]
        else:
            L[f] = [x[f]]
            L[f].append(y[f])
    return L

# Call function
H=LGGSet(Train)

# Process the result and put into Spam_1 and Spam_0 respectively
HP=H.T
Spam_1={}
Spam_0={}
for f in names:
    HP[0][f].sort()
    HP[1][f].sort()

    if HP[1][f] != ['1','2','3'] and HP[1][f] != HP[0][f]:
        Spam_1[f] = HP[1][f]

    if HP[0][f] != ['1','2','3'] and HP[0][f] != HP[1][f]:
        Spam_0[f] = HP[0][f]
print("Spam_0:\n",Spam_0)
print("Spam_1:\n",Spam_1)

#  ----------------------Evaluation---------------------
TP=0
TN=0
P=0
N=0
FN=0
FP=0
for i in range(len(Test)):
    count=0
    if Test.loc[Test.index[i],'spam'] == 1:
        P+=1
        for attr in Spam_1.keys():
            if Test.loc[Test.index[i],attr] in Spam_1[attr]:
                count += 1
        if count == len(Spam_1.keys()):
            TP += 1
        else:
            FN+=1
    else:
        N+=1
        for attr in Spam_0.keys():
            if Test.loc[Test.index[i],attr] in Spam_0[attr]:
                count += 1
        if count == len(Spam_0.keys()):
            TN += 1
        else:
            FP += 1


Accuracy=(TP+TN)/len(Test)*100
Precision=TP/(TP+FP)*100
Recall=TP/(FN+TP) *100
F1=2*Precision*Recall/(Precision+Recall)
print("--------Performance for Spam Classifier---------")
print("FN",FN)
print("FP",FP)
print("TP",TP)
print("TN",TN)
print("Accuracy:",round(Accuracy,2),"%")
print("Recall:",round(Recall,2),"%")
print("Precision:",round(Precision,2),"%")
print("F1",round(F1,2),"%")
