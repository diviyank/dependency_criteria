
# coding: utf-8

# In[15]:

#get_ipython().system('jupyter nbconvert --to script perform_linear_test.ipynb')


# In[11]:

import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import scipy.stats as stats
from random import shuffle
from multiprocessing import Pool,Array
import sys
proc=15
number_lines=50000
res_p=Array('d',range(number_lines))
p_val_p=Array('d',range(number_lines))
res_MI=Array('d',range(number_lines))
p_val_MI=Array('d',range(number_lines))

# In[ ]:

def bin_variable(var1):   # bin with normalization  
    var1=np.array(var1).astype(np.float)
    if abs(np.std(var1))>0.01:
        var1 = (var1 - np.mean(var1))/np.std(var1)
    else:
        var1 = (var1 - np.mean(var1))
    val1 = np.digitize(var1, np.histogram(var1, bins='fd')[1])

    return val1

def p_val_mi(x,y):
    count=0.0
    iterations=3000
    score=metrics.adjusted_mutual_info_score(x,y)
    for i in range(iterations):
        shuffle(x)
        shuffle(y)
        if metrics.adjusted_mutual_info_score(x,y)>=score:
            count+=1.0
    return count/iterations
        
def test1(x,y) :
    return stats.pearsonr(np.array(x).astype(np.float),np.array(y).astype(np.float))[0]
def p_val_test1(x,y) :
    return stats.pearsonr(np.array(x).astype(np.float),np.array(y).astype(np.float))[1]

def test2(x,y) :
    #print(x)
    return metrics.adjusted_mutual_info_score(bin_variable(x),bin_variable(y))
def p_val_test2(x,y):
	return p_val_mi(bin_variable(x),bin_variable(y))


# In[14]:

def job_compute_scores(row):
    print(row)	    
    x=row['X'].split(' ')
    y=row['Y'].split(' ')
    sys.stdout.write('row : '+ str(row['ID']))
    sys.stdout.flush()
    if x[0]=='':
        x.pop(0)
        y.pop(0)
    x=[float(i) for i in x]
    y=[float(j) for j in y]
    #print(x)
    #print(y)
    
    r1=test1(x,y)
    p1=p_val_test1(x,y)
    r2=test2(x,y)
    p2=p_val_test2(x,y)
    
    #Writing results into shared memory
    n_id= int(row['ID'])
    res_p[n_id]=r1
    p_val_p[n_id]=p1
    res_MI[n_id]=r2
    p_val_MI[n_id]=p2
                                                                           


# In[ ]:

#Load dataset
chunked_data=pd.read_csv('linear_dataset.csv',chunksize=10**4)
data=pd.DataFrame()
for chunk in chunked_data:
    data=pd.concat([data,chunk])


# In[12]:

#Main computation loop
p=Pool(processes=proc)
idlist=[]
coeff=[]
noise_sig=[]
nb_pts=[]

for idx,row in data.iterrows():
    #print(idx)
    p.apply_async(job_compute_scores,args=(row,))
    #job_compute_scores(row,res_p,p_val_p,res_MI,p_val_MI)
    idlist.append(row['ID'])
    coeff.append(row['Coeff'])
    nb_pts.append(row['Nb_pts'])
    noise_sig.append(row['Noise/Sig'])
p.close()
p.join()

result=[]
for i in range(len(idlist)):
    result.append([idlist[i],res_p[i],p_val_p[i],res_MI[i],p_val_MI[i],
                   coeff[i],nb_pts[i],noise_sig[i]])

res_df=pd.DataFrame(result,columns=['ID','Pearson_Correlation','Pearson_p-val',
                                    'Mutual_information','MI_p-val',
                                    'Coeff','Nb_pts','Noise/Sig'])
res_df.to_csv('result_linear_test.csv',index=False)
    

