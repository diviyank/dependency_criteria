"""
Plot results of the criterion
Author : Diviyan Kalainathan
Date : 11/10/2016
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy
from itertools import cycle
from sklearn.metrics import auc, average_precision_score,precision_recall_curve

inputdata='../output/test/results/'
colors = cycle(['cyan', 'indigo', 'seagreen', 'gold', 'blue', 'darkorange','red','grey','darkviolet','mediumslateblue'])
crit_names = ["Pearson's correlation",
              "AbsPearson's correlation",
              "Pvalue Pearson",
              "Chi2 test",
              "Mutual information",
              "Corrected Cramer's V",
              "Lopez-Paz Causation coefficient",
              #"FSIC",
              "BF2d mutual info",
              "BFMat mutual info",
              "ScPearson correlation",
              "ScPval-Pearson"
              ]

results=[]
#f, axarr = plt.subplots(2, sharex=True)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2=plt.figure()
ax2=fig2.add_subplot(111)
for crit in crit_names:
    inputfile=inputdata+'test_crit_'+crit[:4]+'.csv'
    if os.path.exists(inputfile):
        print(crit),
        df=pd.read_csv(inputfile,sep=';')
        df=df[numpy.isfinite(df['Target'])]
        #df['Target']=df['Target'].astype('float')
        #print(df.dtypes)
        if crit[:4]!='Lope':
            df=df.sort_values(by='Target', ascending=False)
        else:
            df=df.sort_values(by='Target', ascending=True)

        #print(df)
        N_max=len(df.index)
        print(N_max),
        N=0.0
        Mprobes_max = (df['Pairtype']=='P').sum()
        print(Mprobes_max)
        Mprobes = 0.0
        FDR=[]

        for index,row in df.iterrows():
           N=N+1
           if row['Pairtype']=='P':
               Mprobes+=1

           FDR.append((N_max/N)*(Mprobes/Mprobes_max))

        results.append(FDR)


pickle.dump(results,open(inputdata+'res.p','wb'))
#results=pickle.load(open(inputdata+'res.p','rb'))

'''for i in range(len(results)-1):
    print(results[i]==results[-1])

print(len())'''
#print(results)
for i,color in zip(results,colors):
    ax1.plot(range(len(i)),i,color=color)
    ax2.plot(range(len(i)),i,color=color)
plt.legend(crit_names,loc=4)
plt.xlabel('Number of probes retrieved')
plt.ylabel('False discovery rate')
ax1.set_xscale('log')
plt.show()#FDR w/ probes

'''colors = cycle(['cyan', 'indigo', 'seagreen', 'gold', 'blue', 'darkorange','red','grey','darkviolet','mediumslateblue'])
for crit, color in zip (crit_names,colors):
    tpr=[]
    fpr=[]
    ppv=[]


    print(crit)
    try:
        with open("CEfinal_train_"+crit[:4]+'.csv','r') as results_file:
            df=pd.read_csv(results_file,sep=';')
            df=df.sort_values(by='Target', ascending=False)
       
            P=float((df['Pairtype']!=4).sum())
            Plist=(df['Pairtype']!=4).tolist()
            N=float((df['Pairtype']==4).sum())

            TP=0.0
            FP=0.0
            for index,row in df.iterrows():
                if crit[:4]!='Lope':
                    if row['Pairtype']==4:
                        FP+=1.0
                    else:
                        TP+=1.0
                else:
                     if row['Pairtype']!=4:
                        FP+=1.0
                     else:
                        TP+=1.0

                tpr.append(TP/P) #TPR=recall
                fpr.append(FP/N) #FPR
                ppv.append(TP/(TP+FP))
            tpr,fpr, ppv= (list(t) for t in zip(*sorted(zip(tpr,fpr,ppv))))
            auc_score=auc(fpr,tpr)
            pres,rec,_= precision_recall_curve(Plist,df['Target'].tolist())
            ac_pr_score=average_precision_score(Plist,df['Target'].tolist())
            pl1=ax1.plot(fpr,tpr,label=crit+' (area: {0:3f})'.format(auc_score),color=color)
            pl2=ax2.plot(rec,pres,label=crit+' (area: {0:3f})'.format(ac_pr_score),color=color)

    except IOError:
        continue
        

ax1.plot([0, 1], [0, 1], linestyle='--', color='k',
label='Luck')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve independancy criteria on Kaggle Data')
ax1.legend(loc="lower right")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision recall curve on Kaggle Data')
ax2.legend(loc='best')
plt.show()'''


      
