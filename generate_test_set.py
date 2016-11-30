"""
Generation of the test set, w/ permuted vals
Author : Diviyan Kalainathan
Date : 11/10/2016
"""
#ToDo Multiprocess program?

import os,sys
import pandas as pd
from random import randint
from random import shuffle
import csv
from multiprocessing import Pool
import numpy
from sklearn import metrics


inputdata='../output/obj8/pca_var/cluster_5/pairs_c_5.csv'
outputfolder='../output/test/'
info='../output/obj8/pca_var/cluster_5/publicinfo_c_5.csv'
max_proc=int(sys.argv[1])
max_gen=10
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

def chunk_job(chunk,part_no):
    chunk = pd.merge(chunk, publicinfo, on='SampleID')
    chunk.dropna(how='all')
    chunk['Pairtype'] = 'O'

    # Drop flags:


    chunk = chunk[chunk.SampleID.str.contains('flag')==False]
    to_add = []

    for index, row in chunk.iterrows():
        var1 = row['A']
        var2 = row['B']

        for i in range(max_gen):
            mode = randint(1, 3)
            if mode == 1:
                var1 = var1.split()
                shuffle(var1)
                var1 = " ".join(str(j) for j in var1)
            elif mode == 2:
                var2 = var2.split()
                shuffle(var2)
                var2 = " ".join(str(j) for j in var2)
            elif mode == 3:
                var1 = var1.split()
                shuffle(var1)
                var1 = " ".join(str(j) for j in var1)
                var2 = var2.split()
                shuffle(var2)
                var2 = " ".join(str(j) for j in var2)

            to_add.append([row['SampleID'] + '_Probe' + str(i), var1, var2, row['A-Type'], row['B-Type'], 'P'])

    df2 = pd.DataFrame(to_add, columns=['SampleID', 'A', 'B', 'A-Type', 'B-Type', 'Pairtype'])
    chunk=pd.concat([chunk,df2],ignore_index=True)
    sys.stdout.write('Finishing chunk '+str(part_no)+'\n')
    sys.stdout.flush()
    # chunk= chunk.iloc[numpy.random.permutation(len(chunk))] #No need to shuffle
    # chunk.reset_index(drop=True)
    chunk.to_csv(outputfolder + 'test_crit_p' + str(part_no) + '.csv',  sep=';',index=False)


outputfile = open(outputfolder + 'test_crit_.csv', 'wb')  # Create file
writer = csv.writer(outputfile, delimiter=';', lineterminator='\n')
writer.writerow(['SampleID', 'A', 'B', 'Pairtype'])
outputfile.close()


publicinfo=pd.read_csv(info,sep=';')
publicinfo.columns=['SampleID','A-Type','B-Type']

chunksize=10**4
data=pd.read_csv(inputdata,sep=';', chunksize=chunksize)
print(chunksize)

pool=Pool(processes=max_proc)
partno=0
for chunk in data:
    partno+=1
    pool.apply_async(chunk_job,args=(chunk,partno,))

pool.close()
pool.join()







