"""
Goal is to find the dependency criterion adapted to our data
Author : Diviyan Kalainathan
Date : 11/10/2016
"""

import os, sys

sys.path.insert(0, os.path.abspath(".."))  # For imports
import pandas as pd
from multiprocessing import Pool
import scipy.stats as stats
from lib.fonollosa import features
import numpy
from sklearn import metrics
import csv
# import lib.fsic as fsic
#import lib.fsic.data as data
#import lib.fsic.indtest as it #Removed for theano locks
import lib.mutual_info_bf.mutual_info as mi
import lib.lopez_paz.indep_crit as lp_crit
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Need to fix Lopez paz Ic

#lp = lp_crit.lp_indep_criterion()

BINARY = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL = "Numerical"

max_proc = int(sys.argv[1])
#inputdata = '../input/kaggle/CEfinal_train'
inputdata='../output/test/test_crit_'
#crit_names = "Mutual information",

crit_names= ["Pearson's correlation",
              "AbsPearson's correlation",
              "Pval-Pearson",
              "Chi2 test",
              "Mutual information",
              "Corrected Cramer's V",
              "Lopez Paz's coefficient",
              #"FSIC",
              "BF2d mutual info",
              "BFMat mutual info",
              "ScPearson correlation",
              "ScPval-Pearson"]


# FSIC param :
# Significance level of the test

def bin_variables(var1,var1type, var2,var2type):
    if var1type==NUMERICAL:
        val1=numpy.digitize(var1,numpy.histogram(var1,bins='auto'))
    else: val1=var1
    if var2type == NUMERICAL:
        val2 = numpy.digitize(var2, numpy.histogram(var2, bins='auto'))
    else:
        val2= var2

    return val1,val2

def confusion_mat(val1, val2):
    '''
    contingency_table = numpy.zeros((len(set(val1)), len(set(val2))))
    for i in range(len(val1)):
        contingency_table[list(set(val1)).index(val1[i]),
                          list(set(val2)).index(val2[i])] += 1'''
    contingency_table = numpy.asarray(
        pd.crosstab(numpy.asarray(val1, dtype='object'), numpy.asarray(val2, dtype='object')))
    # Checking and sorting out bad columns/rows
    # max_len, axis_del = max(contingency_table.shape), [contingency_table.shape].index(
    #    max([contingency_table.shape]))
    contingency_table += 1
    # toremove = [[], []]

    '''for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            if contingency_table[i, j] < 4:  # Suppress the line
                toremove[0].append(i)
                toremove[1].append(j)
                continue

    for value in toremove:
        contingency_table = numpy.delete(contingency_table, value, axis=axis_del)'''

    return contingency_table


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return numpy.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def f_pearson(var1, var2, var1type, var2type):
    return stats.pearsonr(var1, var2)[0]

def f_abspearson(var1, var2, var1type, var2type):
    return abs(stats.pearsonr(var1, var2)[0])

def f_pval_pearson(var1, var2, var1type, var2type):
    return 1 - abs(stats.pearsonr(var1, var2)[1])


def f_sc_pval_pearson(var1, var2, var1type, var2type):
    #Switching categories to obtain the best correlation values
    if var1type == CATEGORICAL or var1type == BINARY:
        var1 = [int(i) for i in var1]
        df = pd.DataFrame([pd.Categorical(var1), var2], columns=['cate', 'val'])
    elif var2type == CATEGORICAL or var2type == BINARY:
        var2 = [int(i) for i in var2]
        df = pd.DataFrame([pd.Categorical(var2), var1], columns=['cate', 'val'])
    else:
        return 1 - abs(stats.pearsonr(var1, var2)[1])

    mean_values = []  # Per category

    for i in df.cate.cat.categories:
        mean_values.append(df[df['cate'] == i].val.mean())  # getting mean values for each category

    n_categories = sorted(range(len(mean_values)), key=lambda k: mean_values[k])  # getting index of sorting

    df['cate'] = df.cate.cat.rename_categories(n_categories)

    nvar1 = df.cate.tolist()
    nvar2 = df.val.tolist()

    return 1 - abs(stats.pearsonr(nvar1, nvar2)[1])


def f_sc_pearson(var1, var2, var1type, var2type):
    #Switching categories to obtain the best correlation values

    if var1type == CATEGORICAL or var1type == BINARY:
        var1 = [int(i) for i in var1]
        df = pd.DataFrame([pd.Categorical(var1), var2], columns=['cate', 'val'])
    elif var2type == CATEGORICAL or var2type == BINARY:
        var2 = [int(i) for i in var2]
        df = pd.DataFrame([pd.Categorical(var2), var1], columns=['cate', 'val'])
    else:
        return abs(stats.pearsonr(var1, var2)[0])

    mean_values = []  # Per category

    for i in df.cate.cat.categories:
        mean_values.append(df[df['cate'] == i].val.mean())  # getting mean values for each category
    n_categories = sorted(range(len(mean_values)), key=lambda k: mean_values[k])  # getting index of sorting

    df['cate'] = df.cate.cat.rename_categories(n_categories)

    nvar1 = df.cate.tolist()
    nvar2 = df.val.tolist()

    return abs(stats.pearsonr(nvar1, nvar2)[0])


def f_chi2_test(var1, var2, var1type, var2type):
    values1, values2 = bin_variables(var1, var1type, var2, var2type)
    contingency_table = confusion_mat(values1, values2)

    if contingency_table.size > 0 and min(contingency_table.shape) > 1:
        chi2, pval, dof, expd = stats.chi2_contingency(contingency_table)
        return 1 - pval
    else:
        return 0


def f_mutual_info_score(var1, var2, var1type, var2type):
    values1, values2 = bin_variables(var1, var1type, var2, var2type)
    return metrics.adjusted_mutual_info_score(values1, values2)


def f_bf_mutual_info_2d(var1, var2, var1type, var2type):
    return mi.mutual_information_2d(var1, var2)


def f_bf_mutual_info_mat(var1, var2, var1type, var2type):
    var1 = [float(i) for i in var1]
    var2 = [float(i) for i in var2]

    return mi.mutual_information(
        (numpy.reshape(numpy.asarray(var1), (len(var1), 1)), numpy.reshape(numpy.asarray(var2), (len(var2), 1))))


def f_corr_CramerV(var1, var2, var1type, var2type):
    values1, values2 = bin_variables(var1, var1type, var2, var2type)
    contingency_table = confusion_mat(values1, values2)

    if contingency_table.size > 0 and min(contingency_table.shape) > 1:
        return cramers_corrected_stat(contingency_table)
    else:
        return 0


def f_lp_indep_c(var1, var2, var1type, var2type):
    return lp.predict_indep(var1, var2)


def f_fsic(var1, var2, var1type, var2type):
    alpha = 0.01

    # Random seed
    seed = 1

    # J is the number of test locations
    J = 1

    # There are many options for the optimization.
    # Almost all of them have default values.
    # Here, we will list a few to give you a sense of what you can control.
    op = {
        'n_test_locs': J,  # number of test locations
        'max_iter': 200,  # maximum number of gradient ascent iterations
        'V_step': 1,  # Step size for the test locations of X
        'W_step': 1,  # Step size for the test locations of Y
        'gwidthx_step': 1,  # Step size for the Gaussian width of X
        'gwidthy_step': 1,  # Step size for the Gaussian width of Y
        'tol_fun': 1e-4,  # Stop if the objective function does not increase more than this
        'seed': seed + 7  # random seed
    }

    try:
        var1 = [float(i) for i in var1]
        var2 = [float(i) for i in var2]

        pdata = data.PairedData(numpy.reshape(numpy.asarray(var1), (len(var1), 1)),
                                numpy.reshape(numpy.asarray(var2), (len(var2), 1)))
        tr, te = pdata.split_tr_te(tr_proportion=0.5, seed=seed + 1)
        # Do the optimization with the options in op.
        op_V, op_W, op_gwx, op_gwy, info = it.GaussNFSIC.optimize_locs_widths(tr, alpha, **op)
        nfsic_opt = it.GaussNFSIC(op_gwx, op_gwy, op_V, op_W, alpha)
        results = nfsic_opt.perform_test(te)

        print('OK')
        if results['h0_rejected']:
            return 1 - results['pvalue']
        else:
            return results['pvalue']
    except ValueError:
        print('VE')
        # print(var1,var2)
        # print(len(var1),len(var2))
        return 0
    except AssertionError:
        print('AE')
        # print(var1,var2)
        # print(len(var1),len(var2))

        return 0


dependency_functions = [f_pearson,
                        f_abspearson,
                         f_pval_pearson,
                         f_chi2_test,
                         f_mutual_info_score,
                         f_corr_CramerV,
                         f_lp_indep_c,
                        # f_fsic,
                         f_bf_mutual_info_2d,
                         f_bf_mutual_info_mat,
                         f_sc_pearson,
                         f_sc_pval_pearson
                        ]


def process_job_parts(part_number, index):
    data = pd.read_csv(inputdata + 'p' + str(part_number) + '.csv', sep=';')

    # data.columns=['SampleID', 'A', 'B', 'A-Type', 'B-Type', 'Pairtype']
    results = []

    for idx, row in data.iterrows():
        var1 = row['A'].split()
        var2 = row['B'].split()
        var1 = [float(i) for i in var1]
        var2 = [float(i) for i in var2]

        #res = f_mutual_info_score(var1, var2, row['A-Type'], row['B-Type'])
        res = dependency_functions[index](var1, var2, row['A-Type'], row['B-Type'])
        '''Debugging why Mutual Info works bad w/ some pairs ''' #ToDo : Remove section

        results.append([res, row['Pairtype'],row['A-Type'], row['B-Type'],row['SampleID']])

    r_df = pd.DataFrame(results, columns=['Target', 'Pairtype','A-Type','B-Type','SampleID'])
    # r_df = r_df.sort_values(by='Target',ascending=False)
    # r_df.ix[(r_df.index>300) | (r_df['Pairtype']=='O'),['A','B']]=0
    sys.stdout.write('Writing results for ' + crit_names[index][:4] + '-' + str(part_number) + '\n')
    sys.stdout.flush()
    r_df.to_csv(inputdata + crit_names[index][:4] + '-' + str(part_number) + '.csv', sep=';', index=False)


def process_job(index):
    data = pd.read_csv(inputdata + '_pairs.csv', sep=',')
    pub_info = pd.read_csv(inputdata + '_publicinfo.csv', sep=',')
    target = pd.read_csv(inputdata + '_target.csv', sep=',')
    target.columns = ['SampleID', 'T1', 'Pairtype']
    data = pd.merge(data, pub_info, on=['SampleID'])
    data = pd.merge(data, target, on=['SampleID'])

    # data.columns=['SampleID', 'A', 'B', 'A-Type', 'B-Type', 'Pairtype']
    results = []

    for idx, row in data.iterrows():
        var1 = row['A'].split()
        var2 = row['B'].split()
        var1 = [float(i) for i in var1]
        var2 = [float(i) for i in var2]

        res = dependency_functions[index](var1, var2, row['A type'], row['B type'])
        results.append([res, row['Pairtype'], row['A type'], row['B type']])

    r_df = pd.DataFrame(results, columns=['Target', 'Pairtype', 'A type', 'B type'])
    sys.stdout.write('Writing results for ' + crit_names[index][:4] + '\n')
    sys.stdout.flush()
    r_df.to_csv(inputdata + '_' + crit_names[index][:4] + '.csv', sep=';', index=False)

if __name__ == '__main__':
    for idx_crit, name in enumerate(crit_names):
        print(name)
        part_number = 1
        pool = Pool(processes=max_proc)

        while os.path.exists(inputdata + 'p' + str(part_number) + '.csv'):
            print(part_number)
            pool.apply_async(process_job_parts, args=(part_number, idx_crit,))
            part_number += 1
            time.sleep(1)
        pool.close()
        pool.join() # For data in parts'''
        #print('Begin ' + name)
        #process_job(idx_crit)

        # Merging file
        if os.path.exists(inputdata + crit_names[idx_crit][:4] + '-1' + '.csv'):
            with open(inputdata + crit_names[idx_crit][:4] + '.csv', 'wb') as mergefile:
                merger = csv.writer(mergefile, delimiter=';', lineterminator='\n')
                merger.writerow(['Target', 'Pairtype','A-Type','B-Type','SampleID'])
                for i in range(1, part_number):

                    with open(inputdata + crit_names[idx_crit][:4] + '-' + str(i) + '.csv', 'rb') as partfile:
                        reader = csv.reader(partfile, delimiter=';')
                        header = next(reader)
                        for row in reader:
                            merger.writerow(row)
                    os.remove(inputdata + crit_names[idx_crit][:4] + '-' + str(i) + '.csv') #For data in parts
            print("Reorder data")
            outputdata=pd.read_csv(inputdata + crit_names[idx_crit][:4] + '.csv',sep=';')
            outputdata.columns=['Target', 'Pairtype','A-Type','B-Type','SampleID']
            outputdata=outputdata.sort_values(by='Target',ascending=False)
            outputdata.to_csv(inputdata + crit_names[idx_crit][:4] + '.csv',sep=';',index=False)