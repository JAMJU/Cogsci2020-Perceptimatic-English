#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created december 2019
    by Juliette MILLET
    script to compute a probit model based on output file with one line = one stimuli-individua

"""
from statsmodels_2.statsmodels.discrete import discrete_model as stat
import pandas as pd
from statsmodels.formula.api import probit
import numpy as np



def get_dico_for_model(distance_file_model):
    f = open(distance_file_model, 'r')
    separator = '\t'
    ind = f.readline().replace('\n', '').split(separator)
    dico = {}
    for line in f:
        new_line = line.replace('\n', '').split(separator)
        filename = new_line[ind.index('filename')]
        distance = new_line[ind.index('distance')]
        dico[filename] = distance
    return dico

def add_column_distance_models(list_distance_files, list_names, f_in, f_out):
    list_dico = []
    for file in list_distance_files:
        list_dico.append(get_dico_for_model(file))

    f = open(f_in, 'r')
    f_out = open(f_out, 'w')

    ind = f.readline().replace('\n', '').split(',')
    f_out.write(','.join(ind) + ',' + ','.join(list_names) + '\n')
    for line in f:
        new_line = line.replace('\n', '').split(',')
        filename = new_line[ind.index('filename')]
        distance_results = [dic[filename] for dic in list_dico]
        final_list = new_line + distance_results

        f_out.write(','.join(final_list) + '\n')
    f.close()
    f_out.close()



def model_probit_binarized(data_file, data_out, model): # for the model, you have to add the +
    data = pd.read_csv(data_file, sep=',', encoding='utf-8')

    #print(data.head())

    #print(data['TGT_first_code'])
    #print(data['nb_stimuli'])
    #print(data[['TGT_first_code', 'nb_stimuli']])

    data['binarized_answer']  = (data['binarized_answer']+ 1.)/2 # we transform -1 1 into 0 1


    # we fit the probit model
    model_probit = probit("binarized_answer ~ TGT_first_code + nb_stimuli " + model, data)
    result_probit = model_probit.fit()
    # parameters
    #print('Parameters',result_probit.params)
    # summary of the model
    #print('Summary',result_probit.summary())
    # we get the marginal effect
    #probit_margeff = model_probit.get_margeff()
    print('Loglik', model_probit.loglike(result_probit.params))



    #print('Margeff',probit_margeff.summary())
    # how well are we predicting ?
    #print('Prediction table', model_probit.pred_table())
    # compute the residuals
    # predict
    #predictedValues = np.asarray([[k] for k in result_probit.predict(data)])
    #print('predicted values',predictedValues[-20:])

    #yData = data.as_matrix(columns = ['binarized_answer'])
    #print('data', yData[-20:])
    #res = yData - predictedValues
    #print('residuals',res[-20:])
    #res = [k[0] for k in res]
    #residuals = pd.DataFrame({'residual_probit_TGT_first_nb_stimuli_individual':res})
    # add the residual to the original data
    #data = data.join(residuals)
    #print('final data',data[-20:])
    # save residuals
    #data.to_csv(data_out, sep= ',')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from humans vs model\'s outputs')
    parser.add_argument('file_humans', metavar='f_do', type=str,
                        help='file with outputs humans t give')
    #parser.add_argument('file_humans_out_interm', metavar='f_do', type=str,
    #                    help='file with outputs humans with old answers produced')
    parser.add_argument('file_humans_out', metavar='f_do', type=str,
                        help='file with outputs humans with old answers and probit model residuals')

    args = parser.parse_args()
    list_names = ['articulation', 'babelmulti', 'fishermono', 'fishertri', 'deepspeech', 'dpgmm', 'mfccs']
    # if you want to add the model's results to the humans results
    """add_column_distance_models(list_distance_files=['results_our_test/articulation_cosine.csv',
                                                    'results_our_test/bottleneck_babelmulti.csv',
                                                    'results_our_test/bottleneck_fishermono.csv',
                                                    'results_our_test/bottleneck_fishertri.csv',
                                                    'results_our_test/deepspeech_layer5.csv',
                                                    'results_our_test/dpgmm_english_kl.csv',
                                                    'results_our_test/mfcc_cmn_cosine_norm.csv'],
                               list_names=['articulation', 'babelmulti', 'fishermono', 'fishertri', 'deepspeech', 'dpgmm','mfccs'], f_in=args.file_humans, f_out=args.file_humans_out)"""

    #add_column_old_answer(args.file_humans, args.file_humans_out_interm)
    for mod in list_names:
        print(mod)
        model_probit_binarized(data_file=args.file_humans, data_out=args.file_humans_out, model='+ ' + mod)

