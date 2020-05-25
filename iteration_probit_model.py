#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created january 2020
    by Juliette MILLET
    script to compute a probit model based on output file with one line = one stimuli-individua and this time with a sampling of answers to have a equilibrated probit model
"""

import pandas as pd
from statsmodels.formula.api import probit
import numpy as np
import random as rd

def get_dico_corres_file(data_file):
    dico ={}
    f = open(data_file, 'r')
    ind = f.readline().replace('\n', '').split(',')
    count = 0
    for line in f:

        newline = line.replace('\n', '').split(',')
        if newline[ind.index('filename')] in dico:
            dico[newline[ind.index('filename')]].append(count)
        else:
            dico[newline[ind.index('filename')]] = [count]
        count += 1
    f.close()
    return dico


def sample_lines(dico_line_files):
    # we sample three results per filename
    list_lines = []
    for filename in dico_line_files:
        if 'EN' in filename:
            list_lines = list_lines + [dico_line_files[filename][rd.randrange(0,stop= len(dico_line_files[filename]))],
                                       dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))],
                                       dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))]]
    return list_lines




def model_probit_binarized(data_file,  model, lines_sampled): # for the model, you have to add the +
    #print(lines_sampled)
    data = pd.read_csv(data_file, sep=',', encoding='utf-8')
    #print(data)
    data = data.iloc[lines_sampled]
    #print(data)
    #print(data)
    data['binarized_answer']  = (data['binarized_answer']+ 1.)/2 # we transform -1 1 into 0 1


    # we fit the probit model
    model_probit = probit("binarized_answer ~ TGT_first_code + nb_stimuli + C(individual) " + model, data)
    result_probit = model_probit.fit()
    return model_probit.loglike(result_probit.params)

def iteration_model(filename, nb_it, outfile):
    dico_lines = get_dico_corres_file(filename)

    list_names = ['articulation', 'babelmulti', 'fishermono', 'fishertri', 'deepspeech', 'dpgmm', 'mfccs']
    out = open(outfile, 'w')
    out.write('nb,' + ','.join(list_names) + '\n')
    for i in range(nb_it):
        out.write(str(i))
        # we sample
        list_sampled = sample_lines(dico_lines)
        list_log = []
        try:
            for mod in list_names:
                print(mod)
                log = model_probit_binarized(data_file=args.file_humans, model='+ ' + mod, lines_sampled=list_sampled)
                list_log.append(str(log))
            out.write(','.join(list_log))
            out.write('\n')
        except:
            continue

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to evaluate the predictions of output from humans by model\'s delta values (with resampling of humans results)')
    parser.add_argument('file_humans_models', metavar='f_do', type=str,
                        help='file with human outputs and models\' delta values')
    parser.add_argument('outfile', metavar='f_do', type=str,
                        help='output file with log likelihood answers (one line = one sampling)')
    parser.add_argument('nb_it', metavar='f_do', type=int,
                        help='nb of sampling you want to perform')

    args = parser.parse_args()


    iteration_model(filename=args.file_humans, nb_it=args.nb_it, outfile=args.outfile)
