#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created december 2019
    by Juliette MILLET
   Script to convert features into distance for the triplets we are interested in
"""
import os
import numpy as np
from dtw_experiments import compute_dtw_norm, compute_dtw
from joblib import Parallel, delayed
import itertools
import h5py
import dtw
"""The file we are dealing with has the following indexes: index	#file	onset	offset	#phone	prev-phone	next-phone	speaker"""

def get_frames_rep(folder_soumission, language, filename, time_begin_s, time_end_s):
    f = open(os.path.join(folder_soumission, language, filename + '.fea'), 'r')
    separator = ' ' # to chande depending on the encoding of .fea
    previous_time = 0
    rep = []
    begin = False
    for line in f:
        new_line = line.replace('\n', '').split(separator)

        current_time = float(new_line[0])
        if not begin:
            if time_begin_s > previous_time and time_begin_s <= current_time:
                begin = True

                rep.append([float(a) for a in new_line[1:]])
        else:
            if not (time_end_s >= previous_time and time_end_s < current_time):
                rep.append([float(a) for a in new_line[1:]])
            else:
                f.close()
                out = np.asarray(rep)
                #print(out.shape)
                return out
        previous_time = current_time
    print('Time error')




def get_dictionnary(filename):
    f = open( filename , 'r')
    ind = f.readline().replace('\n', '').split('\t')
    dico_corres = {}
    dico_corres['index'] = ind
    for line in f:
        new_line = line.replace('\n', '').split('\t')
        dico_corres[new_line[ind.index('index')]] = new_line
    f.close()
    return dico_corres

def compute_delta(folder_soumission, dico_corres_english, dico_corres_french, language, indexTGT, indexOTH, indexX, distance, adaptive_average = False):
    ind = dico_corres_english['index']
    if language == 'english':
        dico_corres = dico_corres_english
    else:
        dico_corres = dico_corres_french

    func_to_use = get_frames_rep
    #print(dico_corres[indexTGT][ind.index('#file')])
    TGT = func_to_use(folder_soumission = folder_soumission, language=language,
                         filename=dico_corres[indexTGT][ind.index('#file')],
                         time_begin_s=float(dico_corres[indexTGT][ind.index('onset')]),
                         time_end_s=float(dico_corres[indexTGT][ind.index('offset')]))
    #print(dico_corres[indexOTH][ind.index('#file')])
    OTH = func_to_use(folder_soumission=folder_soumission, language=language,
                         filename=dico_corres[indexOTH][ind.index('#file')],
                         time_begin_s=float(dico_corres[indexOTH][ind.index('onset')]),
                         time_end_s=float(dico_corres[indexOTH][ind.index('offset')]))
    #print(dico_corres[indexX][ind.index('#file')])
    X = func_to_use(folder_soumission=folder_soumission, language=language,
                         filename=dico_corres[indexX][ind.index('#file')],
                         time_begin_s=float(dico_corres[indexX][ind.index('onset')]),
                         time_end_s=float(dico_corres[indexX][ind.index('offset')]))

    if not adaptive_average:
        TGT_X = compute_dtw(x = TGT, y = X, dist_for_cdist=distance, norm_div=True)
        OTH_X = compute_dtw(x = OTH, y = X, dist_for_cdist=distance, norm_div=True)
    else:
        TGT_X = compute_dtw_norm(x=TGT, y=X, dist_for_cdist=distance, norm_div=True)
        OTH_X = compute_dtw_norm(x=OTH, y=X, dist_for_cdist=distance, norm_div=True)
    return OTH_X - TGT_X

def to_parallel(line, ind, folder_soumission, dico_english, dico_french, distance, file_out, adaptive_average = False):
    out = open(file_out, 'a')
    new_line = line.replace('\n', '').split('\t')
    filename = new_line[ind.index('filename')]
    delta = compute_delta(folder_soumission=folder_soumission, dico_corres_english=dico_english,
                          dico_corres_french=dico_french,
                          language='french' if 'FR' in filename else 'english',
                          indexTGT=new_line[ind.index('TGT_item')],
                          indexOTH=new_line[ind.index('OTH_item')], indexX=new_line[ind.index('X_item')],
                          distance=distance, adaptive_average=adaptive_average)
    out.write('\t'.join(new_line) + '\t' + str(delta) + '\n')
    out.close()


def compute_all_results(file_list_triplet, folder_soumission, file_out, distance, english_list, french_list, adaptive_average = False):
    f = open(file_list_triplet, 'r')
    ind = f.readline().replace('\n', '').split('\t')

    dico_english = get_dictionnary(english_list)
    dico_french = get_dictionnary(french_list)

    lines = f.readlines()
    args_f = [line for line in lines]
    f.close()
    print(args_f[0])

    Parallel(n_jobs=-1, backend="threading")(
        map(delayed(to_parallel), args_f, itertools.repeat(ind, len(args_f)),
            itertools.repeat(folder_soumission, len(args_f)), itertools.repeat(dico_english, len(args_f)),
            itertools.repeat(dico_french, len(args_f)), itertools.repeat(distance, len(args_f)), itertools.repeat(file_out, len(args_f)),
            itertools.repeat(adaptive_average,len(args_f))))


def compute_scores(file_out):
    f = open(file_out, 'r')
    count_french = 0
    count_english = 0
    score_french = 0
    score_english = 0
    for line in f:
        new_line = line.replace('\n', '').split('\t')
        score = float(new_line[-1])
        filename = new_line[0]
        if 'FR' in filename:
            score_french += 0. if score > 0 else 1.
            count_french += 1
        else:
            score_english += 0. if score > 0 else 1.
            count_english += 1
    print('accuracy french:', 100. - score_french/float(count_french)*100., 'accuracy english:', 100. - score_english/float(count_english)*100.)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute ABX distances and accuracies for a certain list of triplets')
    parser.add_argument('folder_soumission', metavar='f_do', type=str,
                        help='The file with items you want to check')
    parser.add_argument('file_list', metavar='f_ok', type=str,
                        help='The file with the list of triplets')
    parser.add_argument('file_out', metavar='f_ok', type=str,
                        help='The file out')
    parser.add_argument('distance', metavar='d', type=str,
                        help='distance use in dtw, can be cosine, kl or euclidean')
    parser.add_argument('english_list', metavar='f_ok', type=str,
                        help='file with onset and offset of english triplet')
    parser.add_argument('french_list', metavar='f_ok', type=str,
                        help='The file with the onset and offset of french triplet')
    parser.add_argument('adaptive_average', metavar='h', type=str,
                        help='if use of adaptive average method or not')


    args = parser.parse_args()
    adapt = True if args.adaptive_average == 'True' else False

    #dis = lambda x,y: dtw.kl_divergence(x,y)
    compute_all_results(file_list_triplet=args.file_list, folder_soumission=args.folder_soumission, file_out=args.file_out,
                        distance= args.distance, english_list=args.english_list,
                        french_list=args.french_list, adaptive_average=adapt)
    compute_scores(file_out=args.file_out)