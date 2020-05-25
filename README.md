# Cogsci2020-Perceptimatic-English
This github contains the dataset and code of the experiments described in [3]. Please cite the paper if you use our dataset/code. The dataset used is the English part of the Perceptimatic dataset (only English participants results).
### General environment required
* python 3.6/7
* numpy
* scipy
* pandas
* statsmodels

# Dataset
## Cleaned stimuli
We provide the stimuli we used on the form of onset and offset for the 2017 Zerospeech one second across speaker French and English stimuli (the wav files can be downloaded here: https://download.zerospeech.com/). 
The onset, offset and labels of the cleaned French triphones are in DATA/all_aligned_clean_french.csv, the English are in DATA/all_aligned_clean_english.csv. The files have the following columns:

| index	| #file |	onset|	offset |	#phone	| prev-phone | next-phone	| speaker|
| --- | --- | --- | --- | --- | --- | --- | --- |

index is how, combined with the language, we refer to each triphone in the rest of the files,
 '#file in the original 2017 ZeroSpeech wavfile, onset and offset are beginning and end of triphone in second,
 '#phone is the centre phone, prev-phone and next-phone are the surrounding phones, speaker is the reference number of the speaker.
 
 We provide the list of triplets used for the humans and models experiment in the file DATA/all_triplets.csv:
 
 filename|	TGT	|OTH|	prev_phone	|next_phone|	TGT_item	|OTH_item	|X_item	|TGT_first	|speaker_tgt_oth	|speaker_x|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

filename is the name of the file containing the triplet (used for the humans experiment), it can be seen as the id of the triplet (it contains FR if the triplet is in French, and EN if the triplet is in English). 
If we consider the triplet as an A, B and X stimuli, with A and X in the same category, in the file, TGT is the centre phone of 
A and X, OTH is the center phone of B, prev_phone an next_phone are the surrounding phones. TGT_item, OTH_item and X_item refer 
to the indexes of the stimuli used as A, B and X. TGT_first indicate if A comes first in the file or not. We can note that each set of three extracted 
triphones appears in four distinct items, corresponding to orders AB--A (that is, X is another instance of the three-phone sequence A), BA--B, AB--B, and BA--A. 


## Human test
We give all the code to perform the human experiments, and we provide the triplet used on demand (contact juliette.millet@cri-paris.org)
For each triplet, the delay between first and second stimuli is 500 milliseconds, and between second and third 650 milliseconds, 
as pilot subjects reported having difficulty recalling the reference stimuli when the delays were exactly equal.

## Human results

Human results are in DATA/humans_and_models.csv, this file also contains delta values for the models we evaluate in the paper. 
Each line corresponds to a couple (triplet, participant). This file has the following columns:

* individual : code of the individual (unique among one language group)
* language : language group of the participant (French speaking or English speaking: here we only have english)
* filename: triplet id (see section on the cleaned stimuli, contains EN if is a english triplet, FR if is a french triplet)
* TGT: same as in DATA/all_triplets.csv
* OTH: same as in DATA/all_triplets.csv
* prev_phone: same as in DATA/all_triplets.csv
* next_phone: same as in DATA/all_triplets.csv
* TGT_item: same as in DATA/all_triplets.csv
* OTH_item: same as in DATA/all_triplets.csv
* X_item: same as in DATA/all_triplets.csv
* TGT_first: same as in DATA/all_triplets.csv (True or False)
* speaker_tgt_oth: same as in DATA/all_triplets.csv
* speaker_x: same as in DATA/all_triplets.csv
* correct_answer: human answer, either -3, -2, -1, 1, 2 or 3. If it is negative then the participant has chosen the OTH item instead of the (correct) TGT item.
* binarized_answer: binarized version of correct_answer : -1 if correct_answer < 0, 1 otherwise
* nb_stimuli: number of triplets heard by the participants with this triplet included (between 1 and ~190)
* TGT_first_code: 1 if TGT_first is True, 0 otherwise
* language_code: 1 for French participants, 0 for English partcipants (in the data given, we only include English participants)

# Analysis code
In this section we describe all the steps to evaluate any model with our methods.

## Extracting features from your model
First of all you need to extract the model you want to evaluate's representations of the 2017 ZeroSpeech one second stimuli. The original wavfiles can be downloaded here: https://download.zerospeech.com/. 

Our evaluation system requires that your system outputs a vector of feature values for each frame. For each utterance in the set (e.g. s2801a.wav), an ASCII features file with the same name (e.g. s2801a.fea) as the utterance should be generated with the following format (separator = ' '):

| | | | |
| --- | --- | --- | --- |
| time1  | val1  |  ...  | valN |
| time2 | val1 |    ... | valN |

example:

| | | | | | |
| --- | --- | --- | --- | --- | --- |
|0.0125 | 12.3 | 428.8 | -92.3 | 0.021 | 43.23 |
|0.0225 | 19.0 | 392.9 | -43.1 | 10.29 | 40.02 |

Note

The time is in seconds. It corresponds to the center of the frame of each feature. In this example, there are frames every 10ms and the first frame spans a duration of 25ms starting at the beginning of the file, hence, the first frame is centered at .0125 seconds and the second 10ms later. It is not required that the frames be regularly spaced.
  
## Extracting delta values from features and computing ABX accuracies (PEB)

Once your features are in the right format, you need to put them in a global folder (called M here), put your English features in M/english/, and your French features in M/french/. Then do:

`python script_get_file_distance.py M/ DATA/all_triplets.csv $file_delta.csv$ $distance$ DATA/english/all_aligned_clean_english.csv DATA/french/all_aligned_clean_french.csv False`

`$distance$` can be 'euclidean', 'kl' or 'cosine': it is the distance you want to use for the DTW. This can adapted if your representations are not numerical.

This file creates a csv file with the delta distances of your model for each triplet ($file_delta.csv$). The script also print the ABX accuracies over the dataset (for English stimuli and French stimuli).

In order to perform the rest of the analysis easily, you can add your model delta values to our existing file containing human results, and all the model we evaluated 's delta values. To do that you need to do:

`python concatenate_results $file_delta.csv$ $name_model$ DATA/humans_and_models.csv $file_all.csv$`

`$name_model$` if the name of the new column you add to the original file containing human results. You obtain a file (`$file_all.csv$`) containing all the data in humans_and_models.csv and the delta values you computed.

## Comparing your model with humans results (only English speaking participants) and other models.


# Extracting features used in the paper
delta values obtained for the different models can be found in the file DATA/humans_and_models.csv, one column per model with the codenames given in the paper.
 But if you want to extract the features yourself in order to recompute the delta values, you can follow these instructions:
 
 ## Deepseech
 
 
 
 ## MFCCs
The MFCCs used in the paper were extracted with Kaldi toolkit, using the default paramters, adding the first and second derivatives for a total of 39 dimensions, and we applymean-variance normalization over a moving 300 milliseconds window. We provide the extracted MFCCs on demand on demand (contact juliette.millet@cri-paris.org)

## Bottleneck features (Multilingual, FisherMono and FisherTri)
We used the Shennong package (https://github.com/bootphon/shennong) to extract Botteneck features described in [2] 

To extract these features you need the Shennong package installed (added to the list of requirements listed above). To extract features from wavfiles in a folder F to a folder G do

`python extract_from_shennong_bottleneck.py F G $Type$`

with $Type$ equal to FisherMono, FisherTri or BabelMulti.

## DPGMM

We use the kaldi toolkit to extract MFCCs and apply the same VTLN than in [1] (the vtln-mfccs can be provided on demand, contact juliette.millet@cri-paris.org), then we  extract the posteriorgrams from the French and English models from [1] we follow the instructions of https://github.com/geomphon/CogSci-2019-Unsupervised-speech-and-human-perception