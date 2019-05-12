#VoxCeleb1 
* pip install speechpy
* contains over 100,000 utterances for 1,251 celebrities, extracted from videos uploaded to YouTube. 
# Verification split dev test
* of speakers	1,211	40
* of videos	21,819	677
* of utterances	148,642	4,874
# Identification split dev test
* of speakers	1,251	1,251
* of videos	21,245	1,251
* of utterances	145,265	8,251


# Dataset 
* Link to dataset: https://drive.google.com/drive/folders/1__Ob2AUuAdzVDRCVhKtKSvGmTKHUKTuR
* Username: voxceleb1904
* Password: 9hmp7488
# Project files
* Constants.py: Contains all the constants for preprocessing the data.
* vox1_meta.csv: Contains the meta data of the dataset in a csv format.
* preprocess/utils.py: This file includes routines for basic signal processing including framing and computing power spectra. Author: James Lyons 2012