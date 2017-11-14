# Speaker-identification-using-GMMs

It uses popular ML technique GMM to train speaker identification models.

Data-set:

Training corpus : It has been developed from audios taken from 'on-line VoxForge speech database' and consists of 5 speech       utterances for each speaker, spoken by 34 speakers (i.e, 20-30 seconds/speaker).

Test corpus: This consists of remaining 5 unseen utterances of the same 34 speakers taken in train corpus. All audio files are of 3-5 seconds duration and are sampled at 16000 Hz.

The documentation/tutorial for task in this repository can be read from this blog.

https://appliedmachinelearning.wordpress.com/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/



# Installation

You need to install only these (tested with):
    1. Install Anaconda 64 bit Python 2.7 version. (https://www.continuum.io/downloads)
    2. pip install python_speech_features. (for extracting MFCC features)

Also, Download the data-set from the provided link in the beginning of blog.

Note : Directory path used for train and test corpus in code train_models.py and test_speaker.py needs to be properly set depending upon the path where you download the data-set.
