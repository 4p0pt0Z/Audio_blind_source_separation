# Audio_blind_source_separation
Master thesis Fall 2018: Neural Network based Audio Blind source Separation for Noise Suppression @ EPFL &amp; Logitech


### Requirements (conda install)
* Python 3.7
* numpy
* scipy
* pytorch
* matplotlib
* scikit-learn
* librosa (installed with pip)
* pandas
* jupyter lab

### Replicate Blind Source Segmentation on weakly labelled data paper
The first step of this work is to build a framework for audio blind source separation (ABSS) working on weakly labelled data, inspired from the results described in the paper [Sound Event Detection and Time-Frequency Segmentation from Weakly Labeled Data](https://arxiv.org/abs/1804.04715).

#### Dataset 
The framework works on weakly labelled data, however for the results presented in the paper the authors did not work directly with a weakly labelled dataset, rather they created weakly labelled segments from an existing dataset: the [DCASE2013 sound event detection dataset (subtask 2)](http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/description.html). The first step is to build a similar dataset to evaluate our model

This can be done with the script `generate_weakly_labeled_data_from_DCASE2013.py`. The scripts generates audio mix from the events in the DCASE2013 dataset, as well as a ".csv" file containing weak labels for the mixes.

| Parameters | input format | Default value | Description |
| :--------: | :----------: | :-----------: | :---------: |
| length | ]0.0, inf) | 10 | Length (in seconds) of the audio mix to generate. The length of a mix should be bigger than the length of individual components. |
| max_event | [1, inf) | 4 | Maximal number of audio events to include in a mix |
| overlap | {True, true, 1, False, false, 0}| False | Control if the events can overlap in the mix, or should be separated |
| white_noise_ratio | [0.0, 1.0] | 0.05 | Ratio between the mix mean value and the added white noise mean value |
| sampling_rate |  {8000, 16000, 44100} | 16000 | The sampling rate to use for the mix audio format |
| n_files | [1, inf] | 2000 | Number of mix to generate |
| DCASE_2013_stereo_data_folder | str |  | Path to the DCASE2013 sound event detection (subtask 2) dataset |
| output_folder | str |  | Path to the folder where the mix will be saved | 

