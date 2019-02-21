# Audio_blind_source_separation
Master thesis Fall 2018: Neural Network based Audio Blind source Separation for Noise Suppression @ EPFL &amp; 
Signal Processing Lab 2 (LTS2) at EPFL

Audio source separation consists in separating audio signal coming from different sources from a recording containing 
several such sources (audio mixtures). For the thesis, we worked on mono-channel weakly labelled recordings.

Weakly labelled recordings are audio recordings for which the content is known (ie: we know there is speech and traffic 
noise in this mixture) but the separate sources in the recording are not available. In order to use such data in a 
machine learning framework, a training procedure using 2 models has been proposed in 

[1] Qiuquiang Kong et al. "A joint separation-classification model for sound event detection of weakly labelled data". 
In: CoRR abs/1711.03037 (2017à. arXiv: 1711.03037 URL: http://arxiv.org/abs/1711.03037

[2] Qiuqiang Kong et al. "Sound Event Detection and Time-Frequency Segmentation from Weakly Labelled Data". In: CoRR 
abs/1804.04715 (2018). arXiv:1804.04715. URL: http://arxiv.org/abs/1804.04715

The audio separation is based on spectral masking: a "mask" model is trained to produce masks that select each sources 
in the signal in Time-Frequency representation. To be able to train on weakly labelled recordings, the produced masks
are then passed to a second model that performs classification. The two models can thus be trained jointly using the 
weak labels of the recordings.

![alt text](https://github.com/4p0pt0Z/Audio_blind_source_separation/blob/master/diagram_audio_separation_from_weakly_labelled_data.png
"Audio Separation Training from weakly labelled data")


This repository is a pytorch implementation of an audio source separation framework based on the method described in 
[1, 2]. It contains code for :
* generating audio mixtures from the 
[DCASE2013 Sound Event Detection data set](http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/description.html) (task 2)
* generating audio mixtures from [Audioset](https://research.google.com/audioset/) recordings re-labelled by 
CloudFactory
* train a separation model as described in [1,2] with a CNN block-based architecture for the model producing separation
masks and several kind of classifier models, using 3 possible data sets.
* Evaluate the separation performances of the models on validation sets. 


### Data sets
3 data sets have been used for training models:
* [DCASE2013 Sound Event Detection data set](http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/description.html) (task 2)
The audio events of this set (look for DCASE2013_SED/singlesounds_stereo/singlesounds_stereo) can be mixed using the 
`generate_weakly_labelled_audio_mixtures_from_DCASE2013.py` script to create audio mixtures.
* [TUT Rare Sound Event Data set](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-rare-sound-event-detection)
This data set is used in [1] and the mixtures are create using the 
[code from the authors](https://github.com/qiuqiangkong/ICASSP2018_joint_separation_classification). (`runme.sh` 
produces the features in `/mixed_audio` and the labels in `/mixed_yaml`)
* [Audioset](https://research.google.com/audioset/) Recordings from a few classes from Audioset have been re-labelled by 
[CloudFactory](https://www.cloudfactory.com/). These new labels can be used to generate smaller recordings with labels 
for training using `generate_audioset_segments.py` 


### Code Framework
The code framework architecture is as follows:

![alt text](https://github.com/4p0pt0Z/Audio_blind_source_separation/blob/master/diagram_readme.png "Code Organization")

The main executable parses the user arguments and launches 3 possible behavior:
* train: a model is trained, either from scratch or using a saved model in a checkpoint and resuming the training
* evaluate: a model is loaded from checkpoint and the classification performances are evaluated on the validation set
* separate: a model is loaded from checkpoint and used to perform audio source separation on the files of the validation
set. The performances of the separation are then measured.

For the training and evalution part, the class TrainingManager is used. This class has a separation model and a data set 
as member, and implements the training, evaluation and saving routines.

The Separation model is composed of 3 parts: 
* PCEN layer (optional). A layer implementing the trainable Per-channel Energy Normalization processing of a spectrogram
* Mask model: A CNN in charge of producing the separation masks
* Classifier model: A model trained to classify the separation masks.

The common routines of audio processing (STFT, ISTFT, mel scaling, etc...) are implemented in AudioDataSet. This class 
also inherits from torch.data.DataSet and handles the data access during training. 3 specialized classes inherit from 
AudioDataSet, implementing specific methods for handling the audio mixtures from 3 data sets.

The AudioSeparator class is used to use a trained model to perform audio separation. It can be used to perform audio 
separation using a trained model, for all files in a validation data set. Then, the audio separation performances can 
be measured.


### Install
A conda environement is used to be able to manage and easily replicate the development environment for this project.
Miniconda can be installed from [here](https://conda.io/en/latest/miniconda.html).

The dependencies for this project are indicated in environment.yml.
To build a new environment for this project use:
`conda env create -f environment.yml`

Then activate the environment with `conda activate abss` (source activate abss on Linux)

Unfortunately, the library `mir_eval` used to compute separation performances does not have conda packages.
Once the `abss` environment is ready and activated, you can install mir_eval using `pip install mir_eval`.  


### Command line arguments
The execution mode must be selected with the `--mode` argument. Choice are `train`, `evaluate` and `separate`.

For training a model, 3 arguments are required: the type of the data set to use, the type of the mask model and the 
type of the classifier model. The available types are reported in the functions `find_data_set_class`, 
`find_mask_model_class`, and `find_classifier_model_class` respectively in the files `data_set.py`, `mask_model.py` and
`classifier_model.py`. These arguments must be passed if the execution mode is `train`.


Most tunable hyper-parameters of the audio separation framework are available as command line arguments. The tunable 
arguments are always accessible through the `default_config` method of class. To change the parameters of an object, 
look at its `default_config` method, find the name of the parameter and use it as command line argument.

Example: The TrainingManager class handles the training of a model. Its `default_config` method contains an entry for 
the `learning_rate` parameter. To use a learning rate of 0.1: use `--learning_rate 0.1` as a command line argument.

Exceptions: The separation model contains 2 separated models: the mask model and the classifier model. These models can 
have the same CNN architecture, therefore in order to avoid confusion between the parameters of these models, the 
prefixes `mask_` or `class_` must be used.

Example: We use a classifier model of type `GlobalWeightedRankPooling2d`. To change the value of its parameter `dc` to 
0.9, use: `--class_dc 0.9`.  
To change the number of feature maps to use in the mask model to 48 for the first 3 layers, use: 
`--mask_conv_i_c 48 48 48`.

Full example:

```
python -m main --mode train 
--mask_model_type VGGLikeMaskModel 
--mask_n_blocks 3 
--mask_conv_i_c 1 64 64 
--mask_conv_o_c 64 64 10 
--mask_conv_k_f 5 5 5 
--mask_conv_k_t 5 5 5 
--mask_conv_s_f 1 1 1 
--mask_conv_s_t 1 1 1 
--mask_conv_p_f 2 2 2 
--mask_conv_p_t 2 2 2 
--mask_conv_groups 1 1 1 
--mask_dropout_probs 0.1 0.1 0.0 
--mask_activations lr lr sig 
--classifier_model_type GlobalWeightedRankPooling2d 
--data_set_type ICASSP2018JointSeparationClassificationDataSet 
--data_folder path_to_"packed_features/logmel" 
--yaml_file path_to_"mixed_yaml/" 
--audio_folder path_to_"/mixed_audio/" 
--n_epochs 5 
--test_every 2 
--learning_rate 0.0001 
--metric roc_auc_score --average weighted 
--use_cuda True --gpu_no 1 
--save_path test_model.ckpt
```
 