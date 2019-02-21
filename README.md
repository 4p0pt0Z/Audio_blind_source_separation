# Audio_blind_source_separation
Master thesis Fall 2018: Neural Network based Audio Blind source Separation for Noise Suppression @ EPFL &amp; Logitech


### Requirements (conda install)
* Python 3.7
* numpy
* scipy
* pytorch
* matplotlib
* scikit-learn
* librosa
* pandas
* jupyter lab (visualization)
* seaborn (visualization)
* node-js (vizualization) and extension for jupyter lab [here](https://github.com/matplotlib/jupyter-matplotlib)
* h5py
* pyyaml
* mir_eval (installed with pip) for audio separation performance measurements
* iterative-stratification (pip install, for stratified multilabel data split)

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
| white_noise_ratio | [0.0, 1.0] | 0.05 | Ratio between energy of audio events in the mix and background white noise |
| sampling_rate |  {8000, 16000, 44100} | 16000 | The sampling rate to use for the mix audio format |
| n_files | [1, inf] | 2000 | Number of mix to generate |
| DCASE_2013_stereo_data_folder | str |  | Path to the DCASE2013 sound event detection (subtask 2) dataset |
| output_folder | str |  | Path to the folder where the mix will be saved | 

#### Training framework
The training framework is divided in 3 major parts, each in a separate file:
* `data_set.py`: Implements how the audio files are loaded from disk and how the Frequency-Time representation of the audio is extracted. Also divides the data into training, development and validation set.
* `model.py`: Implements the model as a torch.nn.module.
* `train.py`: Implements the training loop and evaluation. Handles the saving and loading of models from checkpoint files.

In each of these file, there is typically one class that is going to be used. For instance, if we want to train a VGG-like model on mixes build from the DCASE2013 data set, we are going to use the `DCASE2013_remixed_data_set` class in `data_set.py`, the `VGG_like_CNN` class in `model.py`, and the `TrainingManager` in `train.py`.  
To train with these classes, you must pass their identifier to the main script. The identifier is often the name of the class ; it is the value look up in the function `find_...` in each of these file, which return the corresponding class.

The rest of the available command line arguments are any parameters appearing in the `default_config` method of the classes that have been selected.

TODO: describe the arguments.

Training can be launched with a command similar to :  
`python -m main --mode train \ `  
`--model_type VGG_like_CNN \ `  
`--drop_out_probs 0.3 0.4 0.5 0.5 0.5 0.0 \ `  
`--classification_mapping GWRP \ `  
`--data_set_type DCASE2013_remixed_data_set \  ` 
`--data_folder Datadir/remixed_DCASE2013_2k  \ `  
`--loss_f BCE \ `  
`--scale_transform True \ `   
`--use_batch_norm True \ `  
`--n_epochs 500 \ `  
`--scheduler_type multiStepLR --scheduler_gamma 1.0 --scheduler_milestones 150 300 \ `   
`--test_every 5 \ `  
`--learning_rate 0.0001 \ `    
`--weight_decay 0.0 \ `   
`--metric f1-score \ `   
`--save_path results/models/test1.ckpt`  
