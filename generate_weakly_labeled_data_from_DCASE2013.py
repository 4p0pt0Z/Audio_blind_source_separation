import numpy as np
import argparse
import os
import csv
import librosa
import uuid

from helpers import str2bool


def generate_mixed_files(audio_files, audio_data, n_files, output_folder, length, max_event, overlap, wn_ratio,
                         sampling_rate):
    """
        Generates mixed audio files from the input audio files. The generated files are saved in the output folder,
        with a '.csv' file keeping track of the classes present in each mix.

        Assumption: the length of the mix should be greater than the length of the audio events to put in the mix
    Args:
        audio_files (lst): List of the audio files (eg ["Alert01.wav", ...]
        audio_data (lst): List of numpy arrays storing the audio data for each file
        n_files (int): Number of mixed files to generate
        output_folder (str): Path to the output folder for the generated files (folder is created if not-existing)
        length (float): Length of the mixed audio files
        max_event (int): Maximal number of events to include in a mixed file
        overlap (bool): Whether or not the audio events can overlap in the remixed files
        wn_ratio (float): White noise ratio (with respect to the mean of mixed file)
        sampling_rate (int): Sampling rate for the mixed audio files

    """
    # The 16 classes present in the DCASE2013 sound event detection data set.
    classes = ["alert", "clearthroat", "cough", "doorslam", "drawer", "keyboard", "keys", "knock", "laughter", "mouse",
               "page-turn", "pendrop", "phone", "printer", "speech", "switch"]
    header = ["filename"] + classes

    total_n_events = len(audio_files)
    mixed_audio_length = int(np.ceil(sampling_rate * length))
    # Write the csv file containing the weak label of the mixed segments
    with open(os.path.join(output_folder, "weak_labels.csv"), 'w', newline='') as label_file:
        label_writer = csv.writer(label_file)
        label_writer.writerow(header)

        # Generate the mixed files
        for _ in range(n_files):
            mixed_audio = np.zeros(mixed_audio_length, dtype=np.float)
            n_events = np.random.randint(low=1, high=max_event + 1)  # number of audio to include in this mixed file
            events_idx = np.random.choice(total_n_events, n_events)  # the audio files to include
            if not overlap:
                # The audio files can not overlap in the mix. We have two possibilities: if the audio files to
                # include last more than the required length of the mix, then we include as much of the audio as we
                # can, and simply cut the remaining. If all audio can fit in the mix, then we randomly draw silences
                # duration to separate the audio in the mix
                cum_length = np.sum([audio_data[i].shape[0] for i in events_idx])
                if cum_length > mixed_audio_length:
                    start = 0
                    for idx in events_idx:
                        available_length = np.amin([audio_data[idx].shape[0],
                                                    np.max([mixed_audio_length - (start + audio_data[idx].shape[0]),
                                                            0])])
                        if available_length == 0:
                            break
                        mixed_audio[start: start + available_length] = audio_data[idx][:available_length]
                        start += available_length
                else:
                    # randomly chose a total of silence amount to include between the files, in the available length.
                    total_silence_between_files = np.random.randint(mixed_audio_length - cum_length)
                    silences_length = np.floor(  # Draw silences duration such that they sum up to the total amount.
                        np.random.dirichlet(np.ones(n_events)) * total_silence_between_files).astype(int)
                    start = 0
                    for i, idx in enumerate(events_idx):
                        mixed_audio[start + silences_length[i]: start + silences_length[i] + audio_data[idx].shape[0]] \
                            = audio_data[idx]
                        start += silences_length[i] + audio_data[idx].shape[0]
            else:
                # Overlap is authorized, so we simply draw random start times and copy the entire audio files in the mix
                for idx in events_idx:
                    event_length = np.amin([audio_data[idx].shape[0], mixed_audio_length - 1])  # clip to length of mix
                    start_time = np.random.randint(mixed_audio_length - event_length)
                    mixed_audio[start_time: start_time + event_length] += audio_data[idx][:event_length]

            # Add white noise
            noise = np.random.random(mixed_audio_length)
            mixed_audio += np.mean(mixed_audio) * wn_ratio * noise
            # Save file
            name = uuid.uuid1().hex + '.wav'
            librosa.output.write_wav(os.path.join(output_folder, name), mixed_audio, sampling_rate)
            # save file labels
            labels = [0] * len(classes)
            for idx in events_idx:
                for i, class_name in enumerate(classes):
                    if audio_files[idx].startswith(class_name):
                        labels[i] = 1
            label_writer.writerow([name] + [str(label) for label in labels])


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Generate audio files by mixing audio events of the DCASE 2013 sound "
                                                 "event detection dataset (task 2). A csv file is also generated "
                                                 "containing the weak labels for the generated segments")
    parser.add_argument("-l", "--length", type=float, default=10,
                        help="Length (in seconds) of the mixed audio files to generate.")
    parser.add_argument("-M", "--max_event", type=float, default=4,
                        help="Maximal number of audio event per mixed file.")
    parser.add_argument("-o", "--overlap", type=str2bool, default=False,
                        help="Controls if the audio events can overlap in the generated mixed files. Default: False.")
    parser.add_argument("-r", "--white_noise_ratio", type=float, default=10.0,
                        help="Ratio between the mean of the clean mixed audio, "
                             "and the white noise to apply as background")
    parser.add_argument("-sr", "--sampling_rate", type=int, default=16000,
                        help="The sampling rate to use for the generated mixed files.")
    parser.add_argument("-N", "--n_files", type=int, default=2000,
                        help="Number of mixed file to generate. Default: 2000")
    parser.add_argument("-p_tr", "--training_percentage", type=float, default=0.8,
                        help="The percentage of the files to use to generate the mixes for the training set. "
                             "(validation percentage is deduced from training and development percentages)")
    parser.add_argument("-p_dev", "--development_percentage", type=float, default=0.1,
                        help="The percentage of the files to use to generate the mixes for the development set. "
                             "(validation percentage is deduced from training and development percentages")
    parser.add_argument("-df", "--DCASE_2013_stereo_data_folder", type=str, required=True,
                        help="Path to the folder 'singlesounds_stereo' provided as part of the DCASE 2013 dataset, "
                             "containing the audio files to mix")
    parser.add_argument("-of", "--output_folder", type=str, required=True,
                        help="Path to the folder where the mixed files will be saved.")

    args = vars(parser.parse_args())

    # Load the DCASE2013 dataset audio files from disk
    audio_files = [f for f in os.listdir(args["DCASE_2013_stereo_data_folder"])
                   if os.path.isfile(os.path.join(args["DCASE_2013_stereo_data_folder"], f)) and f.endswith(".wav")]
    audio_data = [librosa.core.load(os.path.join(args["DCASE_2013_stereo_data_folder"], f), sr=args["sampling_rate"])[0]
                  for f in audio_files]

    # Split the files in training, development and validation set
    permutation = np.arange(len(audio_files))
    np.random.shuffle(permutation)  # random permutation to shuffle the data
    n_tr = int(args["training_percentage"] * len(audio_files))
    n_dev = int(args["development_percentage"] * len(audio_files))

    tr_audio_files = [audio_files[i] for i in permutation[:n_tr]]
    dev_audio_files = [audio_files[i] for i in permutation[n_tr: n_tr + n_dev]]
    test_audio_files = [audio_files[i] for i in permutation[n_tr + n_dev: -1]]

    tr_audio_data = [audio_data[i] for i in permutation[:n_tr]]
    dev_audio_data = [audio_data[i] for i in permutation[n_tr: n_tr + n_dev]]
    test_audio_data = [audio_data[i] for i in permutation[n_tr + n_dev: -1]]

    # Check if the output folder exists, if not creates it, otherwise inform user and stop execution
    for set_name in ["training", "development", "validation"]:
        if not os.path.exists(os.path.join(args["output_folder"], set_name)):
            os.makedirs(os.path.join(args["output_folder"], set_name))
        else:
            raise ValueError('Output folders already exist !')

    generate_mixed_files(tr_audio_files, tr_audio_data,
                         int(args["n_files"] * args["training_percentage"]),
                         os.path.join(args["output_folder"], "training"), args["length"],
                         args["max_event"], args["overlap"], args["white_noise_ratio"], args["sampling_rate"])

    generate_mixed_files(dev_audio_files, dev_audio_data,
                         int(args["n_files"] * args["development_percentage"]),
                         os.path.join(args["output_folder"], "development"), args["length"],
                         args["max_event"], args["overlap"], args["white_noise_ratio"], args["sampling_rate"])

    generate_mixed_files(test_audio_files, test_audio_data,
                         int(np.ceil(args["n_files"]
                                     * (1 - args["training_percentage"] - args["development_percentage"]))),
                         os.path.join(args["output_folder"], "validation"), args["length"],
                         args["max_event"], args["overlap"], args["white_noise_ratio"], args["sampling_rate"])


if __name__ == '__main__':
    main()
