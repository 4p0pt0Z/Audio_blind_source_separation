import librosa
import argparse
import numpy as np
import pandas as pd
import os
import torch
import scipy

import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
from matplotlib.patches import Patch

import seaborn as sns

from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams

import segmentation_model as mod
from separator import AudioSeparator
import librosa.display
import os
import yaml
import mir_eval


class MelScale(mscale.ScaleBase):
    """ Mel Scale transform for axis in plots

    """
    name = 'mel'

    def __init__(self, axis, *, fmin=0.0, fmax=8.0, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.

        thresh: The degree above which to crop the data.
        """
        mscale.ScaleBase.__init__(self)
        self.fmin = fmin
        self.fmax = fmax

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.MelTransform()

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in ``ticker.py``.

        In our case, the Mercator example uses a fixed locator from
        -90 to 90 degrees and a custom formatter class to put convert
        the radians to degrees and put a degree symbol after the
        value::
        """

    #         axis.set_major_formatter(axis.get_major_locator())
    #         axis.set_minor_formatter(axis.get_minor_locator())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return self.fmin, self.fmax

    class MelTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the Mercator scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """

            return librosa.hz_to_mel(a*1000.0)

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return MelScale.InvertedMelTransform()

    class InvertedMelTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return librosa.mel_to_hz(a) / 1000.0

        def inverted(self):
            return MelScale.MelTransform()


def main():
    # Register Mel scale
    mscale.register_scale(MelScale)

    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="For the model specified by input, computes the separated audio "
                                                 "files of the ICASP2018 challenge, then evaluate the separation "
                                                 "metrics")
    parser.add_argument("--sep_audio_folder", type=str, required=True,
                        help="Folder to store the separated audio files.")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to the checkpoint of the model to evaluate.")
    user_args = vars(parser.parse_known_args()[0])

    model_ckpt = user_args['model_ckpt']
    separated_audio_folder = user_args['sep_audio_folder']

    synthetizer = AudioSeparator.from_checkpoint(
        {"checkpoint_path": model_ckpt, "separated_audio_folder": separated_audio_folder})

    config = synthetizer.config

    save_path = os.path.join(os.path.dirname(model_ckpt),
                             os.path.splitext(os.path.basename(model_ckpt))[0] + '_figures')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise RuntimeError('Figure directory already exists ! ')

    baby_cry_example = 17
    _, babycrymask = synthetizer.model(synthetizer.data_set.__getitem__(baby_cry_example)[0].unsqueeze(0))
    babycrymask = babycrymask.detach().clone().squeeze()[0]
    fig1, axs = plt.subplots(1, 1, figsize=(5, 3))
    axs.pcolormesh(np.arange(babycrymask.shape[1]) * synthetizer.data_set.config['STFT_frame_shift_ms'] / 1000,
                         librosa.filters.mel_frequencies(64, fmin=1, fmax=8000) / 1000,
                         np.squeeze(babycrymask),
                         cmap='magma', vmin=0.0, vmax=1.0, zorder=0)
    axs.set_yscale('mel')
    plt.locator_params(axis='y', nbins=4)
    plt.tight_layout()
    plt.show()
    fig1.savefig(os.path.join(save_path, 'babycry_mask.svg'), format='svg', bbox_inches='tight')
    fig1.savefig(os.path.join(save_path, 'babycry_mask.eps'), format='eps', bbox_inches='tight')
    fig1.savefig(os.path.join(save_path, 'babycry_mask.pdf'), format='pdf', bbox_inches='tight')


    gunshot_example = 50
    _, gunshotmask = synthetizer.model(synthetizer.data_set.__getitem__(gunshot_example)[0].unsqueeze(0))
    gunshotmask = gunshotmask.detach().clone().squeeze()[1]
    fig2, axs = plt.subplots(1, 1, figsize=(5, 3))
    axs.pcolormesh(np.arange(gunshotmask.shape[1]) * synthetizer.data_set.config['STFT_frame_shift_ms'] / 1000,
                         librosa.filters.mel_frequencies(64, fmin=1, fmax=8000) / 1000,
                         np.squeeze(gunshotmask),
                         cmap='magma', vmin=0.0, vmax=1.0, zorder=0)
    axs.set_yscale('mel')
    plt.locator_params(axis='y', nbins=4)
    plt.tight_layout()
    plt.show()
    fig2.savefig(os.path.join(save_path, 'gunshot_mask.svg'), format='svg', bbox_inches='tight')
    fig2.savefig(os.path.join(save_path, 'gunshot_mask.eps'), format='eps', bbox_inches='tight')
    fig2.savefig(os.path.join(save_path, 'gunshot_mask.pdf'), format='pdf', bbox_inches='tight')

    glassbreak_example = 131
    _, glassbreakmask = synthetizer.model(synthetizer.data_set.__getitem__(glassbreak_example)[0].unsqueeze(0))
    glassbreakmask = glassbreakmask.detach().clone().squeeze()[2]
    fig3, axs = plt.subplots(1, 1, figsize=(5, 3))
    axs.pcolormesh(np.arange(glassbreakmask.shape[1]) * synthetizer.data_set.config['STFT_frame_shift_ms'] / 1000,
                         librosa.filters.mel_frequencies(64, fmin=1, fmax=8000) / 1000,
                         np.squeeze(glassbreakmask),
                         cmap='magma', vmin=0.0, vmax=1.0, zorder=0)
    axs.set_yscale('mel')
    plt.locator_params(axis='y', nbins=4)
    plt.tight_layout()
    plt.show()
    fig3.savefig(os.path.join(save_path, 'glassbreak_mask.svg'), format='svg', bbox_inches='tight')
    fig3.savefig(os.path.join(save_path, 'glassbreak_mask.eps'), format='eps', bbox_inches='tight')
    fig3.savefig(os.path.join(save_path, 'glassbreak_mask.pdf'), format='pdf', bbox_inches='tight')

    synthetizer.separate(separation_method='in_lin')

    indices_babycry = np.where(synthetizer.data_set.labels.cpu().numpy()[:, 0] == 1)[0]
    indices_glassbreak = np.where(synthetizer.data_set.labels.cpu().numpy()[:, 1] == 1)[0]
    indices_gunshot = np.where(synthetizer.data_set.labels.cpu().numpy()[:, 2] == 1)[0]
    indices_background = np.where(synthetizer.data_set.labels.cpu().numpy()[:, 3] == 1)[0]

    indices_babycry_mix = np.intersect1d(indices_babycry, indices_background)
    indices_glassbreak_mix = np.intersect1d(indices_glassbreak, indices_background)
    indices_gunshot_mix = np.intersect1d(indices_gunshot, indices_background)

    indices_babycry_only = np.setdiff1d(indices_babycry, indices_background)
    indices_glassbreak_only = np.setdiff1d(indices_glassbreak, indices_background)
    indices_gunshot_only = np.setdiff1d(indices_gunshot, indices_background)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdrs, sirs, sars = synthetizer.evaluate_separation()

    format_string = 'mean {:^9.4f}, std {:^9.4f}, median {:^9.4f}\nSIR: mean {:^9.4f}, std {:^9.4f}, ' \
                    'median {:^9.4f}\nSAR: mean {:^9.4f}, std {:^9.4f}, median {:^9.4f}'
    print('Babycry mixes\nSDR: ' + format_string.format(
        sdrs[indices_babycry_mix, 0].mean(), sdrs[indices_babycry_mix, 0].std(),
        np.median(sdrs[indices_babycry_mix, 0]),
        sirs[indices_babycry_mix, 0].mean(), sirs[indices_babycry_mix, 0].std(),
        np.median(sirs[indices_babycry_mix, 0]),
        sars[indices_babycry_mix, 0].mean(), sars[indices_babycry_mix, 0].std(),
        np.median(sars[indices_babycry_mix, 0])))
    print('Babycry only\nSDR: ' + format_string.format(
        sdrs[indices_babycry_only, 0].mean(), sdrs[indices_babycry_only, 0].std(),
        np.median(sdrs[indices_babycry_only, 0]),
        sirs[indices_babycry_only, 0].mean(), sirs[indices_babycry_only, 0].std(),
        np.median(sirs[indices_babycry_only, 0]),
        sars[indices_babycry_only, 0].mean(), sars[indices_babycry_only, 0].std(),
        np.median(sars[indices_babycry_only, 0])))
    print('Glassbreak mixes\nSDR: ' + format_string.format(
        sdrs[indices_glassbreak_mix, 1].mean(), sdrs[indices_glassbreak_mix, 1].std(),
        np.median(sdrs[indices_glassbreak_mix, 1]),
        sirs[indices_glassbreak_mix, 1].mean(), sirs[indices_glassbreak_mix, 1].std(),
        np.median(sirs[indices_glassbreak_mix, 1]),
        sars[indices_glassbreak_mix, 1].mean(), sars[indices_glassbreak_mix, 1].std(),
        np.median(sars[indices_glassbreak_mix, 1])))
    print('Glassbreak only\nSDR: ' + format_string.format(
        sdrs[indices_glassbreak_only, 1].mean(), sdrs[indices_glassbreak_only, 1].std(),
        np.median(sdrs[indices_glassbreak_only, 1]),
        sirs[indices_glassbreak_only, 1].mean(), sirs[indices_glassbreak_only, 1].std(),
        np.median(sirs[indices_glassbreak_only, 1]),
        sars[indices_glassbreak_only, 1].mean(), sars[indices_glassbreak_only, 1].std(),
        np.median(sars[indices_glassbreak_only, 1])))
    print('Gunshot mixes\nSDR: ' + format_string.format(
        sdrs[indices_gunshot_mix, 2].mean(), sdrs[indices_gunshot_mix, 2].std(),
        np.median(sdrs[indices_gunshot_mix, 2]),
        sirs[indices_gunshot_mix, 2].mean(), sirs[indices_gunshot_mix, 2].std(),
        np.median(sirs[indices_gunshot_mix, 2]),
        sars[indices_gunshot_mix, 2].mean(), sars[indices_gunshot_mix, 2].std(),
        np.median(sars[indices_gunshot_mix, 2])))
    print('Gunshot only\nSDR: ' + format_string.format(
        sdrs[indices_gunshot_only, 2].mean(), sdrs[indices_gunshot_only, 2].std(),
        np.median(sdrs[indices_gunshot_only, 2]),
        sirs[indices_gunshot_only, 2].mean(), sirs[indices_gunshot_only, 2].std(),
        np.median(sirs[indices_gunshot_only, 2]),
        sars[indices_gunshot_only, 2].mean(), sars[indices_gunshot_only, 2].std(),
        np.median(sars[indices_gunshot_only, 2])))


if __name__ == '__main__':
    main()
