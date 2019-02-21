import numpy as np
import torch
import torch.nn as nn
import scipy.signal


def no_arti_pcen(S, sr=22050, hop_length=512, gain=0.98, bias=2, power=0.5,
                 time_constant=0.400, eps=1e-6, b=None, max_size=1, ref=None,
                 axis=-1, max_axis=None):
    r"""Copy of the librosa implementation of PCEN that uses forward-backward filtering.

        Librosa provides an implementation of the PCEN transform:
        librosa.core.pcen at https://librosa.github.io/librosa/generated/librosa.core.pcen.html

        This version generates artifacts in the output due to the zero-initialization of the low-pass filtered
        version of the input spectrogram used for normalization.
        This effect can be avoided by applying the filter twice: forward and backward. This implementation is just a
        copy of the librosa function were the filtering is changed from 'forward' to 'forward-backward' using the
        scipy.signal.filtfilt function.

    Args:
        Please refer to librosa.core.pcen for the arguments definition

    Returns:
        PCEN transform of the input spectrogram.
    """

    if power <= 0:
        raise ValueError('power={} must be strictly positive'.format(power))

    if gain < 0:
        raise ValueError('gain={} must be non-negative'.format(gain))

    if bias < 0:
        raise ValueError('bias={} must be non-negative'.format(bias))

    if eps <= 0:
        raise ValueError('eps={} must be strictly positive'.format(eps))

    if time_constant <= 0:
        raise ValueError('time_constant={} must be strictly positive'.format(time_constant))

    if max_size < 1 or not isinstance(max_size, int):
        raise ValueError('max_size={} must be a positive integer'.format(max_size))

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)

    if not 0 <= b <= 1:
        raise ValueError('b={} must be between 0 and 1'.format(b))

    if np.issubdtype(S.dtype, np.complexfloating):
        print('pcen was called on complex input so phase '
              'information will be discarded. To suppress this warning, '
              'call pcen(np.abs(D)) instead.')
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ValueError('Max-filtering cannot be applied to 1-dimensional input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ValueError('Max-filtering a {:d}-dimensional spectrogram '
                                     'requires you to specify max_axis'.format(S.ndim))
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    # We could also use a forward only pass of the filter - but initialize it more carefully.
    # S_smooth, _ = scipy.signal.lfilter([b], [1, b - 1], ref, axis=axis,
    #                                    zi=[scipy.signal.lfilter_zi([b], [1, b - 1])] * S[:, 0].shape[0])
    # But forward-backward provides better results.
    S_smooth = scipy.signal.filtfilt([b], [1, b - 1], ref, axis=axis, padtype=None)

    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    return (S * smooth + bias) ** power - bias ** power


def first_order_iir(E, s):
    r"""Implements a first order Infinite Impulse Response (IIR) forward filter initialized using the input values.

        This is a naïve implementation of the filter M defined in Yuxian Wang et al. "Trainable Frontend For Robust and
        Far-Field Keyword Spotting" (2016)

    Args:
        E (torch.tensor): batch of (mel-) spectrograms. shape: [..., Frequency, Time]
        s (float): parameter of the filter.

    Returns:
        Low-pass filtered version of the input spectrograms.
    """

    M = np.zeros_like(E)
    M[..., 0] = E[..., 0]  # Initializes with the value of the spectrograms.
    for frame in range(1, M.shape[-1]):
        M[..., frame] = (1 - s) * M[..., frame - 1] + s * E[..., frame]
    return M


def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
    r"""Implementation of the PCEN transform to apply on a batch of spectrograms.

        Implementation of the PCEN transform to operate on pytorch tensors. Implementation follows the description in
        Yuxian Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting" (2016)

        PCEN(t,f) = ( E(t, f) / (eps + M(t,f)**alpha + delta )**r - delta**r
        M(t,f) = (1-s) * M(t-1,f) + s * E(t,f)

    Args:
        E (torch.tensor): Batch of (mel-) spectrograms. shape: [..., Frequency, Time]
        alpha (float): 'alpha' parameter of the PCEN transform (power of denominator during normalization)
        delta (float): 'delta' parameter of the PCEN transform (shift before dynamic range compression (DRC))
        r (float): 'r' parameter of the PCEN transform (parameter of the power function used for DRC)
        s (float): 's' parameter of the PCEN transform (parameter of the filter M used for normalization)
        eps (float): 'epsilon' parameter of the PCEN transform. (numerical stability of the normalization)

    Returns:
        PCEN transform of the input bach of spectrograms (shape [..., Frequency, Time])
    """

    # Compute low-pass filtered version of the spectrograms.
    M = first_order_iir(E, s)

    # Naive implementation would be:
    # smooth = (eps + M)**(-alpha)
    # return (E * smooth + delta)**r - delta**r

    # Instead make the normalization in exponential representation for numerical stability
    # "stable reformulation due to Vincent Lostanlen" found at
    # https://gist.github.com/f0k/c837bcf0bfde189ca16eab63637839cb
    M = np.exp(-alpha * (np.log(eps) + np.log1p(M / eps)))
    return np.power(E * M + delta, r) - np.power(delta, r)


def torch_lfilter(b, a, x):
    r"""Implements an auto-grad compliant version of a forward infinite impulse response filter.

        Filtering operations are differentiable, therefore it is possible to train the parameters of a filter using
        pytorch auto-grad mechanism. This implementation makes sure that the filter implementation only uses
        operations for which the backward pass can be automatically computed by the auto-grad package,
        so the backward computation is handled automatically.

        The filter is defined by its transfer function. see Transfer function derivation at
        https://en.wikipedia.org/wiki/Infinite_impulse_response.

        H(z) = (\sum i=0 to P { b_i z**-i}) / (\sum j=0 to Q {a_j z**-j})

        This function is supposed to give the same result than the following less-convoluted implementation,
        but using auto-grad friendly operations.
        def torch_lfilter(b, a, x):
            M = torch.zeros(x.shape, device=x.device)
            b_last_dim = len(b.shape) - 1
            a_last_dim = len(a.shape) - 1
            P = b.shape[b_last_dim] - 1
            Q = a.shape[a_last_dim] - 1
            b_flip = torch.flip(b, [b_last_dim])
            a_flip = torch.flip(a, [a_last_dim])
            init_steps = np.max([P, Q])

            M[..., :init_steps] = x[..., :init_steps]
            for step in range(init_steps, x.shape[-1]):
                M[..., step] = M[..., step] + torch.sum(b_flip * x[..., step - P:step + 1], dim=-1)\
                               - torch.sum(a_flip[..., :-1] * M[..., step - Q:step], dim=-1)
            return M / a_flip[..., -1].unsqueeze(-1)

    Args:
        b (torch.tensor): Vector of coefficient of the numerator polynomial of the filter transfer function.
                          shape: [(Number of filters,) Number of frequency bins, feedforward filter order]
        a (torch.tensor): Vector of coefficient of the denominator polynomial of the filter transfer function.
                          shape: [(Number of filters,) Number of frequency bins, feedback filter order]
            b and a last dimension are the filter orders. their before last dimension is the number of frequency bins
            in the spectrogram. If they have one more dimension (the first dimension), then they contain parameters
            values for several filters, indexed by this dimension.
        x (torch.tensor): batch of spectrograms. shape: [Batch, Channel=1, Frequency, Time]

    Returns:
        batch of filtered spectrograms. shape [B, C=1, Frequency, Time]
    """

    # a, b must have shape [(c,) f, *]
    # ie the value of the filter parameter must be specified for all frequency bins.
    # if the dimension c exists, several filters values are passed, one for each value in range(c).
    P = b.shape[-1]  # Feedforward order
    Q = a.shape[-1]  # Feedback order
    # At time step t, the filter computation uses values in previous time step.
    # Unfortunately, at this moment pytorch does not support negative striding, so we need to flip a and b to align
    # the coeficients with their values in the x tensor.
    b_flip = torch.flip(b, [b.dim() - 1])
    a_flip = torch.flip(a, [a.dim() - 1])
    # The filtered version is initialized by copying as much values in the input spectrograms as needed.
    init_steps = np.max([P, Q])
    sum_length_diff = np.abs(Q - P)

    # Filter values are computed as if using a  sliding window in the time dimension of the input tensor.
    # A auto-grad compatible implementation of this operation is to generate all the values of the sliding window in a
    # new tensor beforehand. This is done by the 'unfold' operator of pytorch.
    # Unfold to generate all the P window in the time dimension (-1) with stride 1.
    # Then move the frequency axis to 2nd position, for broadcasting compatibility with b and a.
    P_strided_x = x.unfold(-1, P, 1).permute(0, 3, 1, 2, 4)

    # Compute the feedforward sum for all time steps.
    # Then move back the frequency axis
    # Then split the sum values along the time dimension in a list. This is so that we can use the value at a precise
    # time step without using indexing (ie: P_sum[..., 0]) that would not be compatible with auto-grad.
    P_sum = torch.sum(b_flip * P_strided_x, dim=-1).permute(0, 2, 3, 1).split(1, -1)

    # Compute the filtered values at each time step in a list.
    result = []  # the list index runs along the time dimension
    # First initialize the filtered values with the input values for the required number of time steps.
    for step in range(init_steps):
        # In case that b and a define several filters, duplicate the spectrograms values of the initialization steps
        # to match the filters dimension
        if b.dim() == 3:
            result.append(x[..., step].expand(-1, b.shape[0], -1))
        # Otherwise, simply copy the values.
        else:
            result.append(x[..., step])

    # Filtered values computation.
    # The forward sum is already computed, this loop handles the feedback sum at each time steps.
    # The result is summed with the feedforward results to get the filtered values.
    for step in range(1 + sum_length_diff, x.shape[-1]):
        result.append(P_sum[step].squeeze(-1)
                      - torch.sum(a_flip[..., :-1] * torch.stack(result[-Q + 1:], -1), dim=-1))

    # Stack the results for each time step in a tensor (time dimension is last) and divide by a_0
    return torch.stack(result, -1) / a_flip[..., -1].unsqueeze(-1)


def torch_filtfilt(b, a, x):
    r"""Implements an auto-grad compatible forward-backward filtering operation based on torch_lfilter.

        Applies the filter defined by the polynomials 'a' and 'b' forward in time, then backward.
        This is in the same spirit of the scipy.signal.filtfilt function, but using auto-grad friendly operations.

    Args:
        b (torch.tensor): Vector of coefficient of the numerator polynomial of the filter transfer function.
                          shape: [(Number of filters,) Number of frequency bins, feedforward filter order]
        a (torch.tensor): Vector of coefficient of the denominator polynomial of the filter transfer function.
                          shape: [(Number of filters,) Number of frequency bins, feedback filter order]
            b and a last dimension are the filter orders. their before last dimension is the number of frequency bins
            in the spectrogram. If they have one more dimension (the first dimension), then they contain parameters
            values for several filters, indexed by this dimension.
        x (torch.tensor): batch of spectrograms. shape: [Batch, Channel=1, Frequency, Time]

    Returns:
        forward-backward filtered version of the input spectrograms,
    """

    # get the index of the time dimension
    last_dim = x.dim() - 1
    # forward pass
    y = torch_lfilter(b, a, x)
    # flip along the time dimension
    z = y.flip(last_dim)
    # backward pass
    zz = torch_lfilter(b, a, z)
    return zz.flip(last_dim)  # flip back the time dimension


class PCENLayer(nn.Module):
    r"""Implements a trainable PCEN transform as a pytorch module.

        As proposed in Yuxian Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting" (2016),
        the PCEN calculation is differentiable with respect to its parameters, hence it is possible to include it in
        a training framework as a model layer.

        PCEN(t,f) = ( E(t, f) / (eps + M(t,f)**alpha + delta )**r - delta**r
        - M(t,f) = (1-s) * M(t-1,f) + s * E(t,f) (paper implementation)
        - M(t,f) defined by its polynomial coefficients a and b as in Transfer function derivation at
        https://en.wikipedia.org/wiki/Infinite_impulse_response

        The parameters 'r', 'delta' and 'alpha' in the paper can be trained directly with gradient descent based
        algorithms. This class handles the case where the same value of these parameters is used for all frequency
        bins, as well as the case where the value of these parameters are optimized independently for each frequency
        bins.

        For training the parameters related to the filtering operation of the PCEN, two methods are available:
            1 - use a fixed set of parameter 's', and associated filters. Learn the weights w_k to give to these
            filters for the final filtered version. This is what is proposed in Yuxian Wang et al. "Trainable
            Frontend For Robust and Far-Field Keyword Spotting" (2016),

            2 - Use only 1 filter, but directly optimize the parameters of this filter (the coefficients of the
            polynomials of its transfer function). If the filter is defined as in 'Transfer function derivation' at
            https://en.wikipedia.org/wiki/Infinite_impulse_response ; then this method consist in directly optimizing
            the values of a_j and b_i.
            In addition, a different value of a_j and b_i can be optimized for each frequency bin.

        The training parameters usually belong in an interval. In order to make sure that their value stays in this
        interval during the optimization, they are stored as real numbers and parametrized by suitable function.
        Eg: 'r' value should be in [0, 1]. To ensure this, 'r' is used as sigmoid(i_sig_r) and the parameter i_sig_r
        is optimized.
    """

    def __init__(self, per_band_param, in_f_size, use_s, s, per_band_filter, b, a, eps=1e-6):
        r"""Constructor. Initializes the PCEN parameters.

            The parameters values are parametrized by suitable function to make sure they stay in the interval of
            definition during the optimization. The parameter are initialized using the default values proposed in
            Yuxian Wang et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting" (2016),

        Args:
            per_band_param (bool): If True, an independent value of the parameters 'alpha', 'delta' and 'r' will be
                                   optimized for each frequency bin. Otherwise, the same value is used for all bins.
            in_f_size (int): Number of frequency bins in the spectrogram to process
            use_s (bool): if True, uses the optimization scheme 1 for the filtering operation of the PCEN (learn the
            weights to give to a set of filters with fixed parameter 's')
            s (list): List of floats: 's' values to use if the optimization scheme 1 has been chosen.
                      values of 's' should lie in [0, 1]
            per_band_filter (bool): If the optimization scheme 2 has been chosen (optimize only 1 filter,
                                    but directly tuning the filter parameters), then this option lets the user chose
                                    between using a single filter for all frequency bins (False), or use independent
                                    filter parameters for each frequency bins (True).
            b (list): Filter feedforward coefficient (only used with optimization scheme 2)
            a (list): Filter feedback coefficients (only used with optimization scheme 2)
                      The lenght of 'a' and 'b' define the filter orders
            eps (float): Value of parameter epsilon (numerical stability)
        """

        super().__init__()

        # Use a independent parameter value for each frequency bin
        self.per_band_param = per_band_param
        if per_band_param:
            # Pick random values around default for each frequency bin
            # Parameters in [0, 1] are parametrized by a sigmoid, parameters in [0, +inf) are parametrized by log
            self.i_sig_alpha = (torch.randn(in_f_size) * 0.05 + 0.9).clamp(min=0.1, max=0.995)
            self.i_sig_alpha = nn.Parameter(torch.log(self.i_sig_alpha / (1.0 - self.i_sig_alpha)))
            self.log_delta = nn.Parameter((torch.randn(in_f_size) * 0.1 + 2.0).clamp(min=1.0, max=3.0).log_())
            self.i_sig_r = (torch.randn(in_f_size) * 0.05 + 0.5).clamp(min=0.1, max=0.9)
            self.i_sig_r = nn.Parameter(torch.log(self.i_sig_r / (1.0 - self.i_sig_r)))
        # use the same value for all frequency bins
        else:
            self.i_sig_alpha = nn.Parameter(torch.log(torch.tensor(0.98 / (1.0 - 0.98))))  # sigmoid^-1(0.98)
            self.log_delta = nn.Parameter(torch.tensor(2.0).log_())
            self.i_sig_r = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5

        # Use a fixed set of filters and lean the weights to give to each filter
        self.use_s = use_s
        if use_s:
            self.s = s
            # vector of weights to give to each filter
            self.z_ks = nn.Parameter(torch.randn((len(s), in_f_size)) * 0.1 + np.log(1 / len(s)))
        # use 1 filter and directly optimize its parameters
        else:
            self.per_band_filter = per_band_filter
            # use independent filter parameters for each frequency bins
            if per_band_filter:
                self.b = nn.Parameter(torch.randn((in_f_size, len(b))) * np.log(1 / len(b)) + torch.tensor(b))
                self.b = nn.Parameter(torch.randn((in_f_size, len(a))) * np.log(1 / len(a)) + torch.tensor(a))
            # use the same filter for all frequency bins
            else:
                self.b = nn.Parameter(torch.tensor(b))
                self.a = nn.Parameter(torch.tensor(a))

        self.eps = eps

    def forward(self, x):
        r"""Implements the forward pass of the PCEN layer

            PCEN(t,f) = ( E(t, f) / (eps + M(t,f)**alpha + delta )**r - delta**r

            First the parameters values are computed (going from real range to the parameter range)
            Then the computes the filtering operation of the PCEN using an auto-grad compatible implementation
            Finally compute the normalization and dynamic range compression.

        Args:
            x (torch.tensor): batch of (mel-) spectrograms. shape: [Batch, C=1, Frequency, Time]

        Returns:
            PCEN transform of each spectrogram in the batch. shape: [Batch, C=1, Frequency, Time]
        """

        # Compute parameter values (they are stored in ]-inf, +inf[, so this moves to their interval of definition)
        if self.per_band_param:  # if we use the same parameter value for all frequency bin, we have to expand it.
            alpha = self.i_sig_alpha.sigmoid().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1],
                                                                                              -1, x.shape[-1])
            delta = self.log_delta.exp().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], -1,
                                                                                        x.shape[-1])
            r = self.i_sig_r.sigmoid().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], -1,
                                                                                      x.shape[-1])
        else:
            alpha = self.i_sig_alpha.sigmoid().expand_as(x)
            delta = self.log_delta.exp().expand_as(x)
            r = self.i_sig_r.sigmoid().expand_as(x)

        # If using optimization scheme 1, we need to compute the weights given to each filter
        if self.use_s:
            # map the weights stored in ]-inf, +inf[ to [0, 1] and such that they sum up to 1 (softmax)
            w_ks = (self.z_ks.exp() / self.z_ks.exp().sum(dim=0)) \
                .unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[0], x.shape[1], -1, x.shape[-1])
            # Compute the filtered version for each filter
            smoothers = torch.stack([
                torch.tensor(scipy.signal.filtfilt([s], [1, s - 1], x, axis=-1, padtype=None).astype(np.float32),
                             device=x.device) for s in self.s])
            # Take the weighted combination of the filters
            M = (smoothers * w_ks).sum(dim=0)
        # If using optimization scheme 2: compute 1 filtered version with the parameters a and b
        else:
            if self.per_band_filter:  # use different filter parameters for each frequency bin
                M = torch_filtfilt(self.b, self.a, x)
            else:  # replicate b and a values for each freq bin
                M = torch_filtfilt(self.b.unsqueeze(0).expand(x.shape[2], -1),
                                   self.a.unsqueeze(0).expand(x.shape[2], -1),
                                   x)

        # Description in paper
        # return (x / (M + self.eps).pow(alpha) + delta).pow(r) - delta.pow(r)
        # More stable version
        M = torch.exp(-alpha * (float(np.log(self.eps)) + torch.log1p(M / self.eps)))
        result = (x * M + delta).pow(r) - delta.pow(r)

        return result


class MultiPCENlayer(nn.Module):
    r"""Implements multiple trainable PCEN transforms of the same input.

        The PCEN transform can enhance the signal for some audio events, and deteriorate it for other events (for
        instance enhancing speech might mean deteriorating alarm sounds). To circumvent this problem, one can try to
        train multiple PCEN transforms, providing their results to the same network so that the network has access to
        enhanced signal for all classes.
        This class implements a stack of trainable PCEN transforms. However, a simplified version of the trainable
        PCEN layer is used in this case, in which all parameters are directly optimized.
        Parameters 'alpha', 'delta', 'r', 's' are directly optimized.

    """

    def __init__(self, n_pcen, eps=1e-6):
        r"""Constructor. Initializes the PCEN transforms parameters to random values close to the default.

        Args:
            n_pcen (int): Number of PCEN transform to use. (positive >= 1)
            eps (float): Numerical stability parameter of the PCEN transforms.
        """

        super(MultiPCENlayer, self).__init__()

        self.n_pcen = n_pcen

        # inverse_sigmoid(alpha), using the default value of alpha: 0.98
        self.i_sig_alpha = torch.log(torch.tensor(0.98 / (1.0 - 0.98)))
        self.i_sig_alpha = nn.Parameter(self.i_sig_alpha * (1.0 + torch.rand(n_pcen) * 0.1))
        # log(delta), using the default value of delta 2.0
        self.log_delta = torch.tensor(2.0).log_()
        self.log_delta = nn.Parameter(self.log_delta * (1.0 + torch.rand(n_pcen) * 0.1))
        # inverse_sigmoid(r), using the default value of r: 0.5
        self.i_sig_r = torch.tensor(0.0)
        self.i_sig_r = nn.Parameter(self.i_sig_r * (1.0 + torch.rand(n_pcen) * 0.1))

        # inverse_sigmoid(s), using the default value of s: 0.04
        self.i_sig_s = torch.log(torch.tensor(0.04 / (1.0 - 0.04)))
        self.i_sig_s = nn.Parameter(self.i_sig_s * (1.0 + torch.rand(n_pcen) * 0.1))

        # inverse_sigmoid(epsilon)
        self.i_sig_eps = nn.Parameter(torch.log(torch.tensor(eps / (1.0 - eps))))

    def forward(self, x):
        r"""Implements the forward pass of multiple PCEN transform

        Args:
            x (torch.tensor): Batch of (mel-) spectrograms. shape: [Batch, C=1, Frequency, Time]

        Returns:
            PCEN transforms of the input batch. shape: [Batch, number of PCEN layer, Frequency, Time]
        """

        # Get parameters values in the right range
        alpha = self.i_sig_alpha.sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1,
                                                                                           x.shape[2], x.shape[3])
        delta = self.log_delta.exp().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1,
                                                                                     x.shape[2], x.shape[3])
        r = self.i_sig_r.sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1,
                                                                                   x.shape[2], x.shape[3])
        b = self.i_sig_s.sigmoid().unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[2], -1)
        a = torch.cat((torch.ones((self.n_pcen, x.shape[2], 1), device=x.device, dtype=x.dtype),
                       self.i_sig_s.sigmoid().unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[2], -1) - 1.0),
                      dim=-1)
        eps = self.i_sig_eps.sigmoid()

        # Compute filtered version
        M = torch_filtfilt(b, a, x)

        # Stable PCEN formula
        M = torch.exp(-alpha * (torch.log(eps) + torch.log1p(M / eps)))
        result = (x * M + delta).pow(r) - delta.pow(r)

        return result
