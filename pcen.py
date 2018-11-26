import numpy as np
import torch
import torch.nn as nn
import scipy.signal


def first_order_iir(S, s):
    M = np.zeros_like(S)
    M[..., 0] = S[..., 0]
    for frame in range(1, M.shape[-1]):
        M[..., frame] = (1 - s) * M[..., frame - 1] + s * S[..., frame]
    return M


def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
    M = first_order_iir(E, s)

    # smooth = (eps + M)**(-alpha)
    # return (E * smooth + delta)**r - delta**r
    M = np.exp(-alpha * (np.log(eps) + np.log1p(M / eps)))
    return np.power(E * M + delta, r) - np.power(delta, r)


def no_arti_pcen(S, sr=22050, hop_length=512, gain=0.98, bias=2, power=0.5,
                 time_constant=0.400, eps=1e-6, b=None, max_size=1, ref=None,
                 axis=-1, max_axis=None):
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

    S_smooth, _ = scipy.signal.lfilter([b], [1, b - 1], ref, axis=axis,
                                       zi=[scipy.signal.lfilter_zi([b], [1, b - 1])] * S[:, 0].shape[0])

    S_smooth = scipy.signal.filtfilt([b], [1, b - 1], ref, axis=axis, padtype=None)

    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    return (S * smooth + bias) ** power - bias ** power


class PCEN(nn.Module):

    def __init__(self, in_f_size, s=None, eps=1e-6):
        super().__init__()

        self.log_alpha = nn.Parameter((torch.randn(in_f_size) * 0.1 + 1.0).log_())
        self.log_delta = nn.Parameter((torch.randn(in_f_size) * 0.1 + 1.0).log_())
        self.log_r = nn.Parameter((torch.randn(in_f_size) * 0.1 + 1.0).log_())

        if s is None:  # Default values
            self.s = [0.015, 0.02, 0.04, 0.08]
        else:
            self.s = s
        self.z_ks = nn.Parameter(torch.randn((len(s), in_f_size)) * 0.1 + np.log(1 / len(s)))

        self.eps = eps

    def forward(self, x):
        alpha = self.log_alpha.exp().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], -1,
                                                                                    x.shape[-1])
        delta = self.log_delta.exp().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], -1,
                                                                                    x.shape[-1])
        r = self.log_r.exp().unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], -1, x.shape[-1])
        w_ks = (self.z_ks.exp() / self.z_ks.exp().sum()) \
            .unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[0], x.shape[1], -1, x.shape[-1])

        smoothers = torch.stack([
            torch.tensor(scipy.signal.filtfilt([s], [1, s - 1], x, axis=-1, padtype=None).astype(np.float32),
                         device=x.device) for s in self.s])
        M = (smoothers * w_ks).sum(dim=0)

        # Description in paper
        # return (x / (M + self.eps).pow(alpha) + delta).pow(r) - delta.pow(r)
        # More stable version
        M = torch.exp(-alpha * (float(np.log(self.eps)) + torch.log1p(M / self.eps)))
        result = (x * M + delta).pow(r) - delta.pow(r)

        return result
