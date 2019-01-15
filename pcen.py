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


class PCENLayer(nn.Module):

    def __init__(self, per_band_param, in_f_size, use_s, s, per_band_filter, b, a, eps=1e-6):
        super().__init__()

        self.per_band_param = per_band_param
        if per_band_param:
            # Pick random values around default for each frequency bin
            # Parameters in [0, 1] are parametrized by a sigmoid, parameters in [0, +inf) are parametrized by log
            self.i_sig_alpha = (torch.randn(in_f_size) * 0.05 + 0.9).clamp(min=0.1, max=0.995)
            self.i_sig_alpha = nn.Parameter(torch.log(self.i_sig_alpha / (1.0 - self.i_sig_alpha)))
            self.log_delta = nn.Parameter((torch.randn(in_f_size) * 0.1 + 2.0).clamp(min=1.0, max=3.0).log_())
            self.i_sig_r = (torch.randn(in_f_size) * 0.05 + 0.5).clamp(min=0.1, max=0.9)
            self.i_sig_r = nn.Parameter(torch.log(self.i_sig_r / (1.0 - self.i_sig_r)))
        else:
            self.i_sig_alpha = nn.Parameter(torch.log(torch.tensor(0.98 / (1.0 - 0.98))))  # sigmoid^-1(0.98)
            self.log_delta = nn.Parameter(torch.tensor(2.0).log_())
            self.i_sig_r = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5

        self.use_s = use_s
        if use_s:
            self.s = s
            self.z_ks = nn.Parameter(torch.randn((len(s), in_f_size)) * 0.1 + np.log(1 / len(s)))
        else:
            self.per_band_filter = per_band_filter
            if per_band_filter:
                self.b = nn.Parameter(torch.randn((in_f_size, len(b))) * np.log(1 / len(b)) + torch.tensor(b))
                self.b = nn.Parameter(torch.randn((in_f_size, len(a))) * np.log(1 / len(a)) + torch.tensor(a))
            else:
                self.b = nn.Parameter(torch.tensor(b))
                self.a = nn.Parameter(torch.tensor(a))

        self.eps = eps

    def forward(self, x):
        if self.per_band_param:
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

        if self.use_s:
            w_ks = (self.z_ks.exp() / self.z_ks.exp().sum(dim=0)) \
                .unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[0], x.shape[1], -1, x.shape[-1])

            smoothers = torch.stack([
                torch.tensor(scipy.signal.filtfilt([s], [1, s - 1], x, axis=-1, padtype=None).astype(np.float32),
                             device=x.device) for s in self.s])
            M = (smoothers * w_ks).sum(dim=0)
        else:
            if self.per_band_filter:  # replicate b and a values for each freq bin
                M = torch_filtfilt(self.b, self.a, x)
            else:
                M = torch_filtfilt(self.b.unsqueeze(0).expand(x.shape[2], -1),
                                   self.a.unsqueeze(0).expand(x.shape[2], -1),
                                   x)

        # Description in paper
        # return (x / (M + self.eps).pow(alpha) + delta).pow(r) - delta.pow(r)
        # More stable version
        M = torch.exp(-alpha * (float(np.log(self.eps)) + torch.log1p(M / self.eps)))
        result = (x * M + delta).pow(r) - delta.pow(r)

        return result


def torch_lfilter(b, a, x):
    """ a, b must have shape [(c,) f, *] """
    P = b.shape[-1]
    Q = a.shape[-1]
    b_flip = torch.flip(b, [b.dim() - 1])
    a_flip = torch.flip(a, [a.dim() - 1])
    init_steps = np.max([P, Q])
    sum_length_diff = np.abs(Q - P)

    # P_strided_x = x.unfold(-1, P, 1).permute(0, 1, 3, 2, 4)  # move frequency to before last position to match b shape
    # P_sum = torch.sum(b_flip * P_strided_x, dim=-1).permute(0, 1, 3, 2).split(1, -1)  # move frequency back
    P_strided_x = x.unfold(-1, P, 1).permute(0, 3, 1, 2, 4)
    P_sum = torch.sum(b_flip * P_strided_x, dim=-1).permute(0, 2, 3, 1).split(1, -1)
    result = []
    for step in range(init_steps):
        if b.dim() == 3:
            result.append(x[..., step].expand(-1, b.shape[0], -1))
        else:
            result.append(x[..., step])
    for step in range(1 + sum_length_diff, x.shape[-1]):
        result.append(P_sum[step].squeeze(-1)
                      - torch.sum(a_flip[..., :-1] * torch.stack(result[-Q + 1:], -1), dim=-1))
    return torch.stack(result, -1) / a_flip[..., -1].unsqueeze(-1)


def torch_filtfilt(b, a, x):
    last_dim = x.dim() - 1
    y = torch_lfilter(b, a, x)
    z = y.flip(last_dim)
    zz = torch_lfilter(b, a, z)
    return zz.flip(last_dim)


# def torch_lfilter(b, a, x):
#     M = torch.zeros(x.shape, device=x.device)
#     b_last_dim = len(b.shape) - 1
#     a_last_dim = len(a.shape) - 1
#     P = b.shape[b_last_dim] - 1
#     Q = a.shape[a_last_dim] - 1
#     b_flip = torch.flip(b, [b_last_dim])
#     a_flip = torch.flip(a, [a_last_dim])
#     init_steps = np.max([P, Q])
#
#     M[..., :init_steps] = x[..., :init_steps]
#     for step in range(init_steps, x.shape[-1]):
#         M[..., step] = M[..., step] + torch.sum(b_flip * x[..., step - P:step + 1], dim=-1)\
#                        - torch.sum(a_flip[..., :-1] * M[..., step - Q:step], dim=-1)
#     return M / a_flip[..., -1].unsqueeze(-1)


class MultiPCENlayer(nn.Module):
    def __init__(self, n_pcen, eps=1e-6):
        super(MultiPCENlayer, self).__init__()

        self.n_pcen = n_pcen

        self.i_sig_alpha = torch.log(torch.tensor(0.98 / (1.0 - 0.98)))
        self.i_sig_alpha = nn.Parameter(self.i_sig_alpha * (1.0 + torch.rand(n_pcen)*0.1))
        self.log_delta = torch.tensor(2.0).log_()
        self.log_delta = nn.Parameter(self.log_delta * (1.0 + torch.rand(n_pcen)*0.1))
        self.i_sig_r = torch.tensor(0.0)
        self.i_sig_r = nn.Parameter(self.i_sig_r * (1.0 + torch.rand(n_pcen)*0.1))

        self.i_sig_s = torch.log(torch.tensor(0.04 / (1.0 - 0.04)))
        self.i_sig_s = nn.Parameter(self.i_sig_s * (1.0 + torch.rand(n_pcen)*0.1))

        self.i_sig_eps = nn.Parameter(torch.log(torch.tensor(eps / (1.0 - eps))))

    def forward(self, x):
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

        M = torch_filtfilt(b, a, x)

        M = torch.exp(-alpha * (torch.log(eps) + torch.log1p(M / eps)))
        result = (x * M + delta).pow(r) - delta.pow(r)

        return result
