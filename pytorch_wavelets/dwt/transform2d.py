import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


#def get_highpass_filter(var_filter):
#    print(var_filter)
#    a = torch.pow(-1., torch.arange(1.,1+ var_filter.size(0)))
#    return a*torch.flip(var_filter,[0])

def get_highpass_filter(var_filter):
    a = torch.zeros_like(var_filter)
    a[0, 0, :, 0] = torch.pow(-1., torch.arange(0., var_filter.size(2)))
    return a*torch.flip(var_filter,[2])

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
    """
    def __init__(self, J=1, wave='db1', mode='zero', separable=True):
        super().__init__()
        self.get_high_from_low = False
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 1:
                self.get_high_from_low = True
                h0_col, h1_col = wave[0], None
                h0_row, h1_row = None, None
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = None, None
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
            self.h0_col = filts[0]
            self.h1_col = filts[1]
            self.h0_row = filts[2]
            self.h1_row = filts[3]
        else:
            filts, self.h0_col, self.h1_col, self.h0_row, self.h1_row = lowlevel.prep_filt_afb2d_nonsep(
                                            h0_col, h1_col, h0_row, h1_row)

            self.h = filts
        self.J = J
        self.mode = mode
        self.separable = separable

    def get_filts(self):
        if self.get_high_from_low:
            self.h0_row = self.h0_col.reshape((1, 1, 1, -1))
            self.h1_col = get_highpass_filter(self.h0_col)
            self.h1_row = self.h1_col.reshape((1, 1, 1, -1))

        if self.separable:
            return (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        else:
            h0_col_f, h1_col_f, h0_row_f, h1_row_f = [h.flatten() for h in (self.h0_col, self.h1_col, self.h0_row, self.h1_row)]
            ll = torch.ger(h0_col_f, h0_row_f)
            lh = torch.ger(h1_col_f, h0_row_f)
            hl = torch.ger(h0_col_f, h1_row_f)
            hh = torch.ger(h1_col_f, h1_row_f)
            filts = torch.stack([ll[None], lh[None],
                                 hl[None], hh[None]], dim=0)
            return filts

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x

        filts = self.get_filts()
        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            if self.separable:
                #filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
                y = lowlevel.afb2d(ll, filts, self.mode)
            else:
                y = lowlevel.afb2d_nonsep(ll, filts, self.mode)

            # Separate the low and bandpasses
            s = y.shape
            y = y.reshape(s[0], -1, 4, s[-2], s[-1])
            ll = y[:,:,0].contiguous()
            yh.append(y[:,:,1:].contiguous())

        return ll, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero', separable=True):
        super().__init__()
        self.mode = mode
        self.separable = separable
        self.g0_col, self.g1_col, self.g1_row, self.g0_row, self.h = None, None, None, None, None

        if wave is None:
            return

        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = None, None
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.g0_col = filts[0]
            self.g1_col = filts[1]
            self.g0_row = filts[2]
            self.g1_row = filts[3]
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(
                g0_col, g1_col, g0_row, g1_row)
            self.h = nn.Parameter(filts, requires_grad=False)



    def forward(self, coeffs):
        return self.reconstruct(coeffs, self.g0_col, self.g1_col, self.g0_row, self.g1_row)

    def reconstruct(self, coeffs, g0_col, g1_col, g0_row, g1_row):
        return reconstruct(coeffs, g0_col, g1_col, g0_row, g1_row, self.h,  self.mode, self.separable)

    def rec_from_dec(self, coeffs, h0_col, h1_col, h0_row, h1_row):
        self.g0_col, self.g1_col, self.g0_row, self.g1_row = get_rec_filters(h0_col, h1_col, h0_row, h1_row)
        return self.reconstruct(coeffs, self.g0_col, self.g1_col, self.g0_row, self.g1_row )


def get_rec_filters(h0_col, h1_col, h0_row, h1_row):
    g0_col = h0_col #torch.flip(h0_col, [2])
    g1_col = h1_col #torch.flip(h1_col, [2])
    g0_row = h0_row #torch.flip(h0_row, [2])
    g1_row = h1_row #torch.flip(h1_row, [2])
    return g0_col, g1_col, g0_row, g1_row

def reconstruct(coeffs, g0_col, g1_col, g0_row, g1_row, h_filts,  mode, separable):
    """
    Args:
        coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
          yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
          W_{in}')` and yh is a list of bandpass tensors of shape
          :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
          the format returned by DWTForward

    Returns:
        Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

    Note:
        :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
        downsampled shapes of the DWT pyramid.

    Note:
        Can have None for any of the highpass scales and will treat the
        values as zeros (not in an efficient way though).
    """
    yl, yh = coeffs
    ll = yl

    # Do a multilevel inverse transform
    for h in yh[::-1]:
        if h is None:
            h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                            ll.shape[-1], device=ll.device)

        # 'Unpad' added dimensions
        if ll.shape[-2] > h.shape[-2]:
            ll = ll[...,:-1,:]
        if ll.shape[-1] > h.shape[-1]:
            ll = ll[...,:-1]

        # Do the synthesis filter banks
        if separable:
            lh, hl, hh = torch.unbind(h, dim=2)
            filts = (g0_col, g1_col, g0_row, g1_row)
            ll = lowlevel.sfb2d(ll, lh, hl, hh, filts, mode=mode)
        else:
            c = torch.cat((ll[:,:,None], h), dim=2)
            ll = lowlevel.sfb2d_nonsep(c, h_filts, mode=mode)
    return ll