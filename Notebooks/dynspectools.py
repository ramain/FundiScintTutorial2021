import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt

def readpsrarch(fname, dedisperse=True, verbose=True):
    """
    Read pulsar archive directly using psrchive
    Requires the python psrchive bindings, only working in python2

    Parameters
    ----------
    fname: string
    file directory
    dedisperse: Bool
    apply psrchive's by-channel incoherent de-dispersion

    Returns archive data cube, frequency array, time(mjd) array, source name
    """
    import psrchive
    
    arch = psrchive.Archive_load(fname)
    source = arch.get_source()
    tel = arch.get_telescope()
    if verbose:
        print("Read archive of {0} from {1}".format(source, fname))

    if dedisperse:
        if verbose:
            print("Dedispersing...")
        arch.dedisperse()
    data = arch.get_data()
    midf = arch.get_centre_frequency()
    bw = arch.get_bandwidth()
    F = np.linspace(midf-bw/2., midf+bw/2., data.shape[2], endpoint=False)
    #F = arch.get_frequencies()

    a = arch.start_time()
    t0 = a.strtempo()
    t0 = Time(float(t0), format='mjd', precision=0)

    # Get frequency and time info for plot axes
    nt = data.shape[0]
    Tobs = arch.integration_length()
    dt = (Tobs / nt)*u.s
    T = t0 + np.arange(nt)*dt
    T = T.mjd
    
    return data, F, T, source, tel


def clean_foldspec(I, plots=True, apply_mask=False, rfimethod='var', flagval=10, offpulse='True', tolerance=0.5, off_gates=0):
    """
    Clean and rescale folded spectrum
    
    Parameters
    ----------
    I: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots
    apply_mask: Bool
    Multiply dynamic spectrum by mask
    rfimethod: String
    RFI flagging method, currently only supports var
    tolerance: float
    % of subints per channel to zap whole channel
    off_gates: slice
    manually chosen off_gate region.  If unset, the bottom 50%
    of the profile is chosen

    Returns
    ------- 
    foldspec: folded spectrum, after bandpass division, 
    off-gate subtraction and RFI masking
    flag: std. devs of each subint used for RFI flagging
    mask: boolean RFI mask
    bg: Ibg(t, f) subtracted from foldspec
    bpass: Ibg(f), an estimate of the bandpass
    
    """

    # Sum to form total intensity, mostly a memory/speed concern
    if len(I.shape) == 4:
        print(I.shape)
        I = I[:,(0,1)].mean(1)

    # Use median over time to not be dominated by outliers
    bpass = np.median( I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass
    
    mask = np.ones_like(I.mean(-1))

    prof_dirty = (I - I.mean(-1, keepdims=True)).mean(0).mean(0)
    if not off_gates:
        off_gates = np.argwhere(prof_dirty<np.median(prof_dirty)).squeeze()
        recompute_offgates = 1
    else:
        recompute_offgates = 0

    if rfimethod == 'var':
        if offpulse:
            flag = np.std(foldspec[..., off_gates], axis=-1)
        else:
            flag = np.std(foldspec, axis=-1)
        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize//4), int(3*flagsize//4) )
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean+flagval*flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0)<tolerance] = 0
        mask[mask.mean(1)<tolerance] = 0
        if apply_mask:
            I[mask < 0.5] = np.mean(I[mask > 0.5])

    
    profile = I.mean(0).mean(0)

    # redetermine off_gates, if off_gates not specified
    if recompute_offgates:
        off_gates = np.argwhere(profile<np.median(profile)).squeeze()
    
    # renormalize, now that RFI are zapped
    bpass = I[...,off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = np.nanmean(foldspec)
    bg = np.mean(foldspec[...,off_gates], axis=-1, keepdims=True)
    foldspec = foldspec - bg
        
    if plots:
        plot_diagnostics(foldspec, flag, mask)

    return foldspec, flag, mask, bg.squeeze(), bpass.squeeze()


def rfifilter_median(dynspec, xs=20, ys=4, sigma=3., fsigma=5., tsigma=0., iters=3):
    """
    Flag hot pixels, as well as anomalous t,f bins in 
    a dynamic spectrum using a median filter
    
    Parameters
    ----------
    dynspec: ndarray [time, freq]
    xs: Filter size in time
    ys: Filter size in freq
    sigma: threshold for bad pixels in residuals
    fsigma: threshold for bad channels
    tsigma: threshold for bad time bins
    iters: int, number of iterations for median filter

    Returns
    ------- 
    ds_med: median filter of dynspec
    mask_filter: boolean mask of RFI
    
    """

    from scipy.ndimage import median_filter

    ds_med = median_filter(dynspec, size=[xs,ys])
    gfilter = dynspec-ds_med
    mask_filter = np.ones_like(gfilter)

    for i in range(iters):
        if i == 0:
            gfilter_masked = np.copy(gfilter)
        sigmaclip = np.nanstd(gfilter_masked)*sigma
        mask_filter[abs(gfilter)>sigmaclip] = 0
        gfilter_masked[abs(gfilter)>sigmaclip] = np.nan
        
    frac = 100.*(mask_filter.size - 1.*np.sum(mask_filter)) / mask_filter.size
    print('{0}/{1} = {2}% subints flagged '.format(
          int(mask_filter.size - 1.*np.sum(mask_filter)), mask_filter.size, frac))
        
    # Filter channels
    if fsigma > 0:
        nchan = gfilter.shape[1]
        gfilter_freq = np.nanstd(gfilter_masked, axis=0)
        gfilter_freq = gfilter_freq / np.nanmedian(gfilter_freq)
        chanthresh = fsigma*np.nanstd( np.sort(np.ravel(gfilter_freq))[nchan//8:7*nchan//8] )
        badchan = np.argwhere( abs(gfilter_freq-1) > chanthresh).squeeze()
        mask_filter[:,badchan] = 0
        gfilter_masked[:,badchan] = np.nan
        print('{0}/{1} channels flagged'.format(len(badchan), nchan))

    # filter bad time bins
    if tsigma > 0:
        ntime = gfilter.shape[0]
        gfilter_time = np.nanstd(gfilter_masked, axis=1)
        gfilter_time = gfilter_time / np.nanmedian(gfilter_time)
        timethresh = tsigma*np.nanstd( np.sort(np.ravel(gfilter_time))[ntime//8:7*ntime//8] )
        badtbins = np.argwhere( abs(gfilter_time-1) > timethresh).squeeze()
        mask_filter[badtbins] = 0
        gfilter_masked[badtbins] = np.nan
        print('{0}/{1} time bins flagged'.format(len(badtbins), ntime))

    return ds_med, mask_filter


def plot_diagnostics(foldspec, flag, mask):
    """
    Plot the outputs of clean_foldspec, and different axis summations of foldspec

    Parameters
    ----------
    foldspec: ndarray [time, freq, phase]
    flag: ndarray [time, freq], std. dev of each subint
    mask: ndarray [time, freq], boolean mask created from flag thresholds

    """
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(231)
    plt.plot(foldspec.mean(0).mean(0), color='k')
    plt.xlabel('phase (bins)')
    plt.ylabel('I (arb.)')
    plt.title('Pulse Profile')
    plt.xlim(0, foldspec.shape[-1])
    
    plt.subplot(232)
    plt.title('RFI flagging parameter (log10)')
    plt.xlabel('time (bins)')
    plt.ylabel('freq (bins)')
    plt.imshow(np.log10(flag).T, aspect='auto')

    plt.subplot(233)
    plt.title('Manual off-gate scaling')
    plt.imshow(mask.T, aspect='auto', cmap='Greys')
    plt.xlabel('time (bins)')
    
    plt.subplot(234)
    plt.imshow(foldspec.mean(0), aspect='auto')
    plt.xlabel('phase')
    plt.ylabel('freq')

    plt.subplot(235)
    plt.imshow(foldspec.mean(1), aspect='auto')
    plt.xlabel('phase')
    plt.ylabel('time')

    plt.subplot(236)
    plt.imshow(foldspec.mean(2).T, aspect='auto')
    plt.xlabel('time')
    plt.ylabel('freq')
    

def create_dynspec(foldspec, template=[1], profsig=5., bint=1, binf=1):
    """
    Create dynamic spectrum from folded data cube
    
    Uses average profile as a weight, sums over phase

    Returns: dynspec, np array [t, f]

    Parameters 
    ----------                                                                                                         
    foldspec: [time, frequency, phase] array
    template: pulse profile I(phase), phase weights to create dynamic spectrum
    profsig: S/N value, mask all profile below this (only if no template given)
    bint: integer, bin dynspec by this value in time
    binf: integer, bin dynspec by this value in frequency
    """
    
    # If no template provided, create profile by summing over time, frequency
    if len(template) <= 1:
        template = foldspec.mean(0).mean(0)
        template /= np.max(template)

        # Noise estimated from bottom 50% of profile
        tnoise = np.std(template[template<np.median(template)])
        # Template zeroed below threshold
        template[template < tnoise*profsig] = 0

    profplot2 = np.concatenate((template, template), axis=0)

    # Multiply the profile by the template, sum over phase
    dynspec = (foldspec*template[np.newaxis,np.newaxis,:]).mean(-1)

    if bint > 1:
        tbins = int(dynspec.shape[0] // bint)
        dynspec = dynspec[-bint*tbins:].reshape(tbins, bint, -1).mean(1)
    if binf > 1:
        dynspec = dynspec.reshape(dynspec.shape[0], -1, binf).mean(-1)

    return dynspec


def write_psrflux(dynspec, dynspec_errs, F, t, fname, psrname=None, telname=None, note=None):
    """
    Write dynamic spectrum along with column info in 
    psrflux format, compatible with scintools
    
    dynspec: ndarray [time, frequency]
    dynspec_errs: ndarray [time, frequency]
    F: astropy unit, channel frequency
    t: astropy Time values for each subintegration
    fname: filename to write psrflux dynspec to
    psrname: optional, string with source name
    telname: optional, string with telescope
    note: optional, note with additional information
    """
    T_minute = (t.unix - t[0].unix)/60.
    dt = (T_minute[1] - T_minute[0])/2.
    T_minute = T_minute + dt
    F_MHz = F.to(u.MHz).value
    with open(fname, 'w') as fn:
        fn.write("# Dynamic spectrum in psrflux format\n")
        fn.write("# Created using scintillation.dynspectools\n")
        fn.write("# MJD0: {0}\n".format(t[0].mjd))
        if telname:
            fn.write("# telescope: {0}\n".format(telname))
        if psrname:
            fn.write("# source: {0}\n".format(psrname))
        if note:
            fn.write("# {0}\n".format(note))
        fn.write("# Data columns:\n")
        fn.write("# isub ichan time(min) freq(MHz) flux flux_err\n")

        for i in range(len(T_minute)):
            ti = T_minute[i]
            for j in range(len(F)):
                fi = F_MHz[j]
                di = dynspec[i, j]
                di_err = dynspec_errs[i, j]
                fn.write("{0} {1} {2} {3} {4} {5}\n".format(i, j, 
                                            ti, fi, di, di_err) )
    print("Written dynspec to {0}".format(fname))


def read_psrflux(fname):
    """
    Load dynamic spectrum from psrflux file
    
    Skeleton from scintools
    
    Returns: 
    dynspec, dynspec_err, T, F, source
    """

    with open(fname, "r") as file:
        for line in file:
            if line.startswith("#"):
                headline = str.strip(line[1:])
                if str.split(headline)[0] == 'MJD0:':
                    # MJD of start of obs
                    mjd = float(str.split(headline)[1])
                if str.split(headline)[0] == 'source:':
                    # MJD of start of obs
                    source = str.split(headline)[1]
                if str.split(headline)[0] == 'telescope:':
                    # MJD of start of obs
                    telescope = str.split(headline)[1]
       
    try:
        source
    except NameError:
        source = ''
       
    data = np.loadtxt(fname)
    dt = int(np.max(data[:,0])+1)
    df = int(np.max(data[:,1])+1)
    
    t = data[::df,2]*u.min
    F = data[:df,3]*u.MHz
    dynspec = (data[:,4]).reshape(dt,df)
    dynspec_err = (data[:,5]).reshape(dt,df)
    T = Time(float(mjd), format='mjd') + t

    return dynspec, dynspec_err, T, F, source
