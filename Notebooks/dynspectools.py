import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def plot_secspec(dynspec, freqs, dt=4*u.s, xlim=None, ylim=None, bintau=2, binft=1, vm=3.,
                bint=1, binf = 1, pad=1, plot=True):

    """
    dynspec:  array with units [time, frequency]
    freqs: array of frequencies in MHz
    dt: Size of time bins, astropy unit
    xlim: xaxis limits in mHz
    ylim: yaxis limits in mus
    
    
    vm: dynamic range of secspec
    bintau:  Binning factor of SS in tau, for plotting purposes
    binft:  Binning factor of S in ft, for plotting purposes
    bint: Binning factor of dynspec in t, for plotting purposes
    binf: Binning factor of dynspec in f, for plotting purposes
    
    Returns:
    CS: 2D FFT of dynamic spectrum
    ft: ft axis of CS
    tau: tau axis of CS
    """
    
    # Get frequency and time info for plot axes
    bw = freqs[-1] - freqs[0]
    df = (freqs[1]-freqs[0])*u.MHz
    nt = dynspec.shape[0]
    T = (nt * dt).to(u.min).value
    
    # Bin dynspec in time, frequency 
    # ONLY FOR PLOTTING
    nbin = dynspec.shape[0]//bint
    dspec_plot = dynspec[:nbin*bint].reshape(nbin, bint, dynspec.shape[-1]).mean(1)
    if binf > 1:
        dspec_plot = dspec_plot.reshape(dspec_plot.shape[0],-1, binf).mean(-1)
    dspec_plot = dspec_plot / np.std(dspec_plot)
    
    # 2D power spectrum is the Secondary spectrum
    dynpad = np.pad(dynspec, pad_width=((dynspec.shape[0]*pad, 0), 
                                       (0, dynspec.shape[1]*pad)), 
                 mode='constant', constant_values=np.mean(dynspec)) 
    CS = np.fft.fft2(dynpad)
    S = np.fft.fftshift(CS)
    S = np.abs(S)**2.0
    
    # Bin in tau and FT - ONLY AFTER SQUARING
    Sb = S.reshape(-1,S.shape[1]//bintau, bintau).mean(-1)
    if binft > 1:
        nftbin = Sb.shape[0]//binft
        print(Sb.shape)
        Sb = Sb[:binft*nftbin].reshape(nftbin, binft, -1).mean(1)
    Sb = np.log10(Sb)
    
    # Calculate the confugate frequencies (time delay, fringe rate), only used for plotting
    ft = np.fft.fftfreq(S.shape[0], dt)
    ft = np.fft.fftshift(ft.to(u.mHz).value)

    tau = np.fft.fftfreq(S.shape[1], df)
    tau = np.fft.fftshift(tau.to(u.microsecond).value)    
    
    slow = np.median(Sb)-0.2
    shigh = slow + vm

    # Not the nicest, have a set of different plots it can produce
    if plot:
        plt.figure(figsize=(10,10))
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)

        plt.subplots_adjust(wspace=0.1)

        # Plot dynamic spectrum image

        ax2.imshow(dspec_plot.T, aspect='auto', vmax=7, vmin=-3, origin='lower',
                    extent=[0,T,min(freqs), max(freqs)], cmap='viridis')
        ax2.set_xlabel('time (min)', fontsize=16)
        ax2.set_ylabel('freq (MHz)', fontsize=16)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Plot Secondary spectrum
        ax3.imshow(Sb.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
                   extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
                  cmap='viridis')
        ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
        ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16) 

        if xlim:
            ax3.set_xlim(-xlim, xlim)
        if ylim:
            ax3.set_ylim(-ylim, ylim)
    return CS, ft, tau

def Gaussfit(dynspec, df, dt):
    
    """
    dynspec:  array with units [time, frequency]
    df: channel width, astropy unit
    dt: subint length, astropy unit
    
    Returns:
    CS: 2D FFT of dynamic spectrum
    ft: ft axis of CS
    tau: tau axis of CS
    """
    
    ccorr = np.fft.ifft2( np.fft.fft2(dynspec) * np.fft.fft2(dynspec).conj() )
    ccorr = ccorr - np.median(ccorr)
    
    # Ignoring zero component with noise-noise correlation
    ccorr_f = abs(ccorr[1]) + abs(ccorr[-1])
    ccorr_f /= np.max(ccorr_f)
    
    ccorr_t = abs(ccorr[:,1]) + abs(ccorr[:,-1])
    ccorr_t /= np.max(ccorr_t)

    ft = np.fft.fftfreq(dynspec.shape[0], dt)
    ft = np.fft.fftshift(ft.to(u.mHz).value)
    
    tau = np.fft.fftfreq(dynspec.shape[1], df)
    tau = np.fft.fftshift(tau.to(u.microsecond).value)
    
    df_axis = np.fft.fftfreq(dynspec.shape[1], d=(tau[1]-tau[0]) )
    dt_axis = np.fft.fftfreq(dynspec.shape[0], d=(ft[1]-ft[0])*u.mHz ).to(u.min).value
    
    # Starting guess, currently hardcoded
    p0 = [5., 1, 0]
    popt, pcov = curve_fit(Gaussian, df_axis, ccorr_f, p0=p0)

    nuscint = abs(popt[0])
    nuscint_err = np.sqrt(pcov[0,0])
    
    fscint =  np.sqrt(2*np.log(2)) * nuscint
    fscinterr =  np.sqrt(2*np.log(2)) * nuscint_err
    
    # Starting guess, currently hardcoded
    pT = [20, 1, 0]
    poptT, pcovT = curve_fit(Gaussian, dt_axis, ccorr_t, p0=p0)
    
    tscint = np.sqrt(2) * abs(poptT[0]) * 60.
    tscinterr = np.sqrt(2) * np.sqrt(pcov[0,0]) * 60.

    # Compute "finite scintle error"
    # THIS MAY BE BUGGY, I NEED TO TEST
    Tobs = dynspec.shape[0] * dt.value / 60.
    BW = dynspec.shape[1] * df.value
    fillfrac = 0.2
    
    fin_scinterr = (1 + fillfrac * BW / nuscint) * (1 + fillfrac* Tobs / tscint)

    tscinterr = np.sqrt( tscinterr**2 + tscint/fin_scinterr  )
    fscinterr = np.sqrt( fscinterr**2 + fscint/fin_scinterr  )

    ccorr = abs(ccorr)

    vmax = np.mean(ccorr) + 10*np.std(ccorr)
    vmin = np.mean(ccorr) - 3*np.std(ccorr)
    
    plt.figure(figsize=(8,8))

    ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
    ax3 = plt.subplot2grid((4, 4), (0, 0), colspan=3)

    plt.subplots_adjust(wspace=0.05)
    
    ax1.imshow(np.fft.fftshift(ccorr).T, aspect='auto', origin='lower',
              extent=[min(dt_axis), max(dt_axis), min(df_axis), max(df_axis)],
              vmax=vmax, vmin=vmin, cmap='Greys')

    ax1.set_xlabel('dt (min)', fontsize=16)
    ax1.set_ylabel(r'd$\nu$ (MHz)', fontsize=16)

    df_shifted = np.fft.fftshift(df_axis)
    dt_shifted = np.fft.fftshift(dt_axis)
    ax2.plot( np.fft.fftshift(ccorr_f), df_shifted, color='k')
    ax2.plot(Gaussian(df_shifted, *popt), df_shifted, color='tab:red',
            linestyle='--')

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    #ax2.set_ylabel(r'$d\nu$ (MHz)', fontsize=16)
    ax2.set_xlabel(r'I (d$\nu$, dt=0)', fontsize=16)
    ax2.set_ylim(min(df_axis), max(df_axis) )

    ax3.plot( dt_shifted, np.fft.fftshift(ccorr_t), color='k')
    ax3.plot( dt_shifted, Gaussian(dt_shifted, *poptT), color='tab:red',
              linestyle='--')
    ax3.set_ylabel(r'I (dt, d$\nu$=0)', fontsize=16)
    ax3.set_xlim(min(dt_axis), max(dt_axis))
    
    return np.fft.fftshift(ccorr), fscint, fscinterr, tscint, tscinterr

def Gaussian(x, sigma, A, C):
    return A*np.exp( -x**2 / (2*sigma**2) ) + C
