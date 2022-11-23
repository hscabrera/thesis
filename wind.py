import moby2
from matplotlib import pyplot as plt
from moby2.analysis.tod_ana import pathologies
from scipy.signal import correlate, argrelextrema
import scipy.optimize
import numpy as np
import time, os
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('x-small')
fb = moby2.scripting.get_filebase()


toddy = '1477177586.1477188934.ar3'
tag_cuts = 'mr3_pa3_s16'
tag_cal = 'mr3_pa3_s16'
depot_path = '/data/actpol/depot'
freqs = [90, 150]
f_min = 0.1
f_max = 5.
resample=8
vlim=0.05
sigma=2.

def process_tod(toddy=toddy, tag_cuts=tag_cuts, tag_cal=tag_cal, depot_path=depot_path, 
                freqs=freqs, f_min=f_min, f_max=f_max, plot=False, outdir=None, 
                sigma=sigma, resample=resample, vlim=vlim, **kwargs):
    tod, lefts, rights, ld = preprocess_tod(toddy, tag_cuts, tag_cal, depot_path,
                       f_min=f_min, f_max=f_max, resample=resample, vlim=vlim)
    res = {}
    for f in freqs:
        res[f] = {}
        ldf = select_frequency(tod, ld, f)
        dist, ang = get_detector_vector(tod, ldf)
        for seg, scan in zip([lefts, rights], ["left", "right"]):
            lags, amps, cubes, taus = process_scans(tod, seg, ldf, **kwargs)
            speeds, speeds_std, phases, phases_std, fits, sels = get_array_speed2(lags, 
                                              dist, ang, full=True, sigma=sigma, **kwargs)
            if plot:
                for i in range(len(speeds)):
                    s = sels[i]
                    plot_phase_fit(ang[s], dist[s], lags[i][s], speeds[i], phases[i], bins=10, 
                                   perc=5)
                    plt.title("%s %s %g GHz" % (toddy, scan, f))
                    plt.savefig(os.path.join(outdir, 
                                    "speed_fit_%s%02d_%s_%g.png" % (scan, i, toddy, f)))
                    plot_phase_fit_bins(ang[s], dist[s], lags[i][s], speeds[i], phases[i], 
                                        bins=10, perc=5)
                    plt.title("%s %s %g GHz" % (toddy, scan, f))
                    plt.savefig(os.path.join(outdir, 
                                    "speed_fit_bins_%s%02d_%s_%g.png" % (scan, i, toddy, f)))
                    plot_amp_array(tod.info.array_data, ldf, amps[i])
                    plt.title("%s %s %g GHz" % (toddy, scan, f))
                    plt.savefig(os.path.join(outdir, 
                                     "amp_map_%s%02d_%s_%g.png" % (scan, i, toddy, f)))
                    plt.close("all")
            res[f].update({"%s speed" % scan: speeds,
                           "%s speed std" % scan: speeds_std,
                           "%s phase" % scan: phases,
                           "%s phase std" % scan: phases_std})
    res["min az"] = tod.az.min()
    res["max az"] = tod.az.max()
    res["elevation"] = tod.alt.mean()
    res["date"] = time.asctime(time.gmtime(tod.ctime[0]))
    res["scan speed"] = tod.scan_speed
    return res


def get_wind_speed(res, h=100, freqs=freqs):
    a1 = res["min az"]
    a2 = res["max az"]
    alt = res["elevation"]
    da = res["scan speed"]
    cos21 = np.cos(a2)-np.cos(a1)
    sin21 = np.sin(a2)-np.sin(a1)
    A = np.array([[             sin21,             cos21],
                  [np.cos(alt)*cos21, -np.sin(alt)*sin21]])
    factor = (a2-a1)*h / np.sin(alt)
    A1 = factor * np.linalg.inv(A)
    wind={}
    for f in freqs:
        Vl, Covl = project_array_vel(res[f]["left speed"], res[f]["left speed std"],
                                    res[f]["left phase"], res[f]["left phase std"])
        Vr, Covr = project_array_vel(res[f]["right speed"], res[f]["right speed std"],
                                    res[f]["right phase"], res[f]["right phase std"])
        Vl[0] -= da*np.cos(alt)
        Vr[0] += da*np.cos(alt)
        Wl = np.dot(A1, Vl)
        Wr = np.dot(A1, Vr)
        Wl_cov = np.swapaxes(np.dot(np.dot(A1, Covl.T), A1.T), 0, 1) 
        Wr_cov = np.swapaxes(np.dot(np.dot(A1, Covr.T), A1.T), 0, 1) 
        wind[f] = {"left wind": Wl,
                   "right wind": Wr,
                   "left wind cov": Wl_cov,
                   "right wind cov": Wr_cov}
    return wind



def project_array_vel(speed, speed_std, phase, phase_std):
    """
    Recibe velocidades en grados por segundo y las entrega en radianes por segundo
    """
    sp = np.deg2rad(speed)
    dsp = np.deg2rad(speed_std)
    V = np.array([sp * np.sin(phase), sp * np.cos(phase)])
    c11 = (dsp * np.sin(phase))**2 + (sp * np.cos(phase) * np.array(phase_std))**2
    c22 = (dsp * np.cos(phase))**2 + (sp * np.sin(phase) * np.array(phase_std))**2
    c12 = (dsp**2 + sp**2 * np.array(phase_std)**2) * np.abs(np.sin(phase)*np.cos(phase))
    cov = np.array([[c11, c12],
                    [c12, c22]])
    return V, cov



def world2tel(q, az, alt):
    R = np.array([[            np.cos(az),            -np.sin(az),            0],
                  [np.sin(az)*np.sin(alt), np.cos(az)*np.sin(alt), -np.cos(alt)],
                  [np.sin(az)*np.cos(alt), np.cos(az)*np.cos(alt),  np.sin(alt)]])
    return np.dot(R, q)


def preprocess_tod(toddy, tag_cuts, tag_cal, depot_path, f_min=None, f_max=None,
                   resample=resample, vlim=vlim):
    # Open TOD
    obs = str(toddy)                  
    filename = fb.filename_from_name(obs,single=True)   
    cuts = moby2.scripting.get_cuts(            
        {'depot':depot_path,                
         'tag':tag_cuts},              
        tod=obs)                        
    tod = moby2.scripting.get_tod(          
        {'filename':filename,               
         'repair_pointing':True,
         'fix_sign':True,
         "start":cuts.sample_offset})              
    moby2.tod.remove_mean(tod)              
    # Get cuts
    tod.cuts = cuts
    moby2.tod.cuts.fill_cuts(tod=tod)
    ld = cuts.get_uncut()               
    # Downsample TOD
    tod = tod.copy(resample=resample)
    tod.dt = np.diff(tod.ctime).mean() 
    # Filter TOD
    if f_min is not None or f_max is not None:
        if f_min is not None:
            filt = moby2.tod.filters.sine2highPass(tod=tod, fc=f_min, df=f_min/10)
            if f_max is not None:
                filt *= moby2.tod.filters.sine2lowPass(tod=tod, fc=f_max, df=f_max/10)
        else:
            filt = moby2.tod.filters.sine2lowPass(tod=tod, fc=f_max, df=f_max/10)
        tod.data[ld] = moby2.tod.filter.apply_simple(tod.data[ld], filt, detrend=True)
    # Analyze scan
    swipes = pathologies.analyzeScan(tod.az, dt=tod.dt, vlim=vlim)
    tod.scan_speed = swipes["az_speed"] / tod.dt
    mins, maxes = pivots(argler(tod.az, switch = True),swipes['pivot'],True,argler(tod.az))         
    if mins[0] < maxes[0]:
        lefts = [[mins[i], maxes[i]] for i in range(len(maxes))]
        rights = [[maxes[i], mins[i+1]] for i in range(len(mins)-1)]
    else:
        lefts = [[mins[i], maxes[i+1]] for i in range(len(maxes)-1)]
        rights = [[maxes[i], mins[i]] for i in range(len(mins))]
    return tod, lefts, rights, ld


def process_scans(tod, segments, ld, **kwargs):
    amps = []
    lags = []
    cubes = []
    taus = []
    for s in segments:
        cube, tau = get_corr(tod, s, ld, **kwargs)
        lag, amp = get_lag(cube, tau)
        amps.append(amp)
        lags.append(lag)
        cubes.append(cube)
        taus.append(tau)
    lags = np.array(lags)
    amps = np.array(amps)
    cubes = np.array(cubes)
    taus = np.array(taus)
    return lags, amps, cubes, taus


def get_corr(tod, segment, ld, ds=1, dN=120, offset=10, **kwargs):
    s = np.linspace(0, dN*ds, dN+1, dtype=int)
    A = tod.data[ld][:,segment[0]+offset:segment[1]-offset-s[-1]]
    norm_A = np.linalg.norm(A, axis=1)
    cube = []
    for ss in s:
        B = tod.data[ld][:,int(segment[0]+offset+ss):int(segment[1]-offset+ss-s[-1])]
        norm_B = np.linalg.norm(A, axis=1)
        corr = np.dot(A, B.T) / np.outer(norm_A, norm_B)
        cube.append(corr)
    cube = np.array(cube).T
    tau = s * tod.dt
    return cube, tau
 
def get_lag(cube, tau):
    """
    Gets the lag and correlation amplitude directly. Fast
    """
    imax = cube.argmax(axis=2)
    lags = tau[imax]
    amps = cube.max(axis=2)
    side = amps - amps.T < 0
    amps[side] = amps.T[side]
    lags[side] = -lags.T[side]
    return lags, amps


def get_lag_fit(cube, tau, window=2, oversamp=10):
    """
    Gets the lag and correlation amplitude by interpolating between points. Slow.
    """
    taus = np.hstack([np.flipud(-tau), tau[1:]])
    N = cube.shape[0]
    lags = np.zeros([N,N])
    amps = np.ones([N,N])
    for i in range(N):
        for j in range(i):
            c = np.hstack([np.flipud(cube[i,j,:]), cube[j, i, 1:]])
            imax = np.argmax(c)
            inds = [max([0, int(imax - window/2)]), min(int(imax + window/2), len(c))]
            itaus = np.linspace(taus[inds[0]], taus[inds[1]], window*oversamp + 1)
            #q = interp1d(taus[inds[0]:inds[1]+1], c[inds[0]:inds[1]+1], "quadratic") 
            #corrs = q(itaus)
            coeff = np.polyfit(taus[inds[0]:inds[1]+1], c[inds[0]:inds[1]+1],2)
            corrs = np.polyval(coeff, itaus)
            qmax = np.argmax(corrs)
            tau_max = itaus[qmax]
            corr_max = corrs[qmax]
            lags[i, j] = -tau_max
            lags[j, i] = tau_max
            amps[i, j] = corr_max
            amps[j, i] = corr_max
    return lags, amps


def get_array_speed2(lags, dist, ang, perc=5, sigma=3., full=False, **kwargs):
    d = np.sort(dist.flatten())
    d_min = d[int(len(d)*perc/100)]
    fits = []
    speeds = []
    speeds_std = []
    phases = []
    phases_std = []
    sel = (dist > d_min)
    sels = []
    for lag in lags:
        res = fit_sin(ang[sel], lag[sel]/dist[sel])
        resid = lag[sel] - sinfunc(ang[sel], res["amp"], res["phase"])*dist[sel] 
        sel2 = np.zeros(sel.shape, dtype=bool)
        sel2[sel] = np.abs(resid) < sigma * resid.std()
        res = fit_sin(ang[sel2], lag[sel2]/dist[sel2])
        amp_err, phase_std = np.sqrt(np.diag(res["cov"]))
        speed_std = amp_err / res["amp"]**2
        sels.append(sel2)
        fits.append(res)
        speeds.append(1 / res["amp"])
        phases.append(res["phase"])
        speeds_std.append(speed_std)
        phases_std.append(phase_std)
    if full:
        return speeds, speeds_std, phases, phases_std, fits, sels
    else:
        return speeds, speeds_std, phases, phases_std


def get_array_speed(lag_m, dist, ang, bins=10, perc=5, plot=False, full=False, **kwargs):
    d = np.sort(dist.flatten())
    d_min = d[int(len(d)*perc/100)]
    d_max = d[int(len(d)*(1 - perc/100))]
    d_bins = np.linspace(d_min, d_max, bins+1)
    fits = []
    speeds = []
    phases = []
    for i in range(bins):
        sel = (dist > d_bins[i]) * (dist <= d_bins[i+1])
        res = fit_sin(ang[sel], lag_m[sel])
        fits.append(res)
        sp = (d_bins[i] + d_bins[i+1]) / 2 / res["amp"]
        speeds.append(sp)
        phases.append(res["phase"])
    speed = np.mean(speeds)
    speed_std = np.std(speeds)
    phase = np.mean(phases)
    phase_std = np.std(phases)
    if full:
        return speed, speed_std, phase, phase_std, speeds, phases, fits
    else:
        return speed, speed_std, phase, phase_std


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters 
       "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    guess_amp = np.std(yy) * np.sqrt(2)
    #guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 0.])
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, p = popt
    res = {"amp": A, 
           "phase": p, 
           #"offset": c, 
           "cov": pcov, 
           "guess": guess}
    return res

def sinfunc(t, A, p):  
    return A * np.sin(t + p) 

def pivots(lleft, pivot, right, *args):
    markers = np.array(lleft)
    markers = markers[markers >= pivot]
    mask = (markers[1:]-markers[:-1]) > 50
    markers = markers[np.append(mask,True)]
    mask = np.isclose((markers[1:]-markers[0])/float(max(markers[1:]-markers[:-1]))%1,1,atol = 0.01) + ((markers[1:]-markers[0])/float(max(markers[1:]-markers[:-1]))%1 ==0)
    markers = markers[np.append(True,mask)]
    if right:
        return markers, pivots(args,markers[0],False)
    else:
        return markers


def argler(lista, switch = False):              #DETERMINE POSITIONS FOR BEGINNING/ENDS OF EACH SCAN SWIPE /SCAN PREP
    if switch:
        values = argrelextrema(lista,np.less_equal,order=1)[0]
    else:
        values = argrelextrema(lista,np.greater_equal,order=1)[0]
    return values


def select_frequency(tod, ld, freq):
    sel = tod.info.array_data["nom_freq"] == freq
    sel = sel[ld]
    return ld[sel]


def get_detector_vector(tod, ld):
    x = tod.info.array_data["sky_x"][ld]
    y = tod.info.array_data["sky_y"][ld]
    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    dx = x1 - x2
    dy = y1 - y2
    dist = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)
    return dist, ang


def plot_amp_array(array_data, ld, amps, sel=None, ax=None):
    if sel is None:
        sel = np.ones(amps.shape, dtype=bool)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot()
    x = array_data["sky_x"][ld] 
    y = array_data["sky_y"][ld] 
    for i in range(len(x)): 
        plt.scatter(x[sel[i]]-x[i], y[sel[i]]-y[i], c=amps[i][sel[i]], alpha=0.1)
    plt.colorbar(shrink=0.8)
    plt.xlabel("delta az [deg]")
    plt.ylabel("delta el [deg]")
    return ax


def plot_phase_fit(ang, dist, lag, speed, phase, bins=10, perc=5):
    d = np.sort(dist.flatten())
    d_min = d[int(len(d)*perc/100)]
    sel = (dist > d_min)
    ph = (np.arange(21) / 20 - 0.5) * 2 * np.pi
    plt.figure()
    ax = plt.gca()
    ax.plot(ang[sel], lag[sel]/dist[sel], ".")
    ax.plot(ph, sinfunc(ph, 1./speed, phase), "k")
    ax.set_xlabel("Pair angle [rad]")
    ax.set_ylabel("lag / dist [s/deg]")
    return ax


def plot_phase_fit_bins(ang, dist, lag, speed, phase, bins=10, perc=5):
    # Plot in bins
    d = np.sort(dist.flatten())
    d_min = d[int(len(d)*perc/100)]
    d_max = d[int(len(d)*(1 - perc/100))]
    d_bins = np.linspace(d_min, d_max, bins+1)
    ph = (np.arange(21) / 20 - 0.5) * 2 * np.pi
    plt.figure()
    ax = plt.gca()
    for i in range(bins):
        sel = (dist > d_bins[i]) * (dist <= d_bins[i+1])
        d = (d_bins[i] + d_bins[i+1]) / 2
        ax.plot(ang[sel], lag[sel], ".")
        ax.plot(ph, sinfunc(ph, d / speed, phase), "k")
        ax.set_xlabel("Pair angle [rad]")
        ax.set_ylabel("lag [s]")
    return ax


def plot_cube(cube, tau, d1, d2, **kwargs):
    plt.figure()
    ax = plt.subplot()
    for i, j in zip(d1, d2):
        if i <= j:
            plt.plot(tau, cube[i, j], **kwargs)
            plt.plot(-tau, cube[j, i], color=ax.lines[-1].get_color(), **kwargs)
        else:
            plt.plot(tau, cube[j, i], **kwargs)
            plt.plot(-tau, cube[i, j], color=ax.lines[-1].get_color(), **kwargs)
    plt.xlabel("Tau [sec]")
    plt.ylabel("Correlation")


def plot_winds(wind, freqs=freqs, symbols=["o", "+"]):
    plt.figure()
    for f, sym in zip(freqs, symbols):
        for scan, color in zip(["left", "right"], ["b", "r"]):
            v = wind[f]["%s wind" % scan]
            cov = wind[f]["%s wind cov" % scan]
            dvx = np.sqrt(cov[:,0,0])
            dvy = np.sqrt(cov[:,1,1])
            plt.errorbar(v[0], v[1], xerr=dvx, yerr=dvy, fmt=".", color=color)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlabel("Vx")
    plt.ylabel("Vy")
    plt.grid()
    

def plot_array_results(res):
    fig = plt.figure()
    ax = plt.subplot()
    alt = res["elevation"]
    da = res["scan speed"]
    for f in freqs:
        Vl, Covl = project_array_vel(res[f]["left speed"], res[f]["left speed std"],
                                    res[f]["left phase"], res[f]["left phase std"])
        Vr, Covr = project_array_vel(res[f]["right speed"], res[f]["right speed std"],
                                    res[f]["right phase"], res[f]["right phase std"])
        dvxl = np.sqrt(Covl[0,0,:])
        dvyl = np.sqrt(Covl[1,1,:])
        dvxr = np.sqrt(Covr[0,0,:])
        dvyr = np.sqrt(Covr[1,1,:])
        plt.errorbar(Vl[0], Vl[1], xerr=dvxl, yerr=dvyl, fmt=".", label="Vl%d"%f)
        plt.errorbar(Vr[0], Vr[1], xerr=dvxr, yerr=dvyr, fmt=".", label="Vr%d"%f)
        plt.errorbar(Vl[0] - da*np.cos(alt), Vl[1], xerr=dvxl, yerr=dvyl, 
                     fmt=".", label="Vl%d - scan"%f)
        plt.errorbar(Vr[0] + da*np.cos(alt), Vr[1], xerr=dvxr, yerr=dvyr, 
                     fmt=".", label="Vr%d - scan"%f)
        plt.grid()
        ax.set_aspect("equal")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
       
    

