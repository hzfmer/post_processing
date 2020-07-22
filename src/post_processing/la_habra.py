# -*- coding: utf-8 -*-
"""A few subroutines to be used for post-processing
TODO:
    1. Adjust how to imshow maps
    2. Correct the way incrementally pick psa/vel
    3. use multiprocessing to accelerate?

Note:
    site_idx: orig is left upper, x goes toward southeast, y points northeast
    source_idx: sx along northeast; sy along southeast
    topo/vs30: orig the same as site_idx
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import PowerNorm
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import struct
import imageio
import collections
import pickle
import sys
import re
import os
import requests
import subprocess
from pathlib import Path
from filter_BU import filt_B
import my_pyrotd
from post_processing import __version__

mpl.rcParams['figure.dpi'] = 600

__author__ = "Zhifeng Hu"
__copyright__ = "Zhifeng Hu, SDSU/UCSD"
__license__ = "mit"


def force_iterable(x):
    return x if isinstance(x, collections.abc.Iterable) or isinstance(x, str) else [x]

def call_sub(cmd, shell=False):
    p = subprocess.run(cmd, shell=shell, capture_output=True)
    if p.stderr:
        print("Error\n", p)
    return p


def read_param(model, conf_file='param.sh'):
    '''Read parameters from local "param.sh"
    '''
    tmax = dt = tskip = wstep = nfile = 1
    with open(Path(model, conf_file), 'r') as fid:
        for line in fid:
            if "TMAX" in line:
                tmax = float(re.findall(r'(?<=TMAX )\d+\.+\d+(?= )', line)[0])
            if "DT" in line:
                dt = float(re.findall(r'(?<=DT )\d+\.+\d+(?= )', line)[0])
            if "NTISKP" in line:
                tskip = int(re.findall(r'NTISKP (\d+)', line)[0])
            if "WRITE_STEP" in line:
                wstep = int(re.findall(r'WRITE_STEP (\d+)', line)[0])
    nfile = int(tmax / dt) // wstep
    return tmax, dt, tskip, wstep, nfile


def filt(vel, fs=1, lowcut=0.15, highcut=5, causal=False):
    '''Filter data
    Input:
        vel     dict.keys=['dt', 'X', 'Y', 'Z'], or list
        rest    parameters for Butterworth filters.
    '''
    if not issubclass(type(vel), dict):
        res = filt_B(vel, fs, lowcut, highcut, causal=causal)
        return res
    data = {}
    for comp in vel.keys():
        if comp in 'XYZ':
            data[comp] = filt_B(np.array(vel[comp], dtype='float32'), 1 / vel['dt'], lowcut=lowcut, highcut=highcut, causal=causal)
        else:
            data[comp] = vel[comp]
    return data


def rotate(vel, angle):
    '''Rotate the horizontal velocities by angel degree
    Input
        vel:   dict with 'X' and 'Y' components or list of [velx, vely]
    '''
    if issubclass(type(vel), dict):
        vel['X'], vel['Y'] = rotate([vel['X'], vel['Y']], angle)
        return vel
    angle = np.radians(angle)
    vx, vy = vel[0], vel[1]
    vx, vy = vx * np.cos(angle) - vy * np.sin(angle), vx * np.sin(angle) + vy * np.cos(angle)
    return vx, vy


def check_mesh_cont(fmesh_0, fmesh_1, nx, ny, nz, verbose=False, nvar=3, skip=3):
    '''Check continuity between upper and lower blocks
    Input:
        fmesh_0: Upper block
        fmesh_1: Lower block
        nx, ny, nz: 3D dimensions of the upper block
        nvar: Number of variables in the block mesh
        skip: Ratio of dimensions between lower and upper block
    '''
    max_diff = 0
    with open(fmesh_0, 'rb') as f0, open(fmesh_1, 'rb') as f1:
        f0.seek(4 * nvar * nx * ny * (nz - 8), 0)
        for _ in range(3):       
            data0 = np.frombuffer(f0.read(4 * nvar * nx * ny),
                                  dtype='float32').reshape(ny, nx, nvar)
            data1 = np.frombuffer(f1.read(4 * nvar * nx * ny // skip ** 2),
                                  dtype='float32').reshape(ny//skip, nx//skip, nvar)
            diff = data0[1::skip, ::skip, :] - data1
            max_diff = max(np.max(diff), max_diff)
            #print(data0[:5, :5, 1], data1[:2, :2, 1])
            if not np.isclose(diff, 0).all():
                loc_y, loc_x, loc_z = np.unravel_index(
                                      np.argmax(np.abs(diff)), diff.shape)
                if verbose:
                    print(f"Not consistent, max difference at {np.argmin(diff)}: {np.min(diff)}")
                    print(f"({loc_y}, {loc_x}, {loc_z}) in {diff.shape}")
                    print("Top block: ", data0[1 + skip * loc_y, skip * loc_x, loc_z])
                    print("Bottom block: ", data1[loc_y, loc_x, loc_z])
            f0.seek(4 * nvar * nx * ny * 2, 1)
    if np.isclose(0, max_diff):
        print("Top and bottom blocks are consistent!")   
    else:
        im=plt.imshow(diff[:, :, loc_z], cmap='RdBu')
        plt.colorbar(im)
        print(f"Top and bottom blocks are not consistent! Max_diff = {max_diff}")
    return True


def read_snapshot(it, mx, my, model="", case="", comp="X"):
    '''Read wave field snapshot
    Input:
        it: time step to read
        mx, my: Surface 2D dimensions
        model: the name of the model
        case: different cases of a model, if exists
    '''
    tmax, dt, tskip, wstep, _ = read_param(model)
    nt = int(tmax / dt)
    model = Path(model, "output_sfc" if not case else f"output_sfc_{case}")
    fnum = int((it - 1) // (wstep * tskip) + 1) * wstep * tskip
    print(f'\r{it} / {nt}, {model}/S{comp}_0_{fnum:07d}', end="\r", flush=True)
    fid = open(f'{model}/S{comp}_0_{fnum:07d}', 'rb')
    skip = 4 * ((it - fnum) // tskip + wstep) * mx * my
    fid.seek(skip, 0)
    v = np.frombuffer(fid.read(mx * my * 4), dtype='float32').reshape(my, mx)
    v = v.copy()
    if np.isnan(v).any():
        print(f"NAN founded\n")
    return v, dt


def interp_vel(vel, dt):
    told = np.arange(len(vel['X'])) * vel['dt']
    tnew = np.arange(0, len(vel['X']) * vel['dt'], dt)
    for comp in 'XYZ':
        v = vel[comp]
        f = interp1d(told, v, fill_value='extrapolate')
        vel[comp] = f(tnew)
    vel['dt'] = dt
    return vel


def trim_vel(vel, tinit, tend):
    t = np.arange(len(vel['X'])) * vel['dt']
    idx = np.argwhere(np.logical_and(t >= tinit, t <= tend))
    for comp in 'XYZ':
        vel[comp] = np.squeeze(vel[comp][idx])
    return vel


def taper_vel(vel, seconds=2):
    """
    This function tapers the last n seconds of vel using a hanning window.
    """
    npts = int(2 * seconds / vel['dt'] +1)
    window = np.hanning(npts)[-int(seconds / vel['dt'] + 1):]
    for comp in 'XYZ':
        vel[comp][-len(window):] *= window
    return vel


def pad_vel(vel, pad_seconds):
    """
    Pad pad_seconds zeros at the end
    """
    npts = int(pad_seconds / vel['dt'])
    for comp in 'XYZ':
        shape = list(vel[comp].shape)
        vel[comp] = np.pad(vel[comp], (0,npts))
    return vel

def prepare_bbpvel(vel, tend, shift=0, dt=None,
        taper_seconds=2, pad_seconds=5):
    '''Prepare vel following bbp approach
    1. 10Hz low-pass filter, 4 poles, filtfilt
    2. Interpolate using 1D linear interpolation method, if dt applied
    3. Trim longest waveforms to match the shortest one
    2. Tapering the last 2 second waveforms
    3. Pad 5 seconds zeros
    4. Filter to desired frequency

    Reference
    ---------
    https://github.com/SCECcode/HighF
    '''
    vel = filt(vel, 1 / vel['dt'], highcut=10)
    if dt is not None:
        vel = interp_vel(vel, dt)
    vel = trim_vel(vel, shift, tend + shift)
    vel = taper_vel(vel, taper_seconds)
    vel = pad_vel(vel, pad_seconds)
    return vel


def read_rec(site, metric='vel', base_shift=582.97):
    '''Read recordings from USC database
    Input:
        site, site name
        base_shift, correct start time, supposed to be 04:09:42.97
        Ref: https://scec.usc.edu/scecpedia/La_Habra_Simulations_on_Titan
    Return:
        data: Dictionary of "metric" 
    '''

    r = requests.get(f'http://hypocenter.usc.edu/research/High-F/lahabra_obs_2019_05_31/{site}.V2')
    if r.status_code != 200:
        print(f"the site {site} not found on the server")
        return None
    recordings = r.text.split('\r\n')

    stime = re.findall(r'\d+:\d+:\d+', recordings[4].split(',')[1])[0].split(':')
    data = {}
    min_len = float('inf')
    for i in range(len(recordings)):
        if 'Chan' in recordings[i] and re.findall(r'^Chan  \d:.*', recordings[i]):
            chan = re.findall(r'^Chan  \d:.*', recordings[i])[0].split()[2]
        # Find lines starting with "7600 points of accel data equally"
        if "points" in recordings[i] and metric in recordings[i]:    
            nums = re.findall(r'\d*\.*\d+', recordings[i])
            if len(nums) <= 1:  # skip headers
                continue
            npts = int(nums[0])
            dt_rec = float(nums[1])        
            start = i + 1
            end = i + (npts - 1) // 8 + 2
            # convert cm/s/s to m/s/s
            tmp = np.array([float(x) / 100 for y in recordings[start : end] \
                                           for x in y.replace('-', ' -').split()])
            # In case records have different length in different components, just crop
            min_len = min(min_len, len(tmp))
            tmp = tmp[:min_len]
            if "Up" not in chan:
                chan = int(chan) if chan.isdigit() else (0 if 'X' in data else 90)
                data['X'] = tmp * np.sin(np.radians(int(chan))) + (data['X'][:min_len] if 'X' in data else 0)
                data['Y'] = tmp * np.cos(np.radians(int(chan))) + (data['Y'][:min_len] if 'Y' in data else 0)
            else:
                data['Z'] = tmp
    assert len(data) == 3
    for comp in 'XYZ':
        data[comp] = data[comp][:min_len]
    print(f"Number of points = {min_len}, dt_recording = {dt_rec}")
    data["dt"] = dt_rec
    data['shift'] = base_shift - (float(stime[-2]) * 60 + float(stime[-1]))
    return data


def read_syn(ix, iy, mx, my, model="", case=""):
    '''Read synthetics
    Input:
        ix, iy: Indices of site
        mx, my: Surface dimensions
    '''

    data = collections.defaultdict(list)
    _, dt, tskip, wstep, nfile = read_param(model)
    dy_syn = dt * tskip  # The synthetics are with larger dt, scaled by tskip
    print(model, dt_syn, tskip, wstep, nfile)

    model = Path(model, "output_sfc" if not case else f"output_sfc_{case}")
    skips = [4 * (j * my * mx + iy * mx + ix) for j in range(wstep)]
    for comp in ["X", "Y", "Z"]:
        for i in range(1, nfile + 1):
            # print(f"\rProcessing step {i}: S{comp}_0_{i * wstep * tskip:07d}", end="\r", flush=True)
            with open(f'{model}/S{comp}_0_{i * wstep * tskip:07d}', 'rb') as fid:
                for j in range(wstep):
                    fid.seek(skips[j], 0)
                    data[comp] += struct.unpack('f', fid.read(4))     

        if np.isnan(data[comp]).any():
            print(f"\nNAN in file {model}/S{comp}_0_{(i - 1) * wstep * tskip}\n")
            return None
        #data[comp] = filt_B(np.array(data[comp], dtype='float32'), 1 / dt_syn, causal=False)
        data[comp] = np.array(data[comp])
        print(f'{model}/S{comp}_0_{i * wstep * tskip:07d}')
    data['X'] = -data['X']
    data['dt'] = dt_syn
    return data


def read_vel(models, syn_sites):
    vel_syn = collections.defaultdict(dict)
    for model in models:
        if Path(model).exists():
            try:
                with open(Path(model, 'vel_sites.pickle'), 'rb') as fid:
                    vel_syn[model] = pickle.load(fid)
            except:
                for isite in range(len(syn_sites)):
                    site_name = syn_sites[isite][0]
                    print(f'Gathering {len(_models)} ground motions at site {site_name}')
                    ix, iy = syn_sites[isite][1:]
                    for model in _models:
                        if model not in vel_syn.keys():
                            vel_syn[model] = dict()
                        vel_syn[model][site_name] = read_syn(ix, iy, mx, my, model=model) 
            for k in vel_syn[model].keys():  # Each site
                vel_syn[model][k] = rotate(vel_syn[model][k], -39.9)
                vel_syn[model][k] = prepare_bbpvel(vel_syn[model][k], tmax)

        else:
            print("Not a simulated model: ", model)
            with open(f'results/vel_{model}.pickle', 'rb') as fid:
                vel_syn[model] = pickle.load(fid)


def pick_vel(mx=0, my=0, models=[], syn_sites={}):
    '''Pick velocities from computed outptu
    
    When running for the first time, models should be the complete tested models available.
    In the following, this function can amend additional models, and save it to the existing data     file.

    '''
    try: 
        with open('results/vel_syn.pickle', 'rb') as fid:
            vel_syn = pickle.load(fid)

    except Exception as e:
        print("Error: ", e, "\nRebuild vel database")
        vel_syn = collections.defaultdict(dict)
    
    if not models or all(x in vel_syn.keys() for x in models):
        print("Models queried!")
        return {k:vel_syn[k] for k in models} if models else vel_syn
    init_len = len(vel_syn)
    _models = models.copy()
    for key in vel_syn.keys():
        if key in _models:
            _models.remove(key)
    _vel_syn = read_vel(_models, syn_sites)
    vel_syn = {**vel_syn, **_vel_syn}
            
    if len(vel_syn) > init_len:
        with open('results/vel_syn.pickle', 'wb') as fid:
            pickle.dump(vel_syn, fid, protocol=pickle.HIGHEST_PROTOCOL)
            
    return {k: vel_syn[k] for k in models}


def pick_psa(mx=0, my=0, models=[], osc_freqs=np.logspace(-1, 1, 91), osc_damping=0.05, syn_sites={}):
    '''Import psa response if available
    If psa pickle files don't exist, create them.
    If some other models are queried, compute them and append to psa files.
    '''

    try: 
        with open('results/psa_syn.pickle', 'rb') as fid:
            psa_syn = pickle.load(fid)
        with open('results/psax_syn.pickle', 'rb') as fid:
            psax_syn = pickle.load(fid)
        with open('results/psay_syn.pickle', 'rb') as fid:
            psay_syn = pickle.load(fid)
    except Exception as e:
        print("Error reading psa_syn: ", e, "\nNew")
        psa_syn = collections.defaultdict(dict)
        psax_syn = collections.defaultdict(dict)
        psay_syn = collections.defaultdict(dict)
    
    if all(x in psa_syn.keys() for x in models):
        print("Models queried!")
        return {k:psa_syn[k] for k in models} if models else psa_syn
    
    # Remove existing models except extra model queried
    init_len = len(psa_syn)
    _models = models.copy()   
    for key in psa_syn.keys():
        if key in _models:
            _models.remove(key)
    vel_syn = pick_vel(mx, my, models.copy(), syn_sites)
    print(_models) 
    for isite in range(len(syn_sites)):
        site_name = syn_sites[isite][0]
        print(f"\nGathering psa_syn for {site_name}")

        for model in _models:
            if model not in psa_syn.keys():
                psa_syn[model] = dict()
                psax_syn[model] = dict()
                psay_syn[model] = dict()

            print(f"\r: {model}", end='\r', flush=True)
            accx, accy = (np.diff(x, prepend=0) / vel_syn[model][site_name]['dt'] \
                          for x in [vel_syn[model][site_name]['X'], vel_syn[model][site_name]['Y']])
            psa_syn[model][site_name] = my_pyrotd.my_calc_rotated_spec_accels(
                vel_syn[model][site_name]['dt'], accx, accy,
                osc_freqs, osc_damping, percentiles=[50])
            psax_syn[model][site_name] = my_pyrotd.my_calc_spec_accels(
                vel_syn[model][site_name]['dt'], accx,
                osc_freqs, osc_damping)
            psay_syn[model][site_name] = my_pyrotd.my_calc_spec_accels(
                vel_syn[model][site_name]['dt'], accy,
                osc_freqs, osc_damping)
    
    # Rewrite if more models queried than existing
    if len(psa_syn) > init_len:
        with open('results/psa_syn.pickle', 'wb') as fid, \
             open('results/psax_syn.pickle', 'wb') as fidx, \
             open('results/psay_syn.pickle', 'wb') as fidy:
            print("Appending new models for psa")
            pickle.dump(psa_syn, fid, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(psax_syn, fidx, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(psay_syn, fidy, protocol=pickle.HIGHEST_PROTOCOL)
            
    return {k:psa_syn[k] for k in models}


def plot_snapshot(it, mx, my, model="", case="", draw=False, backend="inline"):   
    '''Pot surface wave field
    '''
    vx, dt = read_snapshot(it, mx, my, model=model, case=case)
    fig, ax = plt.subplots()
    im = ax.imshow(vx,cmap='bwr')
    cb = plt.colorbar(im)
    cb.ax.set_ylabel('V (m/s)')
    ax.set(xlabel='X', ylabel='Y', title=f'T = {it * dt: .2f}s')
    fig.canvas.draw()
    if backend != "inline":
        return fig
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if not draw:
        plt.close(fig)
    return image


def plot_validation(site_name, models, vels=None, psas=None, lowcut=0.15, highcut=5, metrics=['vel'], comps='XYZ', syn_sites={}, backend="inline", ncol=2, labels=None, plot_rec=True, causal=False):
    '''Plot comparison among multiple models
    '''
    vel_syn = pick_vel()
    print(models)
    if not vel_syn['rec'][site_name]:
        return None
    dt_rec = vel_syn['rec'][site_name]['dt']   
    len_rec = len(vel_syn['rec'][site_name]['X'])
    t_rec = np.arange(len_rec) * dt_rec # - vel_syn['rec'][site_name]['shift']
    if not vels:
        vels = [vel_syn[model][site_name] for model in models]
    image =[]
    if 'vel' in metrics or 'acc' in metrics:
        fig, ax = plt.subplots(len(comps), 1, dpi=600, squeeze=False)
        ax = ax.squeeze(axis=-1)
        fig.tight_layout()
        fig.suptitle(f'{site_name}', y=1.08)
        for i in range(len(comps)):   
            if plot_rec:
                vel = vel_syn['rec'][site_name][comps[i]] * 100
                if 'acc' in metrics:
                    vel = np.gradient(vel, dt_rec)  # Take accelerations
                if highcut:
                    vel = filt(vel, 1 / dt_rec, lowcut=lowcut, highcut=highcut, causal=causal)
                ax[i].plot(t_rec, vel, label='Data, ' + f'Max = {np.max(vel):.4f}' if not labels else "Data")

            for j, model in enumerate(models):
                vel = vels[j][comps[i]] * 100
                dt_syn = vels[j]['dt']
                if 'acc' in metrics:
                    vel = np.gradient(vel, dt_syn)  # Take accelerations
                t_syn = np.arange(len(vel)) * dt_syn 
                if highcut:
                    vel = filt(vel, 1 / dt_syn, lowcut=lowcut, highcut=highcut, causal=causal)
                label = labels[j] if labels else model + f", Max = {np.max(vel):.4f}"
                ax[i].plot(t_syn, vel, lw=0.8, label=label)

            ax[i].set_ylabel(f'V{comps[i]} (cm/s)' if 'vel' in metrics else f'Acc{comps[i]} (cm/s/s)')
            ax[i].set_xlim(0, t_syn[-1])
            ax[i].set_xlabel('Time (s)')
            ax[i].yaxis.set_minor_locator(AutoMinorLocator(4))
            ax[i].grid(which='minor', alpha=0.3)
        ax[0].legend(*ax[-1].get_legend_handles_labels(),
                bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                ncol=ncol, mode="expand", borderaxespad=0)
        image += [fig]

    if 'psa' in metrics:
        if not psas:
            psas = [psa_syn[model][site_name] for model in models]
        psa_syn = pick_psa()
        fig2, ax = plt.subplots(dpi=600)
        fig2.tight_layout()
        fig2.suptitle(f'{site_name}', y=1.05) 
        if plot_rec:
            psa = psa_syn['rec'][site_name]
            ax.plot(psa.osc_freq, psa.spec_accel, label='rec')
        for psa in psas:
            ax.plot(psa.osc_freq, psa.spec_accel, label=model)
        ax.set(ylabel=f'$SA (m/s^2)$', xscale='log', yscale='log')
        ax.xaxis.grid(True, which='both')
        ax.yaxis.grid(True, which='both')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                 ncol=2, mode="expand", borderaxespad=0)
        image += [fig2]                     
    return image


def plot_rotd50_bias(model):
    psa = pickle.load(open('results/psa_syn.pickle', 'rb'))
    psax = pickle.load(open('results/psax_syn.pickle', 'rb'))
    psay = pickle.load(open('results/psay_syn.pickle', 'rb'))
    sites = list(psa['rec'].keys())
    periods = [1 / f for f in psa['rec'][sites[0]].osc_freq]
    def comp_bias(psa):
        m = np.zeros((len(periods),), dtype='float32')
        std = np.zeros((len(periods),), dtype='float32')
        for i in range(len(periods)):
            ratio = [np.log(psa['rec'][s].spec_accel[i] / psa[model][s].spec_accel[i]) for s in sites]
            m[i] = np.mean(ratio)
            std[i] = np.std(ratio)
        return m, std
    
    titles = ['RotD50', 'PSA East', 'PSA North']
    fig, ax = plt.subplots(3, 1, dpi=400)
    fig.subplots_adjust(hspace=0.9)
    for i, p in enumerate([psa, psax, psay]):
        m, std = comp_bias(p)
        ax[i].plot(periods, m)
        ax[i].fill_between(periods, m - std, m + std, color='gray', alpha=0.3)
        ax[i].set(xscale='log', ylabel='ln (data/syn)', xlabel='Period (sec)', title=titles[i])
        ax[i].get_xaxis().set_major_formatter(ScalarFormatter())

    return
    

def plot_syn_freqs(site_name, models, freqs=[[0.15, 1],[0.15, 5]], lowcut=0.15, comp='Z', plot_rec=False, save=False, sfile=""):
    '''
    Plot comparison between synthetics and their psa if required, similar to plot_validation
    Input
    -----
    site_name : string
    models : list
             At most two models accepted
    freqs : list of list
            Frequency bands to look at
    comp : {'X', 'Y', 'Z'}
        The component of velocities
    plot_rec : boolean
        Plot recordings if True
    save : boolean
    sfile : string
        The name of file to save when "save" is True
    '''
    vel_syn = pick_vel()
    psa_syn = pick_psa()
    nm = len(models) + (plot_rec == True)
    nf = len(freqs)
    vmax = 0
    for model in models:
        vmax = max(vmax, np.max(vel_syn[model][site_name][comp])) * 1.5
    if plot_rec:
        vmax = max(vmax, np.max(vel_syn['rec'][site_name][comp]))
    fig, ax = plt.subplots(2, 1, dpi=400)
    ax2 = ax[-1].twinx()
    for i, model in enumerate(models):
        for j, f in enumerate(freqs):
            if type(f) != list or type(f) != tuple:
                f = [lowcut, f]
            vel = vel_syn[model][site_name]
            fvel = filt(vel[comp], 1 / vel['dt'], lowcut=f[0], highcut=f[1], causal=False)
            ax[0].plot(np.arange(len(vel[comp])) * vel['dt'], fvel + vmax * (j * nm + i),
                    color=f'C{i+1}', lw=1.2, label=f'{model}' if j == 0 else None)
            valign = 'top'
            color = ax[0].get_lines()[-1].get_c()
            ax[0].annotate(f'{np.max(np.abs(fvel)):.4f}', (0, fvel[0] + vmax * (j * nm + i)), xytext=(2, 10),
                textcoords="offset points", ha='left', va=valign, color=color)
            if i == 0:
                ax[0].annotate(f'{f[0]:.2f}-{f[1]:.1f}Hz', (25, fvel[0] + vmax * (j * nm + i)), xytext=(0, 10),  textcoords="offset points", ha='left', va=valign, color='k')

        # Plot psa time histories
        psa = psa_syn[model][site_name]
        ax[-1].plot(psa.osc_freq, psa.spec_accel, label=f'{model}')
        # if plot_rec, plot psa ratio against data, otherwise between models
        if plot_rec:
            color = ax[-1].get_lines()[-1].get_c()
            ax2.plot(psa.osc_freq, np.divide(psa_syn[model][site_name].spec_accel,
                                             psa_syn['rec'][site_name].spec_accel),
                     color=color, ls=':', lw=0.8, label=f'{model} / data')
        elif len(models) > 1:
            ax2.plot(psa.osc_freq, np.divide(psa_syn[models[0]][site_name].spec_accel,
                                             psa_syn[models[1]][site_name].spec_accel),
                     'k:', lw=0.8, label=f'{models[0]} / {models[1]}')

    # Plot data time histories if plot_rec == True
    if plot_rec:
        for j, f in enumerate(freqs):
            if type(f) != list:
                f = [lowcut, f]
            vel = vel_syn['rec'][site_name]
            fvel = filt(vel[comp], 1 / vel['dt'], lowcut=f[0], highcut=f[1], causal=False)
            ax[0].plot(np.arange(len(vel[comp])) * vel['dt'], fvel + vmax * (j * nm+ nm - 1), lw=1.2, color=f'C0', label=f'data' if j == 0 else None)
            color = ax[0].get_lines()[-1].get_c()
            valign = 'top'
            ax[0].annotate(f'{np.max(np.abs(fvel)):.4f}', (0, fvel[0] + vmax * (j * nm + nm - 1)), xytext=(2, 10),
                             textcoords="offset points", ha='left', va=valign, color=color)
        psa = psa_syn['rec'][site_name]
        ax[-1].plot(psa.osc_freq, psa.spec_accel, 'k', label=f'data')

    # Adjust figure aesthetics
    ax[0].set(xlabel='Time (s)', ylabel=f'V{comp} (m/s)', xlim=[0, 25], yticklabels=[])
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                 ncol=2, mode="expand", borderaxespad=0)
    ax[-1].set(xlabel='Frequency (Hz)', ylabel=f'$SA (m/s^2)$', xlim=[0, 10])
    ax[-1].xaxis.grid(True, which='both')
    ax[-1].yaxis.grid(True, which='both')
    ax2.set_yscale('log')
    ax2.set_ylabel('Ratio')
    formatter = mpl.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(3, 2.0))
    ax2.yaxis.set_minor_formatter(formatter)
    ax2.yaxis.grid(True, which='major', color='gray', ls='--')
    ax[-1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncol=2, mode="expand", borderaxespad=0)
    fig.tight_layout()
    if save:
        saveto = sfile if sfile else f'syn_freqs_{models[0]}_{freqs[0]:.1f}hz'
        fig.savefig(f"results/{saveto}.svg", dpi=400, bbox_inches='tight', pad_inches=0.05)
    return fig


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = mpl.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
                                 align="center", pad=ppad, sep=sep) 
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)
        
def plot_trace_comp(models, f, isite, site_name, comps='XYZ', gof='tf_misfit', labels=None, ratios=[1], unit='cm', cav=True, vel_syn=None, plot_rec=False, save=False, sfile=""):
    if not vel_syn:
        vel_syn = pick_vel()
    if plot_rec:
        models = models + ['rec']
        labels = labels if not labels else labels + ['Data']
    ratios = force_iterable(ratios)
    if len(ratios) != len(comps):
        ratios += ratios[-1] * (len(comps) - len(ratios))
    r_unit = 100 if unit == 'cm' else 1
    f_low = 0.15  # lowcut frequency
    orient = {'X': 'E-W', 'Y': 'N-S', 'Z': 'UP'}
    tf_misfit = pickle.load(open('results/tf_misfit.pickle', 'rb'))
    gof = pickle.load(open('results/gof.pickle', 'rb'))
    #met = pickle.load(open('results/metrics.pickle', 'rb'))
    f = f if isinstance(f, collections.Iterable) else (f_low, f)
    fig, ax = plt.subplots(len(comps), 2, dpi=500, gridspec_kw={'width_ratios': [3, 1]})
    colors = list(mcolors.TABLEAU_COLORS.keys())
    fig.tight_layout()
    fig.suptitle(f'Site {isite}  {gof[(0.15, 2.5)][models[0]][site_name]["rhypo"]:.2f} km', y=1.1)
    for j, model in enumerate(models):
        dt = vel_syn[model][site_name]['dt']  
        nt = len(vel_syn[model][site_name]['X'])
        vel = filt(vel_syn[model][site_name], 1 / dt, f[0], f[1])
        cumvel = comp_cum_energy(vel_syn[model][site_name], lowcut=f[0], highcut=f[1], cav=cav)
        for i, comp in enumerate(comps): 
            ix = ax[i] if len(comps) > 1 else ax
            dy = np.mean([np.max(vel_syn[m][site_name][comp] * r_unit) for m in models]) * ratios[i]
            ix[0].plot(np.arange(nt) * dt, vel[comp] * r_unit - j * dy, c=colors[j])
            ix[0].annotate(f'{r_unit * np.max(np.abs(vel[comp])):.2f} cm/s', (nt * dt, vel[comp][0] - j * dy), xytext=(0, 6), textcoords="offset points", ha='right', va='center', fontsize=10) 
            ix[1].plot(np.arange(nt) * dt, cumvel[comp] * r_unit, c=colors[j], label=model if not labels else labels[j])
       
    for i, comp in enumerate(comps):
        ix = ax[i] if len(comps) > 1 else ax
        ix[0].tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
        ix[0].set_ylabel(orient[comp], labelpad=-10)
        ix[0].set_frame_on(False)
        ix[1].tick_params(bottom=True, left=True)
    ix = ax[0] if len(comps) > 1 else ax
    ix[0].set_title('Seismograms (cm/s)', pad=10)
    ix[1].set_title('CAV (cm)' if cav else r'Cum. Energy (cm$^2$/s)', pad=10)
    ix = ax[-1] if len(comps) > 1 else ax
    ix[0].set_xlabel('Time (s)')
    ix[1].set_xlabel('Time (s)')
    if labels != False:
        ix[1].legend(loc=2, fontsize=5, frameon=True, fancybox=True)
    scale = 5
    ob = AnchoredHScaleBar(size=15, label=f"{scale} sec", loc=3, frameon=False, pad=0.6, sep=5, linekw=dict(color="k", linewidth=0.8), bbox_to_anchor=(0.02, -0.1 * len(comps)), bbox_transform=ix[0].transAxes) 
    ix[0].add_artist(ob)

    if save:
        saveto = sfile if sfile else f'trace_comp_{site_name}'
        fig.savefig(f"results/{saveto}.png", dpi=500, bbox_inches='tight', pad_inches=0.05)

    return fig, ax
            

def comp_cum_energy(vel, dt=0, lowcut=0.15, highcut=5, cav=False, average=False):
    '''Compute cumulative energy from velociteis
    Input:
        vel (dict or list) : velocities
        dt (float)         : time step
        lowcut (float)     : low bound frequency for filtering
        highcut (float)    : high bound frequency for filtering
        cav (boolean)      : True-cumulative absolute vel; False-vel**2
    '''
    if issubclass(type(vel), dict):
        dt = vel['dt']
        keys = list(vel.keys())
        keys.remove('dt')
        if 'shift' in keys:
            keys.remove('shift')
        cumvel = collections.defaultdict(lambda : np.zeros((len(vel[keys[0]]),), dtype='float32'))
        for k in keys:
            v = filt(vel[k], 1 / dt, lowcut=lowcut, highcut=highcut)
            cumvel[k] = comp_cum_energy(v, dt=dt, cav=cav)
        if average:
            cumvel = [[cumvel[k]] for k in keys]
            return np.mean(cumvel, axis=0)
        return cumvel
    vel = np.array(vel)
    if cav:
        return np.cumsum(np.abs(vel)) * dt
    return np.cumsum(vel ** 2) * dt


def plot_cum_energy(models, nrow=4, ncol=3, lowcut=0.15, highcut=5, dh=0.008, 
        syn_sites={}, seed=None, nsites=[], vs30=None, vs=None, cav=False,
        plot_rec=True, save=False, sfile=""):
    '''Plot cumulative energy time histories
    '''
    vel_syn = pick_vel()
    fig, ax= plt.subplots(nrow, ncol, dpi=200)
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.1)
    if not nsites:
        if seed is not None:
            np.random.seed(seed)
        nsites = np.random.rand(len(syn_sites)).argsort()
    # print(nsites)
    for i in range(len(syn_sites)):
        if i > nrow * ncol - 1:
            break
        row, col = i // ncol, i % ncol
        ax[row, col].axis('off')
        ax[row, col].set_frame_on(True)
        if i >= len(nsites):
            if col == ncol - 1 and row == nrow - 2:
                ax[row, col].legend(*ax[0][0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0, 1.2), bbox_transform=ax[row, col].transAxes)
            ax[row, col].set_frame_on(False)
            continue
        
        j =  nsites[i]
        site_name = syn_sites[j][0]
        for model in models:
            cumvel = comp_cum_energy(vel_syn[model][site_name], lowcut=lowcut, highcut=highcut, cav=cav, average=True)
            ax[row, col].plot(vel_syn[model][site_name]['dt'] * np.arange(len(vel_syn[model][site_name]['X'])), cumvel, label=model)
        cumvel = comp_cum_energy(vel_syn['rec'][site_name], lowcut=lowcut, highcut=highcut, cav=cav, average=True)
        ax[row, col].plot(vel_syn['rec'][site_name]['dt'] * np.arange(len(vel_syn['rec'][site_name]['X'])), cumvel, '--', label='rec')
        if 'vs30' in locals():
            title = 'Vs30'
            vs = vs30
        else:
            title = 'Vs'
        ax[row, col].set_title(f'Site {j}, {title}={vs[syn_sites[j][2], syn_sites[j][1]]:.1f} m/s')
    if save:
        saveto = sfile if sfile else f'cum_{"vel" if cav else "ener"}_{models[0]}_{lowcut:.1f}_{highcut:.1f}hz'
        fig.savefig(f"results/{saveto}.svg", dpi=400, bbox_inches='tight', pad_inches=0.05)
    return fig
                            

def plot_interp_map(model, mx, my, f=(0.15, 2.5), metric='pgv', syn_sites=None, topography=None, step=5, sx=1878, sy=2735):
    x, y = np.meshgrid(np.arange(0, mx, step), np.arange(0, my, step))
    met = pickle.load(open('results/metrics.pickle', 'rb'))
    xi = np.zeros(len(syn_sites))
    yi = np.zeros(len(syn_sites))
    zi = np.zeros(len(syn_sites))
    fig, ax = plt.subplots(dpi=300)
    for i, s in enumerate(syn_sites):
        xi[i] = s[2] 
        yi[i] = mx - s[1] 
        zi[i] = met[f]['rec'][s[0]][metric] / met[f][model][s[0]][metric]
    print(zi)
    z = griddata((xi, yi), zi, (x, y), method='cubic')
    image = ax.contourf(x, y, z, cmap='hot_r')
    ax.scatter(sy, sx, 300, color='k', marker='*')
    ax.contour(x, y, np.rot90(topography[::step, ::step]), 5, cmap='cividis', linewidths=0.5)
    plt.colorbar(image, ax=ax, orientation='vertical')
    im = ax.scatter(xi, yi, s=200, marker='^', c=zi, cmap='binary')
    plt.colorbar(im, ax=ax, orientation='horizontal')
    return


def plot_metric_map(mx, my, models, f=(0.15, 2.5), lowcut=0.15, metric='pgv', vmax = 1e6, topography=None, nd=250, step=5, sx=1878, sy=2735, syn_sites={}, save=False, sfile=""):
    sx, sy = (sx - nd) // step, (sy - nd) // step
    val = collections.defaultdict()
    fig, ax = plt.subplots((len(models) - 1) // 2 + 1, 2, dpi=200)
    plt.suptitle(f'{metric} at {f} Hz', y=1.05)
    fig.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.85, wspace=-0.1, hspace=0.6)
    f = f if isinstance(f, collections.Iterable) else (lowcut, f)
    for i, model in enumerate(models):
        val[model] = np.fromfile(f'{model}/{metric}_{f[0]:05.2f}_{f[1]:05.2f}Hz.bin', dtype='f').reshape(my, mx)[nd : -nd : step, nd : -nd : step]
        # TODO
        # Compute these metrics at each site
        vmax = min(vmax, 0.85 *  np.max(val[models[0]]))
        vmin = np.min(val[models[0]])
        im = ax[i // 2, i % 2].imshow(val[model][nd:-nd:step, nd:-nd:step].T, cmap='bwr', vmin=vmin, vmax=vmax)
        ax[i // 2, i % 2].contour(topography[nd:-nd:step, nd:-nd:step].T, 5, cmap='cividis', linewidths=0.5)
        ax[i // 2, i % 2].scatter(sy, my - sx, 50, color='g', marker='*')
        cbar = plt.colorbar(im, ax=ax[i // 2, i % 2], ticks=np.linspace(vmin, vmax, 5))
        ax[i // 2, i % 2].set_title(model, y=1.05)
    if i % 2 == 0:
        fig.delaxes(ax[-1, -1])
    if save:
        saveto = sfile if sfile else f'{metric}_{models[0]}_{f:.1f}hz'
        fig.savefig(f"results/{saveto}.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
    return fig


def plot_diff_map(mx, my, models, f=(2.5, 5), metric='pgv', vmax=2, titles=[], lowcut=0.15, topography=None, nd=250, step=5, sx=1878, sy=2735, dh=0.008, save=False, sfile=""):
    '''plot ratio map and histogram
    Input
    -----
    models : list
        Currently only accept two models
    f : float or int
        Frequency to plot with
    metric : {'pgv', 'dur', 'arias', 'gmrotD50'}
    nd : int
        Absorbing layers
    '''
    # sx, sy = (sx - nd) // step, (sy - nd) // step
    sx, sy = (sx - nd) * dh, (sy - nd) * dh
    ylabel = {'gof': r'GOF', 'pgv': r'PGV (m/s)', 'pga': r'PGA (m/s$^2$)',
            'ener': r'ENER (m$^2$/s)', 'arias': r'Arias (m/s)', 'dur': 'DUR (s)'}
    val = collections.defaultdict()
    f = f if isinstance(f, collections.Iterable) else (lowcut, f)
    for model in models:
        val[model] = np.fromfile(f'{model}/{metric}_{f[0]:05.2f}_{f[1]:05.2f}Hz.bin', dtype='f').reshape(my, mx)[nd : -nd : step, nd : -nd : step]

    vmax = min(vmax, 0.8 * np.max(val[models[0]]), 0.8 * np.max(val[models[1]]))
    vmin = max(np.min(val[models[0]]), np.min(val[models[1]]))

    fig, ax = plt.subplots((len(models) - 1) // 2 + 1, 2, dpi=400)
    # plt.suptitle(f'Comparison of {metric} at {f[0]}-{f[1]} Hz', y=1.05)
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9, right=0.8, wspace=0.25)
    for i, model in enumerate(models):
        im = ax[i].imshow(val[model].T, cmap='bwr', vmin=vmin, vmax=vmax, extent=[0, (mx - 2 * nd) * dh, 0, (my - 2 * nd) * dh])
        CS = ax[i].contour(np.arange(0, mx - 2 * nd, step) * dh, np.arange(my - 2 * nd, 0, -step) * dh, topography[nd:-nd:step, nd:-nd:step].T, [0, 50, 100, 150, 250, 400], cmap='cividis', linewidths=0.8)
        ax[i].clabel(CS, fmt='%d', inline=True, fontsize=8)
        ax[i].scatter(sy, sx, 200, color='g', marker='*')
#        ax[i].set_title(f'{titles[i] if titles else models[i]}\n' + fr'Max = {np.max(val[model]):.2f} m/s$^2$; Min = {np.min(val[model]):.2f} m/s$^2$', y=1.04)
        ax[i].set_title(f'{titles[i] if titles else models[i]}')
        ax[i].set(xlabel = 'X (km)', ylabel = 'Y (km)')
        ax[i].text(0.4, 21, f'Max = {np.max(val[model]):.2f} m/s$^2$', color='w', fontweight='bold')
        ax[i].text(0.4, 19, f'Min = {np.min(val[model]):.2f} m/s$^2$', color='w', fontweight='bold')
    cax = fig.add_axes([0.85, 0.30, 0.02, 0.4])
    ticks = np.linspace(vmin, vmax, 5)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=ticks)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in ticks])
    cbar.ax.set_ylabel(ylabel[metric])
    if save:
        saveto = sfile if sfile else f'comp_{metric}_{models[0]}_{f[0]:.2f}_{f[1]:.1f}hz'
        fig.savefig(f"results/{saveto}.svg", dpi=500, bbox_inches='tight', pad_inches=0.05)
                            
    return fig


def plot_diff_hist(mx, my, models, freqs=[1, 5], metric='pgv', vmax=2, lowcut=0.15, topography=None, nd=250, step=5, sx=1878, sy=2735, save=False, sfile=""):
    '''plot ratio map and histogram
    Input
    -----
    mx, my : size of domain
    models : list
        Currently "noqf_orig" is necessary
    freqs : list of float or int
        Frequencies to compare with
    metric : {'pgv', 'dur', 'arias', 'gmrotD50'}
    nd : int
        Absorbing layers
    '''
    if "noqf_orig" not in models:
        print('Model "noqf_orig" is needed')
        return None
    if len(freqs) != 2:
        print('Accept TWO frequencies only!')
        return None

    sx, sy = (sx - nd) // step, (sy - nd) // step
    val = collections.defaultdict(dict)
    val_dif = collections.defaultdict()
    freqs = [(lowcut, f) if type(f) != tuple and type(f) != list else tuple(f) for f in freqs] 
    for f in freqs:
        for model in models:
            val[f][model] = np.fromfile(f'{model}/{metric}_{f[0]:05.2f}_{f[1]:05.2f}Hz.bin', dtype='f').reshape(my, mx)
        val_dif[f] = np.divide(sum(val[f][model] for model in models if model != "noqf_orig") / (len(models) - 1), val[f]['noqf_orig'], out=np.zeros_like(val[f][model]), where=val[f]['noqf_orig'] != 0)
        val_dif[f] = val_dif[f][nd:-nd:step, nd:-nd:step]
        val_dif[f][np.isnan(val_dif[f])] = 0
        val_dif[f][np.isinf(val_dif[f])] = 0
        print(f"At {f[0]:.2f}-{f[1]:.1f} Hz: Max = {np.max(val_dif[f]):.2f}, Min = {np.min(val_dif[f]):.2f}")

    vmax = min(vmax, 0.8 * np.max(val_dif[freqs[1]]))
    vmin = np.min(val_dif[freqs[0]])
    print(vmin, vmax)
    fig, ax = plt.subplots(2, 2, dpi=300)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, f in enumerate(freqs):
        im = ax[0, i].imshow(val_dif[f].T, cmap='bwr', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax[0, i], orientation='vertical', ticks=np.linspace(vmin, vmax, 5))
        cbar.ax.set_ylabel('Ratio')
        CS = ax[0, i].contour(topography[nd:-nd:step, nd:-nd:step].T, [0, 50, 100, 150, 250, 400], cmap='cividis', linewidths=0.8)
        ax[0, i].clabel(CS, fmt='%d', inline=True, fontsize=8)
        ax[0, i].scatter(sy, my //step - sx, 150, color='g', marker='*')
        ax[0, i].set_title(f'Median ratio = {100 * np.median(val_dif[f]):.3f}%, f = {f[0]:.1f}-{f[1]:.1f}Hz', y=1.05)

    val_dif_tmp = np.divide(val_dif[freqs[1]], val_dif[freqs[0]], out=np.zeros_like(val_dif[freqs[0]]), where=val_dif[freqs[0]]!=0)
    im = ax[1, 0].imshow(val_dif_tmp.T, cmap='bwr', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax[1, 0], orientation='vertical', ticks=np.linspace(vmin, vmax, 5))
    ax[1, 0].set_title(f'({freqs[1][0]:.1f}-{freqs[1][1]:.1f})Hz / ({freqs[0][0]:.1f}-{freqs[0][1]:.1f})Hz', y=1.05)
    CS = ax[1, 0].contour(topography[nd:-nd:step, nd:-nd:step].T, [0, 50, 100, 150, 250, 400], cmap='cividis', linewidths=0.8)
    ax[1, 0].clabel(CS, fmt='%d', inline=True, fontsize=8)
    ax[1, 0].scatter(sy, my // step - sx, 150, color='g', marker='*')

    bins = np.linspace(vmin, vmax, 12)
    ax[1, 1].hist([np.ravel(val_dif[f]) for f in freqs], bins, label=[f'{f[0]:.2f}-{f[1]:.1f}Hz' for f in freqs], density=True)
    ax[1, 1].set_xlabel('Ratio')
    ax[1, 1].set_ylabel('Count density')
    ax[1, 1].yaxis.tick_right()
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].legend(loc=1)
    if save:
        fstring = "_".join(f'{f[0]:.2f}_{f[1]:.2f} for f in freqs')
        saveto = sfile if sfile else f'diff_hist_{models[0]}_{fstring}hz'
        fig.savefig(f"results/{saveto}.svg", dpi=400, bbox_inches='tight', pad_inches=0.05)

    fig2, ax = plt.subplots(1, 2, dpi=200)
    im1 = ax[0].imshow(topography[nd : -nd : step, nd : -nd : step].T, cmap='terrain', norm=PowerNorm(gamma=0.6))
    ax[0].set_title('Topography', y=1.05)
    cbar1 = plt.colorbar(im1, ax=ax[0], orientation='horizontal')
    cbar1.ax.set_xlabel('Elevation (m)')
    cbar1.set_ticks([0, 50, 100, 200, 400])
    # There is a bug with matplotlib using locator_params on colorbar.ax
    # https://github.com/matplotlib/matplotlib/issues/11937
    # cbar1.set_ticks(np.linspace(np.min(topography), np.max(topography), 6))
    im2 = ax[1].imshow(val_dif[freqs[1]].T - val_dif[freqs[0]].T, cmap='bwr', vmax=2)
    ax[1].set_title(f'Normalized ratio ({freqs[1]}Hz - {freqs[0]}Hz)', y=1.05)
    cbar2 = plt.colorbar(im2, ax=ax[1], orientation='horizontal')
    if save:
        fig2.savefig(f"results/{saveto}_2.svg", dpi=400, bbox_inches='tight', pad_inches=0.05)

    return [fig, fig2], val_dif
                            

def prepare_tf_misfit(model, vel_syn=None, fmin=0.15, fmax=5, exec_path='results', IS_S2_REFERENCE='true', LOCAL_NORM='false'):
    curdir = os.getcwd()
    os.chdir(exec_path) 
    if not vel_syn:
        with open('vel_syn.pickle', 'rb') as fid:
            vel_syn = pickle.load(fid)
    
    try:
        tf_misfit = dict()
        for j, k in enumerate(vel_syn['rec'].keys()):
            with open('HF_TF-MISFIT_GOF', 'w') as fid:
                fid.write(f'{len(vel_syn[model][k]["X"])}\n{vel_syn[model][k]["dt"]}\n{fmin} {fmax}\n'
                          f'syn_{k}.dat\nrec_{k}.dat\n3\n.{IS_S2_REFERENCE}.\n.{LOCAL_NORM}.') 
            command = ['./tf_misfits_gof', 'HF_TF-MISFIT_GOF']
            p = call_sub(command)
            tf_misfit[k] = np.genfromtxt('MISFIT-GOF.DAT')
            print(f"\rDone {j} / {len(vel_syn['rec'].keys())}", end='\r', flush=True)
            os.remove('MISFIT-GOF.DAT')
    except OSError as e:
        print("Error", e)
    finally:
        print("\n")
        os.chdir(curdir)
    return tf_misfit


def plot_tf_misfit(f=(0.15, 5), metric='EG', ref='pgv', radius=100000):
    '''
    Input:
        f (tuple) : lowcut and highcut frequency
        metric (string) : 'EM' (envelope misfit)
                          'PM' (phase misfit)
                          'EG' (envelope gof)
                          'PG' (phase gof)
    '''
    labels = {'EM': 'Env-Misfit',
              'PM': 'Phase-Misfit',
              'EG': 'Env-GoF',
              'PG': 'Phase-GoF'}
    xi = 3 if 'G' in metric else 0  # 3-6 is GoF, 0-3 is Misfit
    yi = 0 if 'E' in metric else 1  # Col 0 is Envelope; Col 1 is Phase
    with open('results/tf_misfit.pickle', 'rb') as fid:
        tf_misfit = pickle.load(fid) 
    met = pickle.load(open(f'results/metrics.pickle', 'rb'))
    gof = pickle.load(open(f'results/gof.pickle', 'rb'))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    comps = 'XYZ'
    cases = ['noqf_orig', 'topo_noqf_orig', 'noqf_s05h005l100', 'topo_qf06_s05h005l100']
    fig, ax = plt.subplots(3, 1, dpi=400, sharex=True)
    plt.tight_layout()
    for i in range(3):
        for j, case in enumerate(cases):
            for k, (dhyp, seed) in enumerate(zip([1,1,2,2,0.5], [1848640878, 387100462, 372823598, 462574446, 1485839278])):

                model = f'dhyp{dhyp:.2f}_s{seed}_{case}' if dhyp != 0.5 else case
                try:
                    data = np.mean(list(tf_misfit[f][model][key][xi + i, yi] for key in tf_misfit[f][model].keys() if gof[f][model][key]['rhypo'] < radius)) 
                    if 'G' in metric:  # Goodness of Fit ~ [0-1]
                        data /= 10
                    r = np.mean(list(met[f][model][site][ref] / met[f]['rec'][site][ref] for site in met[f]['rec'].keys()))
                    marker = '^' if r >= 1 else 'v'
                    label = None if j else f'{dhyp:.1f}_{seed:5d}'
                    ax[i].scatter(j + 1, data, 60 * r ** 1.5, marker=marker, c=colors[k],
                         label=label)
                except:
                    print(model)
                    break

        ax[i].set(xticks=range(1,5), ylabel=f'{labels[metric]}-{comps[i]}')
    ax[-1].set_xticklabels(cases, rotation=0)
    ax[0].legend(bbox_to_anchor=(1, .5), loc='lower left', bbox_transform=plt.gcf().transFigure, fancybox=True)


def plot_tf_misfit_mesh(models, f=(0.15, 5), ref='pga', radius=10000):
    '''
    Input:
        f (tuple) : lowcut and highcut frequency
        metric (string) : 'EM' (envelope misfit)
                          'PM' (phase misfit)
                          'EG' (envelope gof)
                          'PG' (phase gof)
    '''
    with open('results/tf_misfit.pickle', 'rb') as fid:
        tf_misfit = pickle.load(fid) 
    met = pickle.load(open(f'results/metrics.pickle', 'rb'))
    gof = pickle.load(open(f'results/gof.pickle', 'rb'))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    comps = 'XYZ'
    fig, ax = plt.subplots(1, 3, figsize=(6, 3), dpi=400, sharey=True)
    plt.tight_layout()
    for i in range(3):
        for j, model in enumerate(models):
            try:
                EG = np.mean(list(tf_misfit[f][model][key][3 + i, 0] for key in tf_misfit[f][model].keys() if gof[f][model][key]['rhypo'] < radius)) / 10
                PG = np.mean(list(tf_misfit[f][model][key][3 + i, 1] for key in tf_misfit[f][model].keys() if gof[f][model][key]['rhypo'] < radius)) / 10
                r = np.mean(list(met[f][model][site][ref] / met[f]['rec'][site][ref] for site in met[f]['rec'].keys() if gof[f][model][site]['rhypo'] < radius))
                marker = '^' if r >= 1 else 'v'
                ax[i].scatter(EG, j, 40 * r ** 1.5, marker=marker, c=colors[0],
                        label='Env GoF' if j == 0 else None)
                ax[i].scatter(PG, j, 40 * r ** 1.5, marker=marker, c=colors[1],
                        label='Phase GoF' if j == 0 else None)
            except Exception as e:
                print(model, e)
                break

        ax[i].set(ylim=(-0.5, len(models) - 0.5), yticks=range(len(models) + 1), xlabel=f'GoF-{comps[i]}')
    ax[0].set_yticklabels(models, rotation=0)
    ax[-1].legend(bbox_to_anchor=(1., 0.5), bbox_transform=plt.gcf().transFigure, loc='lower left', frameon=True, fancybox=True)


def comp_metrics(vel_syn=None, lowcut=0.15, highcut=5, tmax=35, save=False):
    if not vel_syn:
        with open('results/vel_syn.pickle', 'rb') as fid:
            vel_syn = pickle.load(fid)
    g = 9.8
    _highcut = [highcut] if type(highcut) != list else highcut
    try:
        metrics = pickle.load(open(f'results/metrics.pickle', 'rb'))
    except:
        metrics = {}
    for hc in _highcut:
        key = (lowcut, hc)
        if key in metrics:
            continue
        metrics[key] = {}
        print(f'Processing frequency band = ({lowcut}, {hc}) Hz')
        for model in vel_syn:
            metrics[key][model] = {}
            vel = vel_syn[model]
            for site_name in vel:
                metrics[key][model][site_name] = {}
                if not hc:
                    v = vel[site_name]
                else:
                    v = filt(vel[site_name], lowcut=lowcut, highcut=hc)
                # shift = v.get('shift', 0)
                shift = 0
                try:
                    dt = v['dt']
                except:
                    print(model, site_name, vel.keys(), v.keys())
                    sys.exit(-1)
                start, end = int(shift / dt), int((shift + tmax) / dt)
                vx = v['X'][start : end]
                vy = v['Y'][start : end]
                vz = v['Z'][start : end]
                cx, cy, cz = np.cumsum(vx ** 2), np.cumsum(vy ** 2), np.cumsum(vz ** 2)

                accx = np.diff(vx, prepend=0) 
                accy = np.diff(vy, prepend=0)
                accz = np.diff(vz, prepend=0)
                try:
                    x_5, x_75 = np.argwhere(cx >= 0.05 * cx[-1])[0], np.argwhere(cx >= 0.75 * cx[-1])[0]
                except:
                    print(model, site_name, cx.shape, np.max(cx))
                    break
                y_5, y_75 = np.argwhere(cy >= 0.05 * cy[-1])[0], np.argwhere(cy >= 0.75 * cy[-1])[0]
                z_5, z_75 = np.argwhere(cz >= 0.05 * cz[-1])[0], np.argwhere(cz >= 0.75 * cz[-1])[0]
                metrics[key][model][site_name]['ener'] = np.sum(vx ** 2 + vy ** 2 + vz ** 2) * dt / 3
                metrics[key][model][site_name]['cav'] = np.sum(np.abs(accx) + np.abs(accy) + np.abs(accz)) * dt / 3
                metrics[key][model][site_name]['cad'] = np.sum(np.abs(vx) + np.abs(vy) + np.abs(vz)) * dt / 3
                metrics[key][model][site_name]['pgv'] = np.sqrt(np.max(vx ** 2 + vy ** 2 + vz ** 2))
                metrics[key][model][site_name]['dur'] = (x_75 - x_5 + y_75 - y_5 + z_75 - z_5) * dt / 3
                metrics[key][model][site_name]['pga'] = np.sqrt(np.max(accx ** 2 + accy ** 2 + accz ** 2))
                metrics[key][model][site_name]['arias'] = np.pi / 2 / g * np.sum(accx ** 2 + accy ** 2 + accz ** 2) * dt / 3

        if save:
            with open(f'results/metrics.pickle', 'wb') as fid:
                pickle.dump(metrics, fid, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics if (not save and type(highcut) != list) else None

def errorf(x, y):
    return 100 * erfc(2 * abs(x - y) / (x + y))


def comp_GOF(freqs, models, metrics=['arias', 'dur', 'ener', 'pga', 'pgv'], syn_sites={}, sx=1878,
             sy=2735, sz=708, dh=0.008, lowcut=0.15, vs=None, topography=None, save=True):

    met = pickle.load(open(f'results/metrics.pickle', 'rb'))
    gof = {}
    for f in freqs:
        if type(f) != tuple and type(f) != list:
            f = (lowcut, f)
        # In case f is list
        f = tuple(f)
        if f not in met:
            print(f"Frequency {f} band not computed!\nAborting")
            sys.exit(-1)
        met_rec = met[f]['rec']
        gof[f] = {}
        print(f)
        for model in models:
            if model == 'rec':
                continue
            met_syn = met[f][model]
            gof[f][model] = collections.defaultdict(dict)
            for site in syn_sites:
                site_name = site[0]
                gof[f][model][site_name]['rhypo'] = np.sqrt((site[2] - sx) ** 2 + (site[1] - sy) ** 2 + sz ** 2) * dh
                gof[f][model][site_name]['elev'] = topography[site[2], site[1]]
                gof[f][model][site_name]['vs'] = vs[site[2], site[1]]
                gof[f][model][site_name]['gof'] = np.mean([errorf(met_syn[site_name][metric], met_rec[site_name][metric]) for metric in metrics])
    
    met = f'_{metrics[0]}' if len(metrics) == 1 else ""
    with open(f'results/gof{met}.pickle', 'wb') as fid:
        pickle.dump(gof, fid, protocol=pickle.HIGHEST_PROTOCOL)
    return gof


def plot_gof(freqs, models, metric="gof", syn_sites={}, lowcut=0.15, window=2):
    ''' Plot Goodness-of-fit

    '''
    with open(f'results/gof.pickle', 'rb') as fid:
        gof = pickle.load(fid)
    if metric in ['EM', 'PM', 'EG', 'PG']:
        met = pickle.load(open(f'results/tf_misfit.pickle', 'rb'))
    if metric != 'gof':
        met = pickle.load(open(f'results/metrics.pickle', 'rb'))
    freqs = [freqs] if type(freqs) != list else freqs
    models = [models] if type(models) != list else models
    ylabel = {'gof': r'GOF', 'pgv': r'PGV (m/s)', 'pga': r'PGA (m/s$^2$)',
            'ener': r'ENER (m$^2$/s)', 'arias': r'Arias (m/s)', 'dur': 'DUR (s)'}

    fig, ax = plt.subplots(3, 1, figsize=(6, 8), dpi=200)
    fig.subplots_adjust(top=0.7, hspace=0.4, wspace=0.2)
    for j, f in enumerate(freqs):
        if type(f) != tuple and type(f) != list:
            f = (lowcut, f)
        # In case f is list
        f = tuple(f)
        for i, model in enumerate(models):
            x1 = [gof[f][model][site[0]]['rhypo'] for site in syn_sites]
            x2 = [gof[f][model][site[0]]['elev'] for site in syn_sites]
            x3 = [gof[f][model][site[0]]['vs'] for site in syn_sites]
            if metric == 'gof':
                y = [gof[f][model][site[0]][metric] for site in syn_sites]
            else:
                y = [met[f][model][site[0]][metric] for site in syn_sites]
            xx1, yy1 = zip(*sorted(zip(x1, y)))
            yy1_ma = [np.mean(yy1[k-window : k+window]) for k in range(window, len(yy1) - window)]
            xx2, yy2 = zip(*sorted(zip(x2, y)))
            yy2_ma = [np.mean(yy2[k-window : k+window]) for k in range(window, len(yy2) - window)]
            xx3, yy3 = zip(*sorted(zip(x3, y)))
            yy3_ma = [np.mean(yy3[k-window : k+window]) for k in range(window, len(yy3) - window)]

            ax[0].scatter(xx1, yy1, label=f'{model}, {f}Hz; med={np.median(y):.2f}', color=f'C{i + j * len(models)}')
            ax[0].plot(xx1[window : len(xx1) - window], yy1_ma, color=f'C{i + j * len(models)}')
            ax[0].set(xlabel=r'$R_{hypo} (km)$', ylabel=ylabel[metric])
            ax[0].legend(bbox_to_anchor=(0.1, 1.0, 1, 0.2), loc='lower left', ncol = 1, mode=None)
            ax[1].scatter(xx2, yy2, color=f'C{i + j * len(models)}')
            ax[1].plot(xx2[window : len(xx2) - window], yy2_ma, color=f'C{i + j * len(models)}')
            ax[1].set(xlabel=r'$Elevation (m)$', ylabel=ylabel[metric])
            # ax[1].legend(bbox_to_anchor=(-0.1, 1.0, 1, 0.2), loc='lower left', ncol = 2, mode=None)
            ax[2].scatter(xx3, yy3, color=f'C{i + j * len(models)}')
            ax[2].plot(xx3[window : len(xx3) - window], yy3_ma, color=f'C{i + j * len(models)}')
            ax[2].set(xlabel=r'$Vs30 (m/s)$', ylabel=ylabel[metric])
            # ax[2].legend(bbox_to_anchor=(-0.1, 1.0, 1, 0.2), loc='lower left', ncol = 2, mode=None)
    for ix in ax:
        ix.yaxis.grid(True, which='major', linewidth=1.0)
        ix.yaxis.grid(True, which='minor', linewidth=0.5)
        #ix.yaxis.grid(True, which='both')
        ix.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ix.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        if metric != 'gof':
            ix.set_yscale('log')
    return fig


def plot_gof_gmt(model, f, metric='tf_misfit', env=0, radius=100000, src_lon=-117.932587, src_lat=33.918633):
    '''
    Plot GOF maps using GMT for metric like "metric" or "gof" or "tf_misfit"
    Input:
        model (string) : 
        f (tuple) : (low, high)
        metric (string) : metric type to plot
        env (int) : Eg(env=0) or PG(env=1) for tf_misfit
        radius (int) : radius to rule out stations if set properly
    '''
    # env: 0 for EG, 1 for PG
    syn_sites = pickle.load(open('results/syn_sites.pickle', 'rb'))
    gof = pickle.load(open('results/gof.pickle', 'rb'))
    tf_misfit = pickle.load(open('results/tf_misfit.pickle', 'rb'))
    site_lonlat = np.genfromtxt('la_habra_small_statlist_3456.txt', delimiter=" ", dtype="S8, f, f")

    fname = f'{metric}_{model}_{f[0]}_{f[1]}hz'
    with open(f'results/{fname}.txt', 'w') as fid:
        for i in range(len(site_lonlat)):
            try:
                if metric == 'tf_misfit':
                    data = np.mean(tf_misfit[f][model][syn_sites[i][0]][3:, env])
                    data *= 10  # GOF: 0-10 to 0-100
                else:
                    data = gof[f][model]['gof'][0]
            except:
                return
            fid.write(f'{site_lonlat[i][1]:.5f} {site_lonlat[i][2]:.5f} {data:.5f}\n')

    gmt_cmd = (f'model_region="-R-118.209/-117.785/33.693/34.045"\n'
               f'gmt begin results/{fname} png\n'
               f'gmt basemap -JM15c $model_region -Baf -BSWen\n'
               f'gmt makecpt -Cgray -D -V -I -T-1000/3412/10\n'
               f'gmt grdimage @earth_relief_15s -I+d -t30\n'
               f'gmt surface results/{fname}.txt $model_region -Ginterp1.grd -I128e/128e -T0.15\n'
               f'gmt grdsample interp1.grd $model_region -Ginterp.grd -I8e/8e\n'
               f'gmt grdclip interp.grd -Gclip.grd -Sa79.9/79.9 -Sb20.0001/20.0001\n'
               f'gmt makecpt -Cpolar -T20/80/0.1\n'
               f'gmt grdimage clip.grd -I+d\n'
               f'gmt colorbar -DjMR+o-1c/0+m -I0.3 -Bxa10+lEGOF\n'
               f'gmt plot results/{fname}.txt -St0.4c -C -W1p,black -l"Station"\n'
               f'gmt plot borders.txt -W4p,lightblue -L\n'
               f'echo {src_lon} {src_lat} | gmt plot -Sa0.7c -Ggreen -l"La Habra"\n'
               f'gmt end')
    with open('plot_gof.gmt', 'w') as fid_gmt:
        fid_gmt.write(gmt_cmd)
    cmd = 'bash plot_gof.gmt'
    call_sub(cmd, shell=True)


if __name__ == "__main__":
    print("Leave it so.")
