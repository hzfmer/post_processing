# -*- coding: utf-8 -*-
"""
A few subroutines to be used for post-processing
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import imageio
import collections
import pickle
import re
import requests
import pandas as pd
from pathlib import Path
from filter_BU import filt_B
import my_pyrotd
from post_processing import __version__

__author__ = "Zhifeng Hu"
__copyright__ = "Zhifeng Hu, SDSU/UCSD"
__license__ = "mit"



def read_params(model):
    with open(Path(model, 'param.sh'), 'r') as fid:
        for line in fid:
            if "TMAX" in line:
                tmax = float(re.findall(r'(?<=TMAX )\d+\.+\d+(?= )', line)[0])
            if "DT" in line:
                dt = float(re.findall(r'(?<=DT )\d+\.+\d+(?= )', line)[0])
            if "NTISKP" in line:
                tskip = int(re.findall(r'NTISKP (\d+)', line)[0])
                dt = dt * tskip
            if "WRITE_STEP" in line:
                wstep = int(re.findall(r'WRITE_STEP (\d+)', line)[0])
    nfile = int(tmax / dt) // wstep
    return tmax, dt, tskip, wstep, nfile


def filt(vel, dt=1, lowcut=0.15, highcut=5, causal=False):
    if not issubclass(type(vel), dict):
        return filt_B(vel, 1 / dt, lowcut, highcut, causal=causal) 
    keys = list(vel.keys())
    data = dict()
    if 'dt' in keys:
        data['dt'] = vel['dt']
        keys.remove('dt')
    for comp in keys:
        data[comp] = filt_B(np.array(vel[comp], dtype='float32'), 1 / vel['dt'], lowcut=lowcut, highcut=highcut, causal=causal)
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
    tmax, dt, tskip, wstep, _ = read_params(model)
    nt = int(tmax / dt)
    model = Path(model, "output_sfc" if not case else f"output_sfc_{case}")
    fnum = int(np.ceil((it + 1) / wstep) * wstep * tskip)
    print(f'\r{it} / {nt}, fnum={fnum}', end="\r", flush=True)
    fid = open(f'{model}/S{comp}_0_{fnum:07d}', 'rb')
    skip = 4 * ((it - fnum // tskip + wstep) * mx * my)
    fid.seek(skip, 0)
    v = np.frombuffer(fid.read(mx * my * 4), dtype='float32').reshape(my, mx)
    v = v.copy()
    if np.isnan(v).any():
        print(f"NAN founded\n")
    return v, dt


def read_rec(site, metric='vel'):
    r = requests.get(f'http://hypocenter.usc.edu/research/High-F/lahabra_obs_2019_05_31/{site}.V2')
    if r.status_code != 200:
        print(f"the site {site} not found on the server")
        return None
    recordings = r.text.split('\r\n')

    stime = re.findall(r'\d+:\d+:\d+', recordings[4].split(',')[1])[0].split(':')
    channel = {1: "X", 2: "Z", 3: "Y"}
    count = 0
    data = {}
    for i in range(len(recordings)):
        if "points" in recordings[i] and metric in recordings[i]:    
            nums = re.findall(r'\d*\.*\d+', recordings[i])
            if len(nums) <= 1:  # skip headers
                continue
            count += 1
            npts = int(nums[0])
            dt_rec = float(nums[1])        
            start = i + 1
            end = i + (npts - 1) // 8 + 2
            # convert cm/s/s to m/s/s
            data[channel[count]] = np.array([float(x) / 100 for y in recordings[start : end] \
                                           for x in y.replace('-', ' -').split()])
            assert(len(data[channel[count]]) == npts)
            # data[channel[count]] = filt_B(np.array(data[channel[count]], dtype='float32'),1 / dt_rec, causal=False)
    print(f"Number of points = {npts}, dt_recording = {dt_rec}")
    data["dt"] = dt_rec
    return data


def read_syn(ix, iy, mx, my, model="", case=""):
    data = collections.defaultdict(list)
    _, dt_syn, tskip, wstep, nfile = read_params(model)
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


def pick_vel(mx, my, models, syn_sites={}):
    try: 
        with open('results/vel_rec.pickle', 'rb') as fid:
            vel_rec = pickle.load(fid)
        with open('results/vel_syn.pickle', 'rb') as fid:
            vel_syn = pickle.load(fid)

    except Exception as e:
        print("Error: ", e, "\nRebuild vel database")
        vel_rec = {}
        vel_syn = collections.defaultdict(dict)
    
    if all(x in vel_syn.keys() for x in models):
        print("Models queried!")
        return vel_rec, {k:vel_syn[k] for k in models}
    init_len = len(vel_syn)
    _models = models.copy()
    for key in vel_syn.keys():
        if key in _models:
            _models.remove(key)
    for isite in range(len(syn_sites)):
        site_name = syn_sites[isite][0]
        print(f'Gathering {len(_models)} ground motions at site {site_name}')
        ix, iy = syn_sites[isite][1:]
        if site_name not in vel_rec:
            vel_rec[site_name] = read_rec(site_name)
        for model in _models:
            if model not in vel_syn.keys():
                vel_syn[model] = dict()
            vel_syn[model][site_name] = read_syn(ix, iy, mx, my, model=model) 
            
    if len(vel_syn) > init_len:
        with open('results/vel_rec.pickle', 'wb') as fid:
            pickle.dump(vel_rec, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open('results/vel_syn.pickle', 'wb') as fid:
            pickle.dump(vel_syn, fid, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    return vel_rec, {k: vel_syn[k] for k in models}


def pick_psa(mx, my, models, osc_freqs=np.logspace(-1, 1, 91), osc_damping=0.05, syn_sites={}):
    '''Import psa response if available
    If psa pickle files don't exist, create them.
    If some other models are queried, compute them and append to psa files.
    '''
    try: 
        with open('results/psa_rec.pickle', 'rb') as fid:
            rotd_rec = pickle.load(fid)
        with open('results/psa_syn.pickle', 'rb') as fid:
            rotd_syn = pickle.load(fid)
    except Exception as e:
        rotd_rec = dict()
        rotd_syn = collections.defaultdict(dict)
    
    if all(x in rotd_syn.keys() for x in models):
        print("Models queried!")
        return rotd_rec, {k:rotd_syn[k] for k in models}
    
    # Remove existing models except extra model queried
    init_len = len(rotd_syn)
    _models = models.copy()   
    for key in rotd_syn.keys():
        if key in _models:
            _models.remove(key)
    vel_rec, vel_syn = pick_vel(mx, my, models.copy(), syn_sites)
            
    for isite in range(len(syn_sites)):
        print(f"\rComputing site {isite} / {len(syn_sites)}", end="\r", flush=True)
        site_name = syn_sites[isite][0]
        if site_name not in rotd_rec:
            print(f"\nGathering rotd_rec for {site_name}\n")
            # If site not found on the server
            if not vel_rec[site_name]:
                continue
            accx, accy = (np.diff(x, prepend=0) / vel_rec[site_name]['dt'] \
                          for x in [vel_rec[site_name]['X'], vel_rec[site_name]['Y']])
            rotd_rec[site_name] = my_pyrotd.my_calc_rotated_spec_accels(
                vel_rec[site_name]['dt'], accx, accy,
                osc_freqs, osc_damping, percentiles=[50])

        for model in _models:
            if model not in rotd_syn.keys():
                rotd_syn[model] = dict()
            print(f"\rGathering rotd_syn for {site_name} of {model}", end='\r', flush=True)
            accx, accy = (np.diff(x, prepend=0) / vel_syn[model][site_name]['dt'] \
                          for x in [vel_syn[model][site_name]['X'], vel_syn[model][site_name]['Y']])
            rotd_syn[model][site_name] = my_pyrotd.my_calc_rotated_spec_accels(
                vel_syn[model][site_name]['dt'], accx, accy,
                osc_freqs, osc_damping, percentiles=[50])
    
    # Rewrite if more models queried than existing
    if len(rotd_syn) > init_len:
        with open('results/psa_rec.pickle', 'wb') as fid:
            pickle.dump(rotd_rec, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open('results/psa_syn.pickle', 'wb') as fid:
            pickle.dump(rotd_syn, fid, protocol=pickle.HIGHEST_PROTOCOL)
            
    return rotd_rec, {k:rotd_syn[k] for k in models}


#def comp_dur(
def plot_snapshot(it, mx, my, model="", case="", draw=False, backend="inline"):   
    vx, dt = read_snapshot(it, mx, my, model=model, case=case)
    fig, ax = plt.subplots()
    im = ax.imshow(vx.T,cmap='bwr')
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


def plot_validation(site_name, models, shift=24, lowcut=0.15, highcut=5, metrics=['vel', 'pas'], syn_sites={}, backend="inline", plot_rec=True):
    #vel_rec, vel_syn = pick_vel(0, 0, models)
    #psa_rec, psa_syn = pick_psa(0, 0, models)  # use 0 because we have always prepared these dicts.
    with open('results/vel_rec.pickle', 'rb') as fid:
        vel_rec = pickle.load(fid)
    with open('results/vel_syn.pickle', 'rb') as fid:
        vel_syn = pickle.load(fid)
    print(models)
    if 'rec' in models:
        model.remove('rec')
    if not vel_rec[site_name]:
        return None
    dt_rec = vel_rec[site_name]['dt']   
    len_rec = len(vel_rec[site_name]['X'])
    t_rec = np.arange(len_rec) * dt_rec
    comp = {0: "X", 1: "Y", 2: "Z"}
    image =[]
    if 'vel' in metrics:
        fig, ax = plt.subplots(3, 1, dpi=400)
        fig.tight_layout()
        fig.suptitle(f'{site_name}')
        for i in range(3):   
            if plot_rec:
                vel = filt_B(vel_rec[site_name][comp[i]], 1 / dt_rec, lowcut=lowcut, highcut=highcut, causal=False)
                ax[i].plot(t_rec, vel, label='rec')
            for model in models:
                dt_syn = vel_syn[model][site_name]['dt']
                len_syn = len(vel_syn[model][site_name][comp[i]])
                t_syn = np.arange(len_syn) * dt_syn + shift
                vel = filt_B(vel_syn[model][site_name][comp[i]], 1 / dt_syn, lowcut=lowcut, highcut=highcut, causal=False)
                ax[i].plot(t_syn, vel, lw=0.8, label=model + f", Max = {np.max(vel):.4f}")

            ax[i].set_ylabel(f'V{comp[i]} (m/s)')
            ax[i].set_xlim(shift + 2, shift + 30)
        ax[-1].set_xlabel('Time (s)')
        ax[0].legend(loc=1)
        fig.canvas.draw()
        if backend == "inline":
            temp = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image += [temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))]
    
    if 'psa' in metrics:
        fig2, ax = plt.subplots(dpi=400)
        fig2.tight_layout()
        fig2.suptitle(f'{site_name}') 
        if plot_rec:
            psa = psa_rec[site_name]
            ax.plot(psa.osc_freq, psa.spec_accel, label='rec')
        for model in models:
            psa = psa_syn[model][site_name]
            ax.plot(psa.osc_freq, psa.spec_accel, label=model)
        ax.set(ylabel=f'$SA (m/s^2)$', xscale='log', yscale='log')
        ax.xaxis.grid(True, which='both')
        ax.yaxis.grid(True, which='both')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend()
        fig2.canvas.draw()
        if backend == "inline":
            temp = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
            image += [temp.reshape(fig2.canvas.get_width_height()[::-1] + (3,))]
                             
    return image if backend == "inline" else [fig, fig2]


def comp_cum_energy(vel, dt=0.01):
    if issubclass(type(vel), dict):
        dt = vel['dt']
        keys = list(vel.keys())
        keys.remove('dt')
        cumvel = np.zeros((len(vel[keys[0]]),), dtype='float32')
        for k in keys:
            cumvel += comp_cum_energy(vel[k], dt=dt)
        return cumvel / len(keys) * dt
    vel = np.array(vel)
    return np.cumsum(vel ** 2) * dt


def plot_cum_energy(site_name, models, shift=24, lowcut=0.15, highcut=5, syn_sites={}, backend="inline", plot_rec=True):
    vel_rec, vel_syn = pick_vel(0, 0, models)
    
    site_name = syn_sites[isite][0]
    if not vel_rec[site_name]:
        plot_rec = False
    else:
        dt_rec = vel_rec[site_name]['dt']   
        len_rec = len(vel_rec[site_name]['X'])
        t_rec = np.arange(len_rec) * dt_rec
    len_syn = len(vel_syn[models[0]][site_name]['X'])
    
    comp = {0: "X", 1: "Y", 2: "Z"}
    image =[]
    fig, ax = plt.subplots(3, 1, dpi=400)
    fig.tight_layout()
    fig.suptitle(f'{site_name}')
    for i in range(3):   
        if plot_rec:
            vel = filt_B(vel_rec[site_name][comp[i]], 1 / dt_rec, lowcut=lowcut, highcut=highcut, causal=False)
            vel = np.cumsum(vel ** 2)
            ax[i].plot(t_rec, vel, label='rec')
        for model in models:
            dt_syn = vel_syn[model][site_name]['dt']
            len_syn = len(vel_syn[model][site_name]['X'])
            t_syn = np.arange(len_syn) * dt_syn + shift
            vel = filt_B(vel_syn[model][site_name][comp[i]], 1 / dt_syn, lowcut=lowcut, highcut=highcut, causal=False)
            ax[i].plot(t_syn, vel, lw=1, label=model)

        ax[i].set_ylabel(f'V{comp[i]} (m/s)')
        ax[i].set_xlim(shift, 50)
    ax[-1].set_xlabel('Time (s)')
    ax[0].legend()
    fig.canvas.draw()
    if backend == "inline":
        temp = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image += [temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))]
                             
    return image if backend == "inline" else [fig]


if __name__ == "__main__":
    print("Nothing to tell you")
