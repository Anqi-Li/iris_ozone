#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:53:41 2020

@author: anqil

combined O2(1delta) and ozone retrieval from IRIS stray-light corrected limb radiance
O2 VER uses standard OEM 
Ozone uses Levenberg-Marquardt
a priori taken from CMAM 
other parameters (T, p) taken from MSIS
Sa has off-diagonal elements
Se scale with equilibrium index^4

"""

import numpy as np
import scipy.sparse as sp
import xarray as xr
import matplotlib.pyplot as plt
from o2delta_model import cal_o2delta_new
from astropy.time import Time
from astroplan import Observer
import astropy.units as u
from numba import jit
import os
import sys
import warnings
import datetime as dt

warnings.filterwarnings("ignore")
import gc


#%% load IRIS limb radiance data
filename = "ir_slc_031958_ch3.nc"
# path = 'https://arggit.usask.ca/opendap/'
path = "./data/StrayLightCorrected/"
# orbit = sys.argv[1]
ir = xr.open_dataset(path + filename)  # , chunks={"time": 100})
ir = ir.sel(time=~ir.indexes["time"].duplicated())
day_time_lst = ir.time[ir.sza < 90]  # select only dayglow
ir = ir.sel(time=day_time_lst)
sel_pixels = slice(14, 128)
altitude = ir.altitude.sel(pixel=sel_pixels)
l1 = ir.data.sel(pixel=sel_pixels)
error = ir.sel(pixel=sel_pixels).error
lst = ir.apparent_solar_time

#%% load absorption correction table
file = "abs_table_Voigt2_80.nc"
abs_table = xr.open_dataset(file)
# abs_table = xr.open_dataset('abs_table_Voight2_80.nc')
abs_table = abs_table.set_coords("tangent_pressure").swap_dims(
    {"tangent_altitude": "tangent_pressure"}
)

#%% load climatology (xa, x_initial, Sa, and forward_args)
path = ""
file = "msis_cmam_climatology_z200_lat8576.nc"
clima = xr.open_dataset(path + file)  # .interp(z=z*1e-3)
clima = clima.update(
    {"m": (clima.o.dims, (clima.o + clima.o2 + clima.n2) * 1e-6, {"unit": "cm-3"})}
)
# clima = clima.sel(month=month)
clima_o3 = clima.o3_vmr * clima.m  # cm-3
clima = clima.update({"o3": ((clima_o3.dims), clima_o3, {"unit": "cm-3"})})

#%% load chemical lifetime of o2delta
file = "o2delta_lifetime.nc"
lifetime = xr.open_dataset(file).lifetime
# lifetime.plot(y='z')

#%% construct overlap filter table
aaa = [
    150,
    77.4,
    175,
    75.8,
    200,
    74.4,
    225,
    73.2,
    250,
    72.0,
    275,
    70.8,
    300,
    69.8,
    325,
    68.8,
    350,
    67.8,
]
T_ref = np.array(aaa[::2])
fr_ref = np.array(aaa[1::2])
fr_ref = xr.DataArray(
    fr_ref, dims=("T"), coords=(T_ref,), name="overlap_filter", attrs={"units": "%"}
)

#%% Global variables
A_o2delta = 2.23e-4
# ====drop data below and above some altitudes
bot = 40e3
top = 110e3
# ====retireval grid
z = np.arange(10e3, 130e3, 1e3)  # m


#%% Functions
def cal_sunrise_h(lat, lon, time):
    points = Observer(longitude=lon * u.deg, latitude=lat * u.deg, elevation=89 * u.km)
    times = Time(time, format="datetime64")
    sunrise = points.sun_rise_time(times, which="previous")
    try:
        sunrise_h = (times - sunrise).sec / 3600  # unit: hour
    except:
        sunrise_h = np.nan
    return sunrise_h


def jacobian_num(fun, x_in, dx, args=()):
    x = x_in.copy()
    x[x < 0] = 1e-8  # for negative ozone
    #    y, *temp = fun(x, *args) #fun must return more than 1 variable
    y = fun(x, args)
    if isinstance(dx, (int, float, complex)):
        dx = np.ones(len(x)) * dx

    jac = np.empty((len(y), len(x)))
    x_perturb = x.copy()
    for i in range(len(x)):
        x_perturb[i] = x_perturb[i] + dx[i]
        y_perturb = fun(x_perturb, args)
        jac[:, i] = (y_perturb - y) / dx[i]
        x_perturb[i] = x[i]
    return jac


def LM_oem(y, K, x_old, y_fit_pro, Se_inv, Sa_inv, xa, D, gamma):
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)
    if len(y_fit_pro.shape) == 1:
        y_fit_pro = y_fit_pro.reshape(len(y_fit_pro), 1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa), 1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(xa), 1)

    A = K.T @ (Se_inv) @ (K) + Sa_inv + gamma * D
    b = K.T @ (Se_inv @ (y - y_fit_pro) - Sa_inv @ (x_old - xa))
    # Rodgers page 93 eq 5.35
    x_hat_minus_x_old = np.linalg.solve(A, b)
    x_hat = x_hat_minus_x_old + x_old
    return x_hat.squeeze()


def d_square_test(x_new, x_old, K, Se_inv, Sa_inv, *other_args):
    if len(x_new.shape) == 1:
        x_new = x_new.reshape(len(x_new), 1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(x_old), 1)
    S_hat_inv = Sa_inv + K.T.dot(Se_inv).dot(K)
    d_square = (x_old - x_new).T.dot(S_hat_inv).dot(x_old - x_new)
    return d_square.squeeze() / len(x_new)


def iteration(
    forward_fun, fit_fun, y, x_initial, forward_args=(), fit_args=(), max_iter=30
):
    dx = 1e-3 * x_initial  # determin step size for numerical jacobian
    gamma = 10
    status = 0
    K = jacobian_num(forward_fun, x_initial, dx, forward_args)
    y_fit_pro = forward_fun(x_initial, forward_args)
    cost_x_old, cost_y_old = oem_cost_pro(y, y_fit_pro, x_initial, *fit_args)
    cost_tot_old = cost_x_old + cost_y_old
    x_old = x_initial.copy()

    for n_evaluate in range(max_iter):
        #        print('iteration', n_evaluate)
        K = jacobian_num(forward_fun, x_old, dx, forward_args)
        LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
        x_new = fit_fun(y, K, LM_fit_args)
        y_fit_pro = forward_fun(x_new, forward_args)
        residual = y - y_fit_pro
        cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_new, *fit_args)
        cost_tot = cost_x + cost_y

        if cost_tot <= cost_tot_old:
            d2 = d_square_test(x_new, x_old, K, *fit_args)
            x_change = np.divide(abs(x_new - x_old), abs(x_new))
            gamma /= 10
            if x_change.max() < 1e-2:  # converged, stop iteration
                #                print('converged 1')
                status = 1
            elif d2 < 0.5:  # converged, stop iteration
                #                print('converged 2')
                status = 2

            x_old = x_new.copy()
            cost_x_old = cost_x.copy()
            cost_y_old = cost_y.copy()
            cost_tot_old = cost_tot.copy()

        while cost_tot > cost_tot_old:  # cost increases--> sub iteration
            gamma *= 10
            #            print('increased gamma', gamma)
            if gamma > 1e5:
                #                print('gamma reached to max')
                status = 3
                break
            LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
            x_hat = fit_fun(y, K, LM_fit_args)
            cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_hat, *fit_args)
            cost_tot = cost_x + cost_y

        if status != 0:
            break

    return x_new, K, gamma, residual, status, cost_x, cost_y, n_evaluate


def forward(x, forward_args):
    o2delta, *frac = cal_o2delta_new(x, *forward_args)
    return o2delta * A_o2delta


def fit(y, K, fit_args):
    #    x_hat = lsq(y, K)
    #    x_hat, *temp = linear_oem(y, K, *fit_args)
    x_hat = LM_oem(y, K, *fit_args)
    return x_hat


# path length 1d
def pathl1d_iris(h, z=np.arange(40e3, 110e3, 1e3), z_top=150e3):
    # z: retrieval grid in meter
    # z_top: top of the atmosphere under consideration
    # h: tangent altitude of line of sight
    if z[1] < z[0]:
        z = np.flip(z)  # retrieval grid has to be ascending
        print("z has to be fliped")

    Re = 6370e3  # earth's radius in m
    z = np.append(z, z_top) + Re
    h = h + Re
    pl = np.zeros((len(h), len(z) - 1))
    for i in range(len(h)):
        for j in range(len(z) - 1):
            if z[j + 1] > h[i]:
                pl[i, j] = np.sqrt(z[j + 1] ** 2 - h[i] ** 2)

    pathl = np.append(np.zeros((len(h), 1)), pl[:, :-1], axis=1)
    pathl = pl - pathl
    pathl = 2 * pathl
    return pathl


def oem_cost_pro(y, y_fit, x_hat, Se_inv, Sa_inv, xa, *other_args):
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)
    if len(y_fit.shape) == 1:
        y_fit = y_fit.reshape(len(y_fit), 1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa), 1)
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(xa), 1)
    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa) / len(y)
    cost_y = (y - y_fit).T.dot(Se_inv).dot(y - y_fit) / len(y)
    return cost_x.squeeze(), cost_y.squeeze()


# convert ordinary to relative AVK
def rel_avk(xa, A):
    # xa 1D array lenth n
    # A 2D array size (n, n)
    A_rel = np.zeros(A.shape)
    for i in range(len(xa)):
        for j in range(len(xa)):
            A_rel[i, j] = xa[j] * A[i, j] / xa[i]
    return A_rel


def mr_and_Sm(x_hat, K, Sa_inv, Se_inv):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat), 1)
    A = Sa_inv + K.T.dot(Se_inv).dot(K)
    b = K.T.dot(Se_inv)
    G = np.linalg.solve(A, b)  # gain matrix
    AVK = G.dot(K)
    MR = AVK.sum(axis=1)
    Se = np.linalg.inv(Se_inv)
    #    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T)  # retrieval noise covariance
    Ss = (
        (AVK - np.eye(len(AVK)))
        .dot(np.linalg.inv(Sa_inv))
        .dot((AVK - np.eye(len(AVK))).T)
    )
    #     Ss = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(Sa_inv).dot(np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv))
    return MR, AVK, Sm, Ss


def Sa_inv_off_diag(sigma, dz, h):
    # Rodgers P218 B.71
    n = len(sigma)
    alpha = np.exp(-dz / h)
    c0 = (1 + alpha ** 2) / (1 - alpha ** 2)
    c1 = -alpha / (1 - alpha ** 2)
    S = sp.diags([c1, c0, c1], [-1, 0, 1], shape=(n, n)).tocsr()
    S[0, 0] = 1 / (1 - alpha ** 2)
    S[-1, -1] = 1 / (1 - alpha ** 2)
    sigma_inv = 1 / sigma
    Sa_inv = S.multiply(sigma_inv[:, None].dot(sigma_inv[None, :]))
    return Sa_inv


def Sa_off_diag(sigma, dz, h):
    # Rodgers P38 Eq2.82
    # diag_elements: sigma squared
    n = len(sigma)
    alpha = np.exp(-dz / h)
    Sa = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Sa[i, j] = sigma[i] * sigma[j] * alpha ** (abs(i - j))
    return Sa


def linear_oem(y, K, Se_inv, Sa_inv, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa), 1)
    G = np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))
    x_hat = xa + G.dot(y - K.dot(xa))

    return x_hat.squeeze(), G


# @jit(nopython = True, cache = True)
def path_z(z_top, z_tangent, sol_zen, nsteps, extend=False):
    # z_top, z_tangent in km!!
    Re = 6375  # in km
    sol_zen /= 180 / np.pi
    B = np.arcsin((Re + z_tangent) * np.sin(np.pi - sol_zen) / (Re + z_top))
    S_top = np.sin(sol_zen - B) * (Re + z_top) / np.sin(np.pi - sol_zen)

    Ret2 = (Re + z_tangent) ** 2
    step = S_top / nsteps
    S_top_half = S_top - step / 2
    z_step = [
        np.sqrt(
            Ret2
            + (S_top_half - i * step) ** 2
            - 2 * (Re + z_tangent) * (S_top_half - i * step) * np.cos(np.pi - sol_zen)
        )
        - Re
        for i in range(nsteps)
    ]
    if extend:
        z_step.extend(z_step[-2::-1])  # do not count tangent point twice
    return z_step, step  # in km


def pathlength_correction(z, h, abs_table, nsteps, background_atm):
    # z, h: numpy array in km
    # abs_table, background_atm: xarray, coordinate in km
    pathlength_corrected = np.zeros((len(h), len(z)))
    pathlength_uncorrected = np.zeros((len(h), len(z)))
    for ipixel in range(len(h)):
        z_steps, step_size = path_z(
            z_top=abs_table.z_top.item(),
            z_tangent=h[ipixel],  # in km
            sol_zen=90,
            nsteps=nsteps,
            extend=True,
        )  # in km
        distance_from_z_top = np.arange(len(z_steps)) * step_size  # in km
        corr = abs_table.factor.interp(
            tangent_pressure=background_atm.p.interp(z=h[ipixel]),
            distance=distance_from_z_top,
        )
        layer_indices = z.searchsorted(z_steps)
        u = np.unique(layer_indices[np.where(layer_indices < len(z))])
        pathl_corrected = np.bincount(
            layer_indices[np.where(layer_indices < len(z))],
            corr[np.where(layer_indices < len(z))] * step_size,
        )  # in km
        pathl_uncorrected = np.bincount(
            layer_indices[np.where(layer_indices < len(z))],
            np.ones(len(layer_indices[np.where(layer_indices < len(z))])) * step_size,
        )  # in km
        pathlength_corrected[ipixel, : u.max() + 1] = pathl_corrected  # in km
        pathlength_uncorrected[ipixel, : u.max() + 1] = pathl_uncorrected  # in km
    return pathlength_corrected, pathlength_uncorrected


def pathlength_geometry_correction(z, h, abs_table, background_atm):
    # z, h: numpy array in km
    # abs_table, background_atm: xarray, coordinate in km
    if z[1] < z[0]:
        z = np.flip(z)  # retrieval grid has to be ascending
        print("z has to be fliped")
    Re = 6375  # earth's radius in km
    z_top_Re = abs_table.z_top.values + Re  # in km
    z_Re = z + Re  # in km
    h_Re = h + Re  # in km
    p_sum = np.zeros((len(h), len(z)))
    for i in range(len(h)):
        for j in range(len(z)):
            if z[j] > h[i]:
                p_sum[i, j] = np.sqrt(z_Re[j] ** 2 - h_Re[i] ** 2)
    p = np.roll(p_sum, 1, axis=1)
    p[:, 0] = 0
    p = p_sum - p  # pahtlength of each layer (half side) in km (len(h), len(z))
    p_extend = np.append(
        np.fliplr(p), p, axis=1
    )  # pathlength of each layer (2 sides) in km (len(h), len(z)*2)
    extend_layer_index = np.append(
        np.flip(np.arange(len(z))), np.arange(len(z))
    )  # (len(z)*2)
    p_top = np.sqrt(z_top_Re ** 2 - h_Re ** 2) - np.sqrt(
        z_Re[-1] ** 2 - h_Re ** 2
    )  # pathlength from z_top (200km) to the last layer of z in km (len(h))
    d_extend = (
        p_extend.cumsum(axis=1).T + p_top
    ).T - p_extend / 2  # distance from z top in km (len(h), len(z)*2)
    # p_extend_corr = np.zeros(p_extend.shape)
    p_corr = np.zeros(p.shape)
    for i in range(len(h)):
        corr = abs_table.factor.interp(
            tangent_pressure=background_atm.p.interp(z=h[i]), distance=d_extend[i, :]
        )
        #     p_extend_corr[i,:] = p_extend[i,:]*corr
        p_corr[i, :] = np.bincount(extend_layer_index, p_extend[i, :] * corr)
    return p_corr, p * 2


#%%Main reteival


def f(i):
    alt_chop_cond = (altitude.isel(time=i) > bot) & (
        altitude.isel(time=i) < top
    )  # select altitude range
    if alt_chop_cond.sum() < 2:
        print(
            "wrong alt range {}/{} in orbit {}".format(
                i - image_lst[0], len(image_lst), orbit_nr
            )
        )
        pass
    elif l1.isel(time=i).notnull().sum() == 0:
        print(
            "l1 is all nan {}/{} in orbit {}".format(
                i - image_lst[0], len(image_lst), orbit_nr
            )
        )
        pass
    elif error.isel(time=i).notnull().sum() == 0:
        print(
            "error is all nan {}/{} in orbit {}".format(
                i - image_lst[0], len(image_lst), orbit_nr
            )
        )
        pass

    else:
        try:
            print(
                "process image {}/{} in orbit {}".format(
                    i - image_lst[0], len(image_lst), orbit_nr
                )
            )
            #             print(i)

            background_atm = clima.sel(
                lat=ir.latitude.isel(time=i),
                month=Time(ir.time[i], format="datetime64").datetime.month,
                method="nearest",
            )
            # make apriori ver
            o3_clima = background_atm.o3.sel(lst=lst.isel(time=i), method="nearest")
            T_a = background_atm.T
            m_a = background_atm.m
            p_a = background_atm.p
            sol_zen = ir.sza[i].values.item()
            forward_args = (
                T_a.values,
                m_a.values,
                clima.z.values * 1e3,  # in m
                sol_zen,
                p_a.values,
            )
            ver_clima = forward(o3_clima.values, forward_args)
            ver_a = np.interp(z * 1e-3, clima.z, ver_clima)

            # make jacobian matrix
            h = (
                altitude.isel(time=i)
                .where(l1.isel(time=i).notnull(), drop=True)
                .where(alt_chop_cond, drop=True)
            )  # in m
            pathlength, pathlength_ref = pathlength_geometry_correction(
                z * 1e-3, h.values * 1e-3, abs_table, background_atm
            )  # in km
            K = pathlength * 1e5  # km-->cm

            # normalise with filter overlap
            ir_T = T_a.interp(z=h * 1e-3)
            overlap_filter = (
                fr_ref.interp(T=ir_T, kwargs=dict(fill_value="extrapolate")) * 1e-2
            )
            normalise = np.pi * 4 / overlap_filter.values
            limb = normalise * l1.isel(time=i).sel(pixel=h.pixel).values
            error_2 = (normalise * error.isel(time=i).sel(pixel=h.pixel).values) ** 2

            Se_inv = np.diag(1 / error_2)
            #            Sa_inv = np.diag(1/(0.75*ver_a)**2)
            Sa_inv = Sa_inv_off_diag(0.75 * ver_a, dz=1, h=5).toarray()
            ver_hat, G = linear_oem(limb, K, Se_inv, Sa_inv, ver_a)
            ver_mr, ver_A, ver_Sm, ver_Ss = mr_and_Sm(ver_hat, K, Sa_inv, Se_inv)
            ver_meas_error2 = np.diag(ver_Sm)
            ver_smooth_error2 = np.diag(ver_Ss)
            limb_fit = K.dot(ver_hat)
            limb_residual = limb - limb_fit
            ver_cost_x, ver_cost_y = oem_cost_pro(
                limb, limb_fit, ver_hat, Se_inv, Sa_inv, ver_a
            )
            ver_A_rel = rel_avk(ver_a, ver_A)
            ver_mr_rel = ver_A_rel.sum(axis=1)

            #%%%%%%%%%%% select only high ver_mr_rel to retireve o3 %%%%%%%%%%%
            cond_ver_mr = np.where(ver_mr_rel > 0.8)
            z_mr = z[cond_ver_mr]  # in meter
            ver_cond = ver_hat[cond_ver_mr]
            ver = np.interp(
                z_mr, z_mr[np.where(ver_cond > 0)], ver_cond[np.where(ver_cond > 0)]
            )  # eliminate neg ver
            o3_initial = o3_clima.interp(z=z_mr * 1e-3).values  # cm-3
            forward_args = (
                T_a.interp(z=z_mr * 1e-3).values,  # K
                m_a.interp(z=z_mr * 1e-3).values,  # cm-3
                z_mr,  # m
                sol_zen,  # degree
                p_a.interp(z=z_mr * 1e-3).values,
                True,
            )  # m
            hour_after_sunrise = cal_sunrise_h(
                ir.latitude.isel(time=i), ir.longitude.isel(time=i), ir.time[i]
            )
            if np.isnan(hour_after_sunrise):
                scale_lifetime = 1
            else:
                scale_lifetime = (
                    1 - np.exp(-hour_after_sunrise / lifetime.interp(z=z_mr * 1e-3))
                ) ** 4
            o3_a = o3_initial.copy()
            Sa_inv = Sa_inv_off_diag(0.75 * o3_a, dz=1, h=5).toarray()
            #            Se = ver_Sm[cond_ver_mr][:,cond_ver_mr].squeeze()
            #            Se_inv = np.linalg.inv(Se) #full Se matrix
            Se_inv = np.diag(
                scale_lifetime ** 2 / ver_meas_error2[cond_ver_mr]
            )  # only diaganol
            D = Sa_inv  # for Levenberg Marquardt
            fit_args = (Se_inv, Sa_inv, o3_a, D)

            LM_result = iteration(
                forward,
                fit,
                ver,
                o3_initial,
                forward_args=forward_args,
                fit_args=fit_args,
            )
            (
                o3_hat,
                K,
                o3_gamma,
                ver_residual,
                o3_status,
                o3_cost_x,
                o3_cost_y,
                o3_n_evaluate,
            ) = LM_result
            o3_mr, o3_A, o3_Sm, o3_Ss = mr_and_Sm(o3_hat, K, Sa_inv, Se_inv)
            o3_meas_error2 = np.diag(o3_Sm)
            o3_smooth_error2 = np.diag(o3_Ss)
            o3_A_rel = rel_avk(o3_a, o3_A)
            o3_mr_rel = o3_A_rel.sum(axis=1)

            #%%%%%%%% reindexing to an uniformed length %%%%%%%%%%%%%%%%%%%%%%%%%%
            o3_hat = xr.DataArray(o3_hat, coords={"z": z_mr}, dims="z").reindex(z=z)
            ver_residual = xr.DataArray(
                ver_residual, coords={"z": z_mr}, dims="z"
            ).reindex(z=z)
            o3_mr = xr.DataArray(o3_mr, coords={"z": z_mr}, dims="z").reindex(z=z)
            o3_mr_rel = xr.DataArray(o3_mr_rel, coords={"z": z_mr}, dims="z").reindex(
                z=z
            )
            o3_meas_error2 = xr.DataArray(
                o3_meas_error2, coords={"z": z_mr}, dims="z"
            ).reindex(z=z)
            o3_smooth_error2 = xr.DataArray(
                o3_smooth_error2, coords={"z": z_mr}, dims="z"
            ).reindex(z=z)
            limb_residual = xr.DataArray(
                limb_residual, coords={"pixel": h.pixel}, dims="pixel"
            ).reindex(pixel=l1.pixel)

            return (
                day_time_lst[i].values,
                # ver retrieval
                ver_hat,
                ver_mr,
                ver_mr_rel,
                np.sqrt(ver_meas_error2),
                np.sqrt(ver_smooth_error2),
                limb_residual,
                ver_cost_x,
                ver_cost_y,
                # o3 retrieval
                o3_hat,
                o3_mr,
                o3_mr_rel,
                np.sqrt(o3_meas_error2),
                np.sqrt(o3_smooth_error2),
                ver_residual,
                o3_cost_x,
                o3_cost_y,
                o3_n_evaluate,
                o3_gamma,
                o3_status,
                # apriori
                o3_clima.values,
                ver_clima,
                # the others
                hour_after_sunrise,
                ver_A_rel,
                o3_A_rel,  # not saving to nc file
                ver_A,
                o3_A,  # not saving to nc file
            )

        except:
            print("something is wrong ({})".format(i))

            #             raise
            pass


#%%
def run_and_save(image_lst, orbit_nr):
    result = []
    for i in image_lst:
        result.append(f(i))  # run and append retrieval

    result = [i for i in result if i]  # filter out all None element in list

    if len(result) > 0:
        mjd = np.stack([result[i][0] for i in range(len(result))])
        ds = xr.Dataset()
        ds = ds.update(
            {
                # coordinates
                "z": (["z"], z * 1e-3, {"units": "km"}),
                "mjd": (["mjd"], mjd),
                "clima_z": (["clima_z"], clima.z, {"units": "km"}),
                "pixel": (["pixel"], l1.pixel),
                # ver retrieval
                "ver": (
                    ["mjd", "z"],
                    np.stack([result[i][1] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                "ver_mr": (
                    ["mjd", "z"],
                    np.stack([result[i][2] for i in range(len(result))]),
                ),
                "ver_mr_rel": (
                    ["mjd", "z"],
                    np.stack([result[i][3] for i in range(len(result))]),
                ),
                "ver_retrieval_error": (
                    ["mjd", "z"],
                    np.stack([result[i][4] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                "ver_smoothing_error": (
                    ["mjd", "z"],
                    np.stack([result[i][5] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                "limb_residual": (
                    ["mjd", "pixel"],
                    np.stack([result[i][6] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                "ver_cost_x": (
                    ["mjd"],
                    np.stack([result[i][7] for i in range(len(result))]),
                ),
                "ver_cost_y": (
                    ["mjd"],
                    np.stack([result[i][8] for i in range(len(result))]),
                ),
                # o3 retrieval
                "o3": (
                    ["mjd", "z"],
                    np.stack([result[i][9] for i in range(len(result))]),
                    {"units": "cm-3"},
                ),
                "o3_mr": (
                    ["mjd", "z"],
                    np.stack([result[i][10] for i in range(len(result))]),
                ),
                "o3_mr_rel": (
                    ["mjd", "z"],
                    np.stack([result[i][11] for i in range(len(result))]),
                ),
                "o3_retrieval_error": (
                    ["mjd", "z"],
                    np.stack([result[i][12] for i in range(len(result))]),
                    {"units": "cm-3"},
                ),
                "o3_smoothing_error": (
                    ["mjd", "z"],
                    np.stack([result[i][13] for i in range(len(result))]),
                    {"units": "cm-3"},
                ),
                "ver_residual": (
                    ["mjd", "z"],
                    np.stack([result[i][14] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                "o3_cost_x": (
                    ["mjd"],
                    np.stack([result[i][15] for i in range(len(result))]),
                ),
                "o3_cost_y": (
                    ["mjd"],
                    np.stack([result[i][16] for i in range(len(result))]),
                ),
                "o3_evaluate": (
                    ["mjd"],
                    np.stack([result[i][17] for i in range(len(result))]),
                    {"note": "number of iterations"},
                ),
                "o3_gamma": (
                    ["mjd"],
                    np.stack([result[i][18] for i in range(len(result))]),
                ),
                "o3_status": (
                    ["mjd"],
                    np.stack([result[i][19] for i in range(len(result))]),
                    {"note": "1:converged by xchange, 2:converged by d2, 3:diverged"},
                ),
                # apriori
                "o3_a": (
                    ["mjd", "clima_z"],
                    np.stack([result[i][20] for i in range(len(result))]),
                    {"units": "cm-3", "name": "CMAM O3"},
                ),
                "ver_a": (
                    ["mjd", "clima_z"],
                    np.stack([result[i][21] for i in range(len(result))]),
                    {"units": "photons cm-3 s-1"},
                ),
                # the others
                "hour_after_sunrise": (
                    ["mjd"],
                    np.stack([result[i][22] for i in range(len(result))]),
                ),
                "longitude": (["mjd"], ir.longitude.sel(mjd=mjd), {"units": "deg E"}),
                "latitude": (["mjd"], ir.latitude.sel(mjd=mjd), {"units": "deg N"}),
                "sza": (["mjd"], ir.sza.sel(mjd=mjd), {"units": "deg"}),
                "lst": (["mjd"], lst.sel(mjd=mjd), {"units": "hour"}),
                "orbit": (["mjd"], ir.orbit.sel(mjd=mjd), {"name": "orbit number"}),
            }
        )
        ds.to_netcdf(
            "~/Documents/osiris_database/iris_ver_o3/ver_o3/v2p0/iri_ver_o3_orbit{}_v2p0.nc".format(
                orbit_nr
            )
        )


#%%%%%%%%%%%%%%%%%%%% main part #####################
if __name__ == "__main__":

    orbit_nr = np.unique(ir.orbit)[0]  # find orbit in dataset

    image_lst = ir.time.searchsorted(
        ir.where(ir.orbit == orbit_nr, drop=True).time
    )  # get list of all images
    run_and_save(image_lst, orbit_nr)
    gc.collect()
