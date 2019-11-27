import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
import os as os


def fit_state_space_with_g_func_grp_ud():
    pass


def fit_state_space_with_g_func_2_state_grp_ud():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    p_rec = np.empty((0, 10))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    x_obs_all = d.groupby(['cnd', 'Target', 'trial', 'rot_dir']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        for j in ('cw', 'ccw'):
            x_obs = x_obs_all[x_obs_all['cnd'] == i]
            x_obs = x_obs[x_obs['rot_dir'] == j]
            x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            args = (x_obs, rot)
            bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60), (0, 1),
                      (0, 1), (0, 60), (0, 1))
            results = differential_evolution(func=fit_obj_func_sse_2_state_ud,
                                             bounds=bounds,
                                             args=args,
                                             maxiter=300,
                                             disp=True,
                                             polish=True,
                                             updating="deferred",
                                             workers=-1)
            p = results["x"]
            p_rec = np.append(p_rec, [p], axis=0)

            f_name_p = "../fits/fit_2state_grp_ud" + str(i) + '_' + j
            with open(f_name_p, "w") as f:
                np.savetxt(f, p, "%0.4f", ",")

    return (p_rec)


def simulate_state_space_with_g_func_2state_usedependent(p, rot):
    p = [
        0.4548, 0.9987, 59.1602, 0.7685, 0.3191, 19.0796, 0.9592, 0.9605,
        12.0569, 0.4107
    ]
    rot = np.concatenate((np.zeros(120 * 4), np.random.normal(0, 2, 120),
                          np.random.normal(15, 2,
                                           400), np.random.normal(15, 2, 72),
                          np.random.normal(15, 2, 100), np.zeros(100)))
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]
    alpha_ud = p[6]
    beta_ud = p[7]
    sigma_ud = p[8]
    w_mix = p[9]
    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    def g_func(theta, theta_mu, sigma):
        if sigma != 0:
            G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        else:
            G = np.zeros(12)
        return (G)

    # Just do training
    n_simulations = 100
    x = np.zeros((12, num_trials))
    xs = np.zeros((12, num_trials))
    xf = np.zeros((12, num_trials))
    xud = np.zeros((12, num_trials))
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        elif i > 1172:
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        G = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)

        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xf[:, i + 1] = beta_f * xf[:, i]
            xud[:, i + 1] = beta_ud * xud[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * G
            xf[:, i + 1] = beta_f * xf[:, i] - alpha_f * delta[i] * G
            xud[:, i + 1] = beta_ud * xud[:, i] + alpha_ud * g_func(
                theta_values, xud[theta_ind[i], i], sigma_ud)

        x[:, i + 1] = np.vstack((w_mix * (xs[:, i + 1] + xf[:, i + 1]),
                                 (1 - w_mix) * xud[:, i + 1])).mean(0)

    for i in range(12):
        plt.plot(x[i, :])
    plt.show()

    xg = np.mean(x[:, 1000:1072], 1)
    plt.plot(xg, '-')
    plt.plot(xg, '.')
    plt.xticks(ticks=np.arange(0, 12, 1), labels=theta_values)
    plt.show()

    return x.T


def simulate_state_space_with_g_func_usedependent(p, rot):
    # p = [0.1, 0.99, 30, 0.05, 0.999, 30, 0.7]
    # rot = np.concatenate((np.zeros(120 * 4), np.random.normal(0, 2, 120),
    #                       np.random.normal(15, 2,
    #                                        400), np.random.normal(15, 2, 72),
    #                       np.random.normal(15, 2, 100), np.zeros(100)))
    alpha = p[0]
    beta = p[1]
    g_sigma = p[2]
    ud_alpha = p[3]
    ud_beta = p[4]
    ud_sigma = p[5]
    w_mix = p[6]

    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    def g_func(theta, theta_mu, sigma):
        if sigma != 0:
            G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        else:
            G = np.zeros(12)
        return (G)

    # Just do training
    n_simulations = 100
    x = np.zeros((12, num_trials))
    ud = np.zeros((12, num_trials))
    xud = np.zeros((12, num_trials))
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        elif i > 1172:
            delta[i] = 0.0
        else:
            delta[i] = xud[theta_ind[i], i] - rot[i]

        G = g_func(theta_values, theta_values[theta_ind[i]], g_sigma)

        if np.isnan(rot[i]):
            x[:, i + 1] = beta * x[:, i]
            ud[:, i + 1] = ud_beta * ud[:, i]
        else:
            x[:, i + 1] = beta * x[:, i] - alpha * delta[i] * G
            ud[:, i + 1] = ud_beta * ud[:, i] + ud_alpha * g_func(
                theta_values, xud[theta_ind[i], i], ud_sigma)

        xud[:, i + 1] = np.vstack(
            (w_mix * x[:, i + 1], (1 - w_mix) * ud[:, i + 1])).mean(0)

    return xud.T


def simulate_state_space_with_g_func_2_state(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]

    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    def g_func(theta, theta_mu, sigma):
        if sigma != 0:
            G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        else:
            G = np.zeros(12)
        return (G)

    # Just do training
    n_simulations = 100
    x = np.zeros((12, num_trials))
    xs = np.zeros((12, num_trials))
    xf = np.zeros((12, num_trials))
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        elif i > 1172:
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        Gs = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        Gf = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_f)
        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xf[:, i + 1] = beta_f * xf[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * Gs
            xf[:, i + 1] = beta_f * xf[:, i] - alpha_f * delta[i] * Gf

        x[:, i + 1] = xs[:, i + 1] + xf[:, i + 1]

    return (xs.T + xf.T)


def simulate_state_space_with_g_func(p, rot):
    alpha = p[0]
    beta = p[1]
    g_sigma = p[2]
    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    def g_func(theta, theta_mu, sigma):
        if sigma != 0:
            G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        else:
            G = np.zeros(12)
        return (G)

    # Just do training
    n_simulations = 100
    x = np.zeros((12, num_trials))
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        elif i > 1172:
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        G = g_func(theta_values, theta_values[theta_ind[i]], g_sigma)

        if np.isnan(rot[i]):
            x[:, i + 1] = beta * x[:, i]
        else:
            x[:, i + 1] = beta * x[:, i] - alpha * delta[i] * G

    return x.T


def fit_obj_func_sse(params, *args):
    x_obs = args[0]
    rot = args[1]
    x_pred = simulate_state_space_with_g_func(params, rot)

    sse_rec = np.zeros(12)
    for i in range(12):
        sse_rec[i] = (np.nansum((x_obs[:, i] - x_pred[:, i])**2))
        sse = np.nansum(sse_rec)
    return (sse)


def fit_obj_func_sse_2_state_ud(params, *args):
    alpha_s = params[0]
    beta_s = params[1]
    g_sigma_s = params[2]
    alpha_f = params[3]
    beta_f = params[4]
    g_sigma_f = params[5]
    alpha_ud = params[6]
    beta_ud = params[7]
    g_sigma_ud = params[8]
    w_mix = params[9]

    if alpha_s >= alpha_f:
        sse = 10**100
    elif beta_s <= beta_f:
        sse = 10**100
    else:
        x_obs = args[0]
        rot = args[1]
        x_pred = simulate_state_space_with_g_func_2state_usedependent(
            params, rot)

        sse_rec = np.zeros(12)
        for i in range(12):
            sse_rec[i] = np.nansum((x_obs[:, i] - x_pred[:, i])**2)
            sse = np.nansum(sse_rec)

    return (sse)


def fit_obj_func_sse_2_state(params, *args):
    alpha_s = params[0]
    beta_s = params[1]
    g_sigma_s = params[2]
    alpha_f = params[3]
    beta_f = params[4]
    g_sigma_f = params[5]

    if alpha_s >= alpha_f:
        sse = 10**100
    elif beta_s <= beta_f:
        sse = 10**100
    else:
        x_obs = args[0]
        rot = args[1]
        x_pred = simulate_state_space_with_g_func_2_state(params, rot)

        sse_rec = np.zeros(12)
        for i in range(12):
            sse_rec[i] = (np.nansum((x_obs[:, i] - x_pred[:, i])**2))
            sse = np.nansum(sse_rec)

    return (sse)


def fit_state_space_with_g_func_2_state_grp_bootstrap():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    n_boot_samp = 1000

    for i in d['cnd'].unique():
        for j in d['rot_dir'].unique():
            p_rec = -1 * np.ones((n_boot_samp, 6))
            for b in range(n_boot_samp):
                print(i, j, b)
                dd = d[(d['cnd'] == i) & (dd['rot_dir'] == j)]
                rot = dd['Appl_Perturb'].values[0:1272]
                subs = dd['sub'].unique()
                boot_subs = np.random.choice(subs,
                                             size=subs.shape[0],
                                             replace=True)
                x_boot_rec = []
                for k in boot_subs:
                    x_boot_rec.append(d[d['sub'] == k])
                    x_boot = pd.concat(x_boot_rec)

                x_obs = x_boot.groupby(['cnd', 'rot_dir', 'Target',
                                        'trial']).mean()
                x_obs.reset_index(inplace=True)

                x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
                x_obs = x_obs.pivot(index="trial",
                                    columns="target_deg",
                                    values="Endpoint_Error")

                x_obs = x_obs.values

                args = (x_obs, rot)
                bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60))
                results = differential_evolution(func=fit_obj_func_sse_2_state,
                                                 bounds=bounds,
                                                 args=args,
                                                 maxiter=300,
                                                 disp=False,
                                                 polish=False,
                                                 updating="deferred",
                                                 workers=-1)
                p = results["x"]
                p_rec[b, :] = p

                f_name_p = "../fits/fit_grp_2state_bootstrap_" + str(
                    i) + '_' + j
                with open(f_name_p, "w") as f:
                    np.savetxt(f, p_rec, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_grp_bootstrap():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    n_boot_samp = 1000
    p_rec = -1 * np.ones((n_boot_samp, 3))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    for b in range(n_boot_samp):
        for i in d['cnd'].unique():
            for j in d['rot_dir'].unique():
                print(b, i, j)
                dd = d[d['cnd'] == i]
                dd = dd[dd['rot_dir'] == j]
                subs = dd['sub'].unique()
                boot_subs = np.random.choice(subs,
                                             size=subs.shape[0],
                                             replace=True)
                x_boot_rec = []
                for k in boot_subs:
                    x_boot_rec.append(d[d['sub'] == k])
                    x_boot = pd.concat(x_boot_rec)

                x_obs = x_boot.groupby(['cnd', 'Target', 'trial']).mean()
                x_obs.reset_index(inplace=True)

                x_obs = x_obs[[
                    "Endpoint_Error", "Target", "target_deg", "trial"
                ]]
                x_obs = x_obs.pivot(index="trial",
                                    columns="target_deg",
                                    values="Endpoint_Error")

                x_obs = x_obs.values

                args = (x_obs, rot)
                bounds = ((0, 1), (0, 1), (0, 60))
                results = differential_evolution(func=fit_obj_func_sse,
                                                 bounds=bounds,
                                                 args=args,
                                                 maxiter=300,
                                                 disp=False,
                                                 polish=True,
                                                 updating="deferred",
                                                 workers=-1)
                p = results["x"]
                p_rec[b, :] = p

                f_name_p = "../fits/fit_grp_bootstrap_" + str(i) + '_' + j
                with open(f_name_p, "w") as f:
                    np.savetxt(f, p_rec, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    p_rec = np.empty((0, 3))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    x_obs_all = d.groupby(['cnd', 'Target', 'trial', 'rot_dir']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        for j in ('cw', 'ccw'):
            x_obs = x_obs_all[x_obs_all['cnd'] == i]
            x_obs = x_obs[x_obs['rot_dir'] == j]
            x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            args = (x_obs, rot)
            bounds = ((0, 1), (0, 1), (0, 60))
            results = differential_evolution(func=fit_obj_func_sse,
                                             bounds=bounds,
                                             args=args,
                                             maxiter=300,
                                             disp=False,
                                             polish=True,
                                             updating="deferred",
                                             workers=-1)
            p = results["x"]
            x_pred = simulate_state_space_with_g_func(p, rot)

            c = cm.rainbow(np.linspace(0, 1, 12))

            p_rec = np.append(p_rec, [p], axis=0)

            f_name_p = "../fits/fit_grp_" + str(i) + '_' + j
            with open(f_name_p, "w") as f:
                np.savetxt(f, p, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_2_state_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    p_rec = np.empty((0, 6))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    x_obs_all = d.groupby(['cnd', 'Target', 'trial', 'rot_dir']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        for j in ('cw', 'ccw'):
            x_obs = x_obs_all[x_obs_all['cnd'] == i]
            x_obs = x_obs[x_obs['rot_dir'] == j]
            x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            args = (x_obs, rot)
            bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60))
            results = differential_evolution(func=fit_obj_func_sse_2_state,
                                             bounds=bounds,
                                             args=args,
                                             maxiter=300,
                                             disp=False,
                                             polish=True,
                                             updating="deferred",
                                             workers=-1)
            p = results["x"]
            x_pred = simulate_state_space_with_g_func(p, rot)

            c = cm.rainbow(np.linspace(0, 1, 12))

            p_rec = np.append(p_rec, [p], axis=0)

            f_name_p = "../fits/fit_2state_grp_" + str(i) + '_' + j
            with open(f_name_p, "w") as f:
                np.savetxt(f, p, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_2_state():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    sub = d['sub'].unique()
    length_names = sub.shape[0]

    p_rec = np.empty((0, 6))
    for i in range(length_names):
        print(sub[i])
        x_obs = d[d['sub'] == sub[i]]
        rot = x_obs["Appl_Perturb"].values
        x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
        x_obs = x_obs.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")

        x_obs = x_obs.values

        args = (x_obs, rot)
        bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60))
        results = differential_evolution(func=fit_obj_func_sse_2_state,
                                         bounds=bounds,
                                         args=args,
                                         maxiter=300,
                                         disp=False,
                                         polish=True,
                                         updating="deferred",
                                         workers=-1)
        p = results["x"]
        x_pred = simulate_state_space_with_g_func(p, rot)

        c = cm.rainbow(np.linspace(0, 1, 12))

        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = "../fits/fit_2state" + str(sub[i])
        with open(f_name_p, "w") as f:
            np.savetxt(f, p, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    sub = d['sub'].unique()
    length_names = sub.shape[0]

    p_rec = np.empty((0, 3))
    for i in range(length_names):
        print(sub[i])
        x_obs = d[d['sub'] == sub[i]]
        rot = x_obs["Appl_Perturb"].values
        x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
        x_obs = x_obs.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")

        x_obs = x_obs.values

        args = (x_obs, rot)
        bounds = ((0, 1), (0, 1), (0, 60))
        results = differential_evolution(func=fit_obj_func_sse,
                                         bounds=bounds,
                                         args=args,
                                         maxiter=300,
                                         disp=False,
                                         polish=True,
                                         updating="deferred",
                                         workers=-1)
        p = results["x"]
        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = "../fits/fit_" + str(sub[i])
        with open(f_name_p, "w") as f:
            np.savetxt(f, p, "%0.4f", ",")

    return p_rec


def inspect_fits_boot():

    fits_names = []
    for f in os.listdir('../fits'):
        if '2state_bootstrap' in f:
            fits_names.append('../fits/' + f)

    fits_names = np.sort(fits_names)
    fits_list = [pd.read_csv(x, header=None) for x in fits_names]
    fits_list = [x[0:750] for x in fits_list]

    d = pd.concat(fits_list)

    cnd = np.repeat([0, 1, 2], 1500)
    rot_dir = np.tile(np.repeat(['cw', 'ccw'], 750), 3)

    d['cnd'] = cnd
    d['rot_dir'] = rot_dir

    d.columns = [
        'alpha_s', 'beta_s', 'sigma_s', 'alpha_f', 'beta_f', 'sigma_f', 'cnd',
        'rot_dir'
    ]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # ax[0, 0].hist([
    #     d[d['cnd'] == 0 & (d['rot_dir'] == 'cw')].alpha_s.values,
    #     d[d['cnd'] == 1 & (d['rot_dir'] == 'cw')].alpha_s.values,
    #     d[d['cnd'] == 2 & (d['rot_dir'] == 'cw')].alpha_s.values
    # ],
    #               stacked=False)
    ax[0, 0].hist([
        d[d['cnd'] == 0 & (d['rot_dir'] == 'cw')].alpha_s.values,
    ],
                  stacked=False)
    ax[0, 0].hist([
        d[d['cnd'] == 1 & (d['rot_dir'] == 'cw')].alpha_s.values,
    ],
                  stacked=False)
    plt.show()

    # plt.savefig(
    #     '../figures/something.png')  # TODO: do something less stupid here
    # plt.close()


def inspect_fits_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    rot = d[d['sub'] == 1]['Appl_Perturb'].values
    x_obs_all = d.groupby(['cnd', 'trial', 'Target', 'rot_dir']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        for j in ('cw', 'ccw'):

            x_obs = x_obs_all[x_obs_all['cnd'] == i]
            x_obs = x_obs[x_obs['rot_dir'] == j]
            x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

            # NOTE: 1 state
            params = np.loadtxt("../fits/fit_grp_" + str(i) + '_' + j)
            x_pred = simulate_state_space_with_g_func(params, rot)

            for ii in range(12):
                ax[0, 0].scatter(np.arange(0, x_obs.shape[0]),
                                 x_obs[:, ii],
                                 alpha=0.05)
                ax[0, 0].plot(x_pred)

            xg_obs = np.nanmean(x_obs[1000:1072, :], 0)
            xg_pred = np.mean(x_pred[1000:1072, :], 0)
            ax[1, 0].plot(xg_obs, '-')
            ax[1, 0].plot(xg_pred, '-')
            ax[1, 0].set_xticks(ticks=np.arange(0, 12, 1))
            ax[1, 0].set_xticklabels(labels=theta_values)

            # NOTE: 2 state
            params = np.loadtxt("../fits/fit_2state_grp_" + str(i) + '_' + j)
            x_pred = simulate_state_space_with_g_func_2_state(params, rot)

            for ii in range(12):
                ax[0, 1].scatter(np.arange(0, x_obs.shape[0]),
                                 x_obs[:, ii],
                                 alpha=0.05)
                ax[0, 1].plot(x_pred)

            xg_obs = np.nanmean(x_obs[1000:1072, :], 0)
            xg_pred = np.mean(x_pred[1000:1072, :], 0)
            ax[1, 1].plot(xg_obs, '-')
            ax[1, 1].plot(xg_pred, '-')
            ax[1, 1].set_xticks(ticks=np.arange(0, 12, 1))
            ax[1, 1].set_xticklabels(labels=theta_values)

            plt.savefig('../figures/fit_combined_grp_' + str(i) + '_' +
                        str(j) + ".png")
            plt.close()


def inspect_fits():
    f_name = '../fit_input/master_data.csv'
    d = pd.read_csv(f_name)
    sub = d['sub'].unique()
    length_names = sub.shape[0]

    for i in range(length_names):

        x_obs = d[d['sub'] == sub[i]]
        rot = x_obs["Appl_Perturb"].values
        x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
        x_obs = x_obs.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")
        x_obs = x_obs.values

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        # NOTE: 1 state
        params = np.loadtxt("../fits/fit_" + str(sub[i]))
        x_pred = simulate_state_space_with_g_func(params, rot)

        for ii in range(12):
            ax[0].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[0].plot(x_pred)

        # NOTE: 2 state
        params = np.loadtxt("../fits/fit_2state" + str(sub[i]))
        x_pred = simulate_state_space_with_g_func_2_state(params, rot)

        for ii in range(12):
            ax[1].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[1].plot(x_pred)

        plt.savefig('../figures/fit_combined_' + str(sub[i]) + ".png")
        plt.close()


# start_time = time.time()

# fit_state_space_with_g_func()
# fit_state_space_with_g_func_2_state()
# fit_state_space_with_g_func_grp()
# fit_state_space_with_g_func_2_state_grp()
# fit_state_space_with_g_func_grp_ud()
# fit_state_space_with_g_func_2_state_grp_ud()
# fit_state_space_with_g_func_grp_bootstrap()
# fit_state_space_with_g_func_2_state_grp_bootstrap()

# end_time = time.time()
# print("Execution Time = " + str(end_time - start_time))

# inspect_fits()
# inspect_fits_grp()
# inspect_fits_boot()
