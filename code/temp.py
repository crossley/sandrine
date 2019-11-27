import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
import os as os


def inspect_fits():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    for i in d['cnd'].unique():
        for j in d['rot_dir'].unique():
            print(i, j)
            dd = d[(d['cnd'] == i) & (d['rot_dir'] == j)]
            rot = dd['Appl_Perturb'].values[0:1272]
            x_obs = dd.groupby(['cnd', 'rot_dir', 'Target', 'trial']).mean()
            x_obs.reset_index(inplace=True)
            x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            p = np.loadtxt('../fits/fit_' + str(i) + '_' + j, delimiter=',')
            x_pred = simulate_two_state_ud(p, rot)


            a = 0.1 * np.ones(12)
            a[5] = 1.0
            plt.subplot(121)
            for k in range(12):
                plt.plot(x_obs[:, k], '--', alpha = a[k])
                plt.plot(x_pred[:, k], '-', alpha = a[k])
                plt.ylim([-20, 20])

            plt.subplot(122)
            xg_obs = np.nanmean(x_obs[1000:1072, :], 0)
            xg_pred = np.nanmean(x_pred[1000:1072, :], 0)
            plt.plot(xg_obs, '--')
            plt.plot(xg_pred, '-')
            plt.xticks(ticks=np.arange(0, 12, 1), labels=theta_values)

            plt.show()


def fit(dir_output, sim_func, bounds, n_boot):
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    for i in d['cnd'].unique():
        for j in d['rot_dir'].unique():
            p_rec = -1 * np.ones((n_boot, len(bounds)))
            for b in range(n_boot):
                print(i, j, b)
                dd = d[(d['cnd'] == i) & (d['rot_dir'] == j)]
                rot = dd['Appl_Perturb'].values[0:1272]
                subs = dd['sub'].unique()
                boot_subs = np.random.choice(subs,
                                             size=subs.shape[0],
                                             replace=True)
                if n_boot > 1:
                    x_boot_rec = []
                    for k in boot_subs:
                        x_boot_rec.append(d[d['sub'] == k])
                        x_boot = pd.concat(x_boot_rec)

                    x_obs = x_boot.groupby(
                        ['cnd', 'rot_dir', 'Target', 'trial']).mean()
                    x_obs.reset_index(inplace=True)
                else:
                    x_obs = dd.groupby(['cnd', 'rot_dir', 'Target',
                                        'trial']).mean()
                    x_obs.reset_index(inplace=True)

                x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
                x_obs = x_obs.pivot(index="trial",
                                    columns="target_deg",
                                    values="Endpoint_Error")

                x_obs = x_obs.values

                args = (x_obs, rot, sim_func)
                results = differential_evolution(func=obj_func,
                                                 bounds=bounds,
                                                 args=args,
                                                 maxiter=300,
                                                 disp=False,
                                                 polish=False,
                                                 updating="deferred",
                                                 workers=-1)
                p = results["x"]
                p_rec[b, :] = p

                f_name_p = dir_output + "fit_" + str(i) + '_' + j
                with open(f_name_p, "w") as f:
                    np.savetxt(f, p_rec, "%0.4f", ",")

    return p_rec


def obj_func(params, *args):

    x_obs = args[0]
    rot = args[1]
    sim_func = args[2]

    if len(params) > 3:
        alpha_s = params[0]
        beta_s = params[1]
        g_sigma_s = params[2]
        alpha_f = params[3]
        beta_f = params[4]
        g_sigma_f = params[5]
        if (alpha_s >= alpha_f) or (beta_s <= beta_f):
            sse = 10**100
            return sse

    x_pred = simulate(params, *args)

    sse_rec = np.zeros(12)
    for i in range(12):
        sse_rec[i] = (np.nansum((x_obs[:, i] - x_pred[:, i])**2))
        sse = np.nansum(sse_rec)

    return sse


def g_func(theta, theta_mu, sigma):
    if sigma != 0:
        G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
    else:
        G = np.zeros(12)
    return G


def simulate(params, *args):

    x_obs = args[0]
    rot = args[1]
    sim_func = args[2]

    n_simulations = 100
    x = sim_func(params, rot)
    return x


def simulate_one_state(p, rot):
    alpha = p[0]
    beta = p[1]
    g_sigma = p[2]

    num_trials = rot.shape[0]
    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    delta = np.zeros(num_trials)
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


def simulate_two_state(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]

    num_trials = rot.shape[0]
    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    delta = np.zeros(num_trials)
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

        return x.T


def simulate_two_state_ud(p, rot):
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

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    delta = np.zeros(num_trials)
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

    return x.T


dir_output = '../fits/'
bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60), (0, 1), (0, 1),
          (0, 60), (0, 1))
n_boot = 1
p = fit(dir_output, simulate_two_state_ud, bounds, n_boot)

# x_pred = simulate_two_state_ud(p, rot)

# for i in range(12):
#     plt.plot(x[i, :])
# plt.show()

# xg = np.mean(x[:, 1000:1072], 1)
# plt.plot(xg, '-')
# plt.plot(xg, '.')
# plt.xticks(ticks=np.arange(0, 12, 1), labels=theta_values)
# plt.show()
