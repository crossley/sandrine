import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
from scipy.stats import logistic
from scipy import stats
import multiprocessing as mp
import os as os


def bootstrap_ci(x, n, alpha):
    x_boot = np.zeros(n)
    for i in range(n):
        x_boot[i] = np.random.choice(x, x.shape, replace=True).mean()
    ci = np.percentile(x_boot, [alpha / 2, 1.0 - alpha / 2])
    return (ci)


def bootstrap_t(x_obs, y_obs, x_samp_dist, y_samp_dist, n):
    d_obs = x_obs - y_obs

    d_boot = np.zeros(n)
    xs = np.random.choice(x_samp_dist, n, replace=True)
    ys = np.random.choice(y_samp_dist, n, replace=True)
    d_boot = xs - ys
    d_boot = d_boot - d_boot.mean()

    p_null = (1 + np.sum(np.abs(d_boot) > np.abs(d_obs))) / (n + 1)
    return (p_null)


# def bootstrap_t(x, y, n):
#     t_obs = stats.ttest_ind(x, y, equal_var=False).statistic

#     xt = x - x.mean() + np.concatenate((x, y)).mean()
#     yt = y - y.mean() + np.concatenate((x, y)).mean()

#     t_boot = np.zeros(n)
#     for i in range(n):
#         xs = np.random.choice(xt, xt.shape, replace=True)
#         ys = np.random.choice(yt, yt.shape, replace=True)
#         t_boot[i] = stats.ttest_ind(xs, ys, equal_var=False).statistic

#     p_null = (1 + np.sum(np.abs(t_boot) > np.abs(t_obs))) / (n + 1)
#     return (p_null)


def inspect_boot():

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    dd = []
    cnd = [0, 1, 2]
    rot_dir = ['cw', 'ccw']
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for j in range(len(rot_dir)):
        for i in cnd:
            d = pd.read_csv('../fits/fit_grp_2state_bootstrap_' + str(i) +
                            '_' + rot_dir[j],
                            header=None)
            d.columns = [
                'alpha_s', 'beta_s', 'sigma_s', 'alpha_f', 'beta_f', 'sigma_f'
            ]
            d['rot_dir'] = rot_dir[j]
            d['cnd'] = i
            dd.append(d)

            alpha_s = d['alpha_s'].values
            beta_s = d['beta_s'].values
            sigma_s = d['sigma_s'].values
            alpha_f = d['alpha_f'].values
            beta_f = d['beta_f'].values
            sigma_f = d['sigma_f'].values

            b = 25
            a = 0.5

            alpha = 0.05

            ci = np.reshape(
                np.percentile(alpha_s, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[0, 0].bar(0 + i * len(cnd) + j,
                         alpha_s.mean(),
                         color=colors[i])
            ax[0, 0].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[0, 0].set_title('alpha_s')

            ci = np.reshape(
                np.percentile(beta_s, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[0, 1].bar(0 + i * len(cnd) + j,
                         beta_s.mean(),
                         color=colors[i])
            ax[0, 1].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[0, 1].set_title('beta_s')

            ci = np.reshape(
                np.percentile(sigma_s, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[0, 2].bar(0 + i * len(cnd) + j,
                         sigma_s.mean(),
                         color=colors[i])
            ax[0, 2].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[0, 2].set_title('sigma_s')

            ci = np.reshape(
                np.percentile(alpha_f, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[1, 0].bar(0 + i * len(cnd) + j,
                         alpha_f.mean(),
                         color=colors[i])
            ax[1, 0].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[1, 0].set_title('alpha_f')

            ci = np.reshape(
                np.percentile(beta_f, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[1, 1].bar(0 + i * len(cnd) + j,
                         beta_f.mean(),
                         color=colors[i])
            ax[1, 1].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[1, 1].set_title('beta_f')

            ci = np.reshape(
                np.percentile(sigma_f, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)]),
                (2, 1))
            ax[1, 2].bar(0 + i * len(cnd) + j,
                         sigma_f.mean(),
                         color=colors[i])
            ax[1, 2].plot([0 + i * len(cnd) + j, 0 + i * len(cnd) + j], ci, 'k')
            ax[1, 2].set_title('sigma_f')

            # ax[0, 0].hist(alpha_s, color=colors[i], density=True, alpha=a)
            # ax[0, 0].set_title('alpha_s')

            # ax[0, 1].hist(beta_s, color=colors[i], density=True, alpha=a)
            # ax[0, 1].set_title('beta_s')

            # ax[0, 2].hist(sigma_s, color=colors[i], density=True, alpha=a)
            # ax[0, 2].set_title('sigma_s')

            # ax[1, 0].hist(alpha_f, color=colors[i], density=True, alpha=a)
            # ax[1, 0].set_title('alpha_f')

            # ax[1, 1].hist(beta_f, color=colors[i], density=True, alpha=a)
            # ax[1, 1].set_title('beta_f')

            # ax[1, 2].hist(sigma_f, color=colors[i], density=True, alpha=a)
            # ax[1, 2].set_title('sigma_f')

    plt.figlegend(['0', '1', '2'],
                  loc='lower center',
                  ncol=5,
                  labelspacing=0.,
                  borderaxespad=1)
    plt.show()

    # NOTE: Report stats
    dd = pd.concat(dd)
    for j in ['alpha_s', 'beta_s', 'sigma_s', 'alpha_f', 'beta_f', 'sigma_f']:
        for i in dd['cnd'].unique():
            xx = dd[(dd['cnd'] == i) & (dd['rot_dir'] == 'cw')][j].values
            yy = dd[(dd['cnd'] == i) & (dd['rot_dir'] == 'ccw')][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, x)

    print('\n')
    print('\n')

    for j in ['alpha_s', 'beta_s', 'sigma_s', 'alpha_f', 'beta_f', 'sigma_f']:
        print('\n')
        for i in dd['rot_dir'].unique():
            print('\n')
            xx = dd[(dd['cnd'] == 0) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 0) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '00', x)

            xx = dd[(dd['cnd'] == 1) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 1) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '11', x)

            xx = dd[(dd['cnd'] == 2) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 2) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '22', x)

            xx = dd[(dd['cnd'] == 0) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 1) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '01', x)

            xx = dd[(dd['cnd'] == 0) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 2) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '02', x)

            xx = dd[(dd['cnd'] == 1) & (dd['rot_dir'] == i)][j].values
            yy = dd[(dd['cnd'] == 2) & (dd['rot_dir'] == i)][j].values
            x = bootstrap_t(xx.mean(), yy.mean(), xx, yy, 1000)
            print(j, i, '12', x)

        # x = bootstrap_t(xx, yy, 1000)
        # x = stats.ks_2samp(xx, yy)
        # x = x[1]


def inspect_fits(sim_func, p=None):
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    colors_obs = ['black'] * 12
    colors_obs[5] = colors[0]

    colors_pred = ['black'] * 12
    colors_pred[5] = colors[1]

    a = 0.05 * np.ones(12)
    a[5] = 1.0

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))

    for i in d['cnd'].unique():
        for j in range(len(d['rot_dir'].unique())):
            rd = d['rot_dir'].unique()[j]
            dd = d[(d['cnd'] == i) & (d['rot_dir'] == rd)]
            rot = dd['Appl_Perturb'].values[0:1272]
            x_obs = dd.groupby(['cnd', 'rot_dir', 'Target', 'trial']).mean()
            x_obs.reset_index(inplace=True)
            x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
            x_obs = x_obs.pivot(index="trial",
                                columns="target_deg",
                                values="Endpoint_Error")

            x_obs = x_obs.values

            p = np.loadtxt('../fits/fit_' + str(i) + '_' + rd, delimiter=',')
            x_pred = sim_func(p, rot)

            num_trials = rot.shape[0]
            theta_values = np.linspace(0.0, 330.0, 12) - 150.0
            theta_train_ind = np.where(theta_values == 0.0)[0][0]
            theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

            title = str('cnd = ') + str(i) + ', rot_dir = ' + rd
            for k in range(12):
                ax[j, i].plot(x_obs[:, k], '-', alpha=a[k], c=colors_obs[k])
                ax[j, i].plot(x_pred[:, k], '-', alpha=a[k], c=colors_pred[k])
                ax[j, i].set_ylim([-20, 20])
                ax[j, i].set_title(title)

            xg_obs = np.nanmean(x_obs[1000:1072, :], 0)
            xg_pred = np.nanmean(x_pred[1000:1072, :], 0)
            ax[j + 2, i].plot(theta_values, xg_obs, '-', c=colors_obs[5])
            ax[j + 2, i].plot(theta_values, xg_pred, '-', c=colors_pred[5])
            ax[j + 2, i].plot(theta_values, xg_obs, '.', c=colors_obs[5])
            ax[j + 2, i].plot(theta_values, xg_pred, '.', c=colors_pred[5])
            ax[j + 2, i].set_title(title)
            ax[j + 2, i].set_ylim([-15, 15])

    plt.tight_layout()
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

                # x_obs = x_obs[["Endpoint_Error", "target_deg", "trial"]]
                # x_obs = x_obs.pivot(index="trial",
                #                     columns="target_deg",
                #                     values="Endpoint_Error")
                # TODO: is bcee an okay column to use?
                # TODO: what's up with the missing data in some cnds (plots)?
                x_obs = x_obs[["bcee", "target_deg", "trial"]]
                x_obs = x_obs.pivot(index="trial",
                                    columns="target_deg",
                                    values="bcee")

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

    # hack to impose parameter constraints
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
        # pick trial indices for cost function
        fit_inds = np.concatenate(
            (np.arange(600, 700, 1), np.arange(1000, 1072,
                                               1), np.arange(1172, 1272, 1)))
        # fit_inds = np.arange(0, 1272, 1)
        sse_rec[i] = (np.nansum((x_obs[fit_inds, i] - x_pred[fit_inds, i])**2))
        sse = np.nansum(sse_rec)

    return sse


def g_func(theta, theta_mu, sigma):
    if sigma != 0:
        G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
    else:
        G = np.zeros(12)
    return G


def ud_func(theta, theta_mu, sigma):
    if sigma != 0:
        G = -2 * (logistic.cdf(theta, theta_mu, sigma) - 0.5)
        # G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        # G[np.where(theta > theta_mu)] = -G[np.where(theta > theta_mu)]
        # G[np.where(theta == theta_mu)] = 0
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

        G_s = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        G_f = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_f)

        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xf[:, i + 1] = beta_f * xf[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * G_s
            xf[:, i + 1] = beta_f * xf[:, i] - alpha_f * delta[i] * G_f

        x[:, i + 1] = xs[:, i + 1] + xf[:, i + 1]

    return x.T


def simulate_one_state_ud(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_ud = p[3]
    beta_ud = p[4]
    sigma_ud = p[5]

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

        G_s = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        G_ud = ud_func(theta_values, x[theta_ind[i], i], sigma_ud)

        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xud[:, i + 1] = beta_ud * xud[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * G_s
            xud[:, i + 1] = beta_ud * xud[:, i] + alpha_ud * G_ud

        x[:, i + 1] = xs[:, i + 1] + xud[:, i + 1]

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

        G_s = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        G_f = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_f)
        G_ud = ud_func(theta_values, x[theta_ind[i], i], sigma_ud)

        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xf[:, i + 1] = beta_f * xf[:, i]
            xud[:, i + 1] = beta_ud * xud[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * G_s
            xf[:, i + 1] = beta_f * xf[:, i] - alpha_f * delta[i] * G_f
            xud[:, i + 1] = beta_ud * xud[:, i] + alpha_ud * G_ud

        x[:, i + 1] = xs[:, i + 1] + xf[:, i + 1] + xud[:, i + 1]

    return x.T


dir_output = '../fits/'

# bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60))
# n_boot = 1
# p = fit(dir_output, simulate_two_state, bounds, n_boot)
# inspect_fits(simulate_two_state) # TODO: modify this func to accommodate boot

# bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60), (0, 1), (0, 1),
#           (0, 60))
# n_boot = 1
# p = fit(dir_output, simulate_two_state_ud, bounds, n_boot)
# inspect_fits(simulate_two_state_ud)

# dir_output = '../fits/'
# bounds = ((0, 1), (0, 1), (0, 60), (0, 1), (0, 1), (0, 60))
# n_boot = 1
# p = fit(dir_output, simulate_one_state_ud, bounds, n_boot)
# inspect_fits(simulate_one_state_ud)

inspect_boot()
