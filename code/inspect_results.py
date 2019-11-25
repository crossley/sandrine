import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
import os as os
import re as re


def simulate_state_space_with_g_func_2_state(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]
    """
    Define experiment
    Simple state-space model predictions:
    N per Block:
    1:     Prebaseline 120
    2: Familiarisation 120
    3:    Baseline_NFB  96
    4:        Baseline 144
    5:        Training 400
    6:  Generalisation  72
    7:      Relearning 100
    8:         Washout 100
    """
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
    xs = np.zeros((12, num_trials))
    xf = np.zeros((12, num_trials))
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        else:
            delta[i] = xs[theta_ind[i], i] - rot[i]

        Gs = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        Gf = g_func(theta_values, theta_values[theta_ind[i]], g_sigma_f)
        if np.isnan(rot[i]):
            xs[:, i + 1] = beta_s * xs[:, i]
            xf[:, i + 1] = beta_f * xf[:, i]
        else:
            xs[:, i + 1] = beta_s * xs[:, i] - alpha_s * delta[i] * Gs
            xf[:, i + 1] = beta_f * xf[:, i] - alpha_f * delta[i] * Gf
            """
    for i in range(len(rot_type)):
            temp = 221 + i
            plt.subplot(temp)
            plt.ylim(-35, 35)
            plt.plot(rot_type[i])
        for j in range(12):
            plt.plot(xs_type[i][:, j])
            """

    # plt.subplot(121)
    # plt.plot(G)

    # plt.subplot(122)
    # plt.plot(rot, '-k')
    # for i in range(12):
    #    plt.plot(xs[i, :])
    return (xs.T + xf.T)


def simulate_state_space_with_g_func(p, rot):
    alpha = p[0]
    beta = p[1]
    g_sigma = p[2]
    """
    Define experiment
    Simple state-space model predictions:
    N per Block:
    1:     Prebaseline 120
    2: Familiarisation 120
    3:    Baseline_NFB  96
    4:        Baseline 144
    5:        Training 400
    6:  Generalisation  72
    7:      Relearning 100
    8:         Washout 100
    """
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
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        G = g_func(theta_values, theta_values[theta_ind[i]], g_sigma)

        # try:
        if np.isnan(rot[i]):
            x[:, i + 1] = beta * x[:, i]
        else:
            x[:, i + 1] = beta * x[:, i] - alpha * delta[i] * G
            # except:
            #     print(x.shape, G.shape, delta.shape)
    """
    for i in range(len(rot_type)):
        temp = 221 + i
        plt.subplot(temp)
        plt.ylim(-35, 35)
        plt.plot(rot_type[i])
        for j in range(12):
            plt.plot(x_type[i][:, j])
    """

    # plt.subplot(121)
    # plt.plot(G)

    # plt.subplot(122)
    # plt.plot(rot, '-k')
    # for i in range(12):
    #    plt.plot(x[i, :])
    return x.T


# Note: Smarter way to handle the for loop
def fit_obj_func_sse(params, *args):
    x_obs = args[0]
    rot = args[1]
    x_pred = simulate_state_space_with_g_func(params, rot)

    sse_rec = np.zeros(12)
    for i in range(12):
        sse_rec[i] = (np.nansum((x_obs[:, i] - x_pred[:, i])**2))
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


def fit_state_space_with_g_func_grp_bootstrap():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)

    n_boot_samp = 10  # NOTE: This should be pretty big (e.g., 1000)
    p_rec = -1 * np.ones((n_boot_samp, 3))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    for b in range(n_boot_samp):
        for i in d['cnd'].unique():
            print(b, i)
            subs = d[d['cnd'] == i]['sub'].unique()
            boot_subs = np.random.choice(subs,
                                         size=subs.shape[0],
                                         replace=True)
            x_boot_rec = []
            for j in boot_subs:
                x_boot_rec.append(d[d['sub'] == j])
                x_boot = pd.concat(x_boot_rec)

            x_obs = x_boot.groupby(['cnd', 'Target', 'trial']).mean()
            x_obs.reset_index(inplace=True)

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
            p_rec[b, :] = p

            f_name_p = "../fits/fit_grp_bootstrap_" + str(i)
            with open(f_name_p, "w") as f:
                np.savetxt(f, p_rec, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    p_rec = np.empty((0, 3))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    x_obs_all = d.groupby(['cnd', 'Target', 'trial']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        x_obs = x_obs_all[x_obs_all['cnd'] == i]
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

        f_name_p = "../fits/fit_grp_" + str(i)
        with open(f_name_p, "w") as f:
            np.savetxt(f, p, "%0.4f", ",")

    return p_rec


def fit_state_space_with_g_func_2_state_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    p_rec = np.empty((0, 6))

    rot = d[d['sub'] == 1]['Appl_Perturb'].values

    x_obs_all = d.groupby(['cnd', 'Target', 'trial']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        x_obs = x_obs_all[x_obs_all['cnd'] == i]
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

        f_name_p = "../fits/fit_2state_grp_" + str(i)
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
    # 1. guess at parameter values
    # 2. generate predictions from guess
    # 3. compute goodness of fit - (simulated data - actual data)
    # 4. store param-values and goodness of fit
    # 5. do this multiple times
    # 6. find the best fitting param values after searching through lots of them

    # os.chdir("D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed")
    # data_dir = "D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed"
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
        x_pred = simulate_state_space_with_g_func(p, rot)

        # t = np.arange(0, x_obs.shape[0])
        # plt.plot(rot, '-k')
        # for i in range(12):
        #     plt.scatter(t, x_obs[:, i])
        #     plt.plot(x_pred[:, i])
        # plt.show()

        c = cm.rainbow(np.linspace(0, 1, 12))

        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = "../fits/fit_" + str(sub[i])
        with open(f_name_p, "w") as f:
            np.savetxt(f, p, "%0.4f", ",")

    # n_sub = []
    # for k in range(length_names):
    #     sub = re.findall(r'\d+', str(sub[k]))
    #     n_sub.append(sub)

    # n_sub = np.array(n_sub)
    # p_rec = np.insert(p_rec, 0, values=n_sub.T, axis=1)
    # np.savetxt("fit_Exp_SH_AllSubs.csv", p_rec, "%0.4f", ",")

    # plt.figure()
    # for i in range(12):
    #     plt.plot(x_obs[:, i], color=c[i])
    #     plt.plot(x_pred[:, i], color=c[i])
    # '''''
    # # Note: This is the dumb way
    # p_rec = []
    # sse_rec = []
    # for a in np.linspace(0.0, 1.0, 3):
    #     for b in np.linspace(0.0, 1.0, 3):
    #         for c in np.linspace(0.0, 60.0, 3):
    #             p = [a, b, c]

    #             sse_rec.append(sse)
    #             p_rec.append(p)

    # sse_rec = np.array(sse_rec)
    # p_rec = np.array(p_rec)

    # best_fitting_p = p_rec[np.where(sse_rec == sse_rec.min()), :]
    # ''' ''
    # '''''
    # Note: Play around with simulation
    # p = restults["x"]
    # x_pred = simulate_state_space_with_g_func()
    # sse = np.nansum((x_obs - x_pred)**2)
    # print(sse)
    # c = cm.rainbow(np.linspace(0, 1, 12))

    # for i in range(12):
    #     plt.plot(x_obs[:, i], color=c[i])
    #     plt.plot(x_pred[:, i], color=c[i])
    # ''' ''
    return p_rec


def inspect_fits_grp():
    f_name = '../fit_input/master_data.csv'

    d = pd.read_csv(f_name)
    rot = d[d['sub'] == 1]['Appl_Perturb'].values
    x_obs_all = d.groupby(['cnd', 'trial', 'Target']).mean()
    x_obs_all.reset_index(inplace=True)

    for i in range(x_obs_all['cnd'].unique().shape[0]):
        x_obs = x_obs_all[x_obs_all['cnd'] == i]
        x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
        x_obs = x_obs.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")

        x_obs = x_obs.values

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        # NOTE: 1 state
        params = np.loadtxt("../fits/fit_grp_" + str(i))
        x_pred = simulate_state_space_with_g_func(params, rot)

        for ii in range(12):
            ax[0].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[0].plot(x_pred)

        # NOTE: 2 state
        params = np.loadtxt("../fits/fit_2state_grp_" + str(i))
        x_pred = simulate_state_space_with_g_func_2_state(params, rot)

        for ii in range(12):
            ax[1].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[1].plot(x_pred)

        plt.savefig('../figures/fit_combined_grp_' + str(i) + ".png")
        plt.close()


def inspect_fits():
    # os.chdir("D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed")
    # data_dir = "D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed"
    # path = "D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed\Figures"
    # try:
    #     os.mkdir(path)
    # except OSError:
    #     print("Creation of the directory %s failed" % path)
    # else:
    #     print("Successfully created the directory %s " % path)

    # '''   for i in range(length_names):
    # x_obs = pd.read_csv(f_names[i], sep=",", encoding="ISO-8859-1")
    # rot = x_obs["Appl_Perturb"].values
    # x_obs = x_obs[["Endpoint_Error", "Target", "Target_Deg", "Trial_Total"]]
    # x_obs = x_obs.pivot(index="Trial_Total",
    #                     columns="Target_Deg",
    #                     values="Endpoint_Error")
    # x_obs = x_obs.values

    # prams = np.loadtxt("fit_" + f_names[i])
    # x_pred = simulate_state_space_with_g_func(prams, rot)

    # fig = plt.figure(figsize=(6,6))
    # plt.plot(rot)
    # plt.plot(x_obs[:,0], alpha=.5, linestyle="--")
    # plt.plot(x_pred[:,0])

    # plt.savefig(path + "\\" + f_names[i] + ".png")
    # plt.close()'''

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

        # fig = plt.figure(figsize=(6, 6))
        # plt.plot(rot, '-k')
        # plt.plot(x_obs[:, 5], alpha=1, linestyle="-")
        for ii in range(12):
            ax[0].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[0].plot(x_pred)
            # plt.show()

        # plt.savefig('../figures/fit_' + str(sub[i]) + ".png")
        # plt.close()

        # NOTE: 2 state
        params = np.loadtxt("../fits/fit_2state" + str(sub[i]))
        x_pred = simulate_state_space_with_g_func_2_state(params, rot)

        # fig = plt.figure(figsize=(6, 6))
        # plt.plot(rot, '-k')
        # plt.plot(x_obs[:, 5], alpha=1, linestyle="-")
        for ii in range(12):
            ax[1].scatter(np.arange(0, x_obs.shape[0]),
                          x_obs[:, ii],
                          alpha=0.2)
            ax[1].plot(x_pred)
            # plt.show()

        # plt.savefig('../figures/fit_2state_' + str(sub[i]) + ".png")
        # plt.close()

        plt.savefig('../figures/fit_combined_' + str(sub[i]) + ".png")
        plt.close()


def inspect_fits_fancy():
    f_name = '../fit_input/master_data.csv'
    d = pd.read_csv(f_name)
    sub = d['sub'].unique()
    length_names = sub.shape[0]

    # x_obs = d[d['sub'] == sub[i]]
    # rot = x_obs["Appl_Perturb"].values
    # x_obs = x_obs[["Endpoint_Error", "Target", "target_deg", "trial"]]
    # x_obs = x_obs.pivot(index="trial",
    #                     columns="target_deg",
    #                     values="Endpoint_Error")
    # x_obs = x_obs.values

    rot = d["Appl_Perturb"].values  # TODO: Something like this?
    x_obs = d.groupby(['cnd', 'trial',
                       'target_deg']).Endpoint_Error.mean().reset_index()
    x_obs_1 = x_obs[x_obs['cnd'] == 0]
    x_obs_2 = x_obs[x_obs['cnd'] == 1]
    x_obs_3 = x_obs[x_obs['cnd'] == 2]

    x_obs_1 = x_obs_1.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")
    x_obs_1 = x_obs_1.values

    x_obs_2 = x_obs_2.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")
    x_obs_2 = x_obs_2.values

    x_obs_3 = x_obs_3.pivot(index="trial",
                            columns="target_deg",
                            values="Endpoint_Error")
    x_obs_3 = x_obs_3.values

    x_pred_1 = []
    x_pred_2 = []
    x_pred_3 = []
    for i in range(0, length_names):
        params = np.loadtxt("../fits/fit_" + str(sub[i]))
        x_pred = simulate_state_space_with_g_func(params, rot)

        if d[d['sub'] == i + 1]['cnd'].unique()[0] == 0:
            x_pred_1.append(x_pred)

        if d[d['sub'] == i + 1]['cnd'].unique()[0] == 1:
            x_pred_2.append(x_pred)

        if d[d['sub'] == i + 1]['cnd'].unique()[0] == 2:
            x_pred_3.append(x_pred)

    x_pred_1_mean = np.zeros(x_pred.shape)
    for i in range(len(x_pred_1)):
        x_pred_1_mean += x_pred_1[i]
        x_pred_1_mean /= len(x_pred_1)

    x_pred_2_mean = np.zeros(x_pred.shape)
    for i in range(len(x_pred_2)):
        x_pred_2_mean += x_pred_2[i]
        x_pred_2_mean /= len(x_pred_2)

    x_pred_3_mean = np.zeros(x_pred.shape)
    for i in range(len(x_pred_3)):
        x_pred_3_mean += x_pred_3[i]
        x_pred_3_mean /= len(x_pred_3)

    # for ii in range(12):
    ii = 5
    plt.scatter(np.arange(0, x_obs_1.shape[0]),
                x_obs_1[:, ii],
                c='r',
                alpha=0.2)
    plt.plot(x_pred_1_mean[:, ii], '-r')
    plt.ylim(-30, 30)

    plt.scatter(np.arange(0, x_obs_2.shape[0]),
                x_obs_2[:, ii],
                c='g',
                alpha=0.2)
    plt.plot(x_pred_2_mean[:, ii], '-g')
    plt.ylim(-30, 30)

    plt.scatter(np.arange(0, x_obs_3.shape[0]),
                x_obs_3[:, ii],
                c='b',
                alpha=0.2)
    plt.plot(x_pred_3_mean[:, ii], '-b')
    plt.ylim(-30, 30)

    plt.show()


def simulate_state_space_with_g_func_usedependent(p, rot):
    alpha = p[0]
    beta = p[1]
    g_sigma = p[2]
    """
    Define experiment
    Simple state-space model predictions:
    N per Block:
    1:     Prebaseline 120
    2: Familiarisation 120
    3:    Baseline_NFB  96
    4:        Baseline 144
    5:        Training 400
    6:  Generalisation  72
    7:      Relearning 100
    8:         Washout 100
    """
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
    for i in range(0, num_trials - 1):
        if i > 1000 and i <= 1072:
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        G = g_func(theta_values, theta_values[theta_ind[i]], g_sigma)

        if np.isnan(rot[i]):
            x[:, i + 1] = beta * x[:, i]
        else:
            x[:, i + 1] = beta * x[:, i] - alpha * delta[i] * G

        ud[:, i] += g_func(theta_values, x[theta_ind[i], i], 30.0)
        x[:, i + 1] += .15 * ud[:, i]

    # for i in range(12):
    #     plt.plot(x[i, :])
    # plt.show()

    xg = np.mean(x[:, 1000:1072], 1)
    plt.plot(xg, '-')
    plt.plot(xg, '.')
    plt.xticks(ticks=np.arange(0, 12, 1), labels=theta_values)
    plt.show()

    # udg = np.mean(ud[:, 1000:1072], 1)
    # udg = np.mean(ud[:, 0:500], 1)
    # plt.plot(udg, '-')
    # plt.plot(udg, '.')
    # plt.xticks(ticks=np.arange(0, 12, 1), labels=theta_values)
    # plt.show()

    return x.T


# p = [0.1, 0.99, 30]
# rot = np.concatenate((np.zeros(120 + 120 + 96), np.random.normal(0, 2, 144),
#                       np.random.normal(15, 2,
#                                        400), np.random.normal(15, 2, 72),
#                       np.random.normal(15, 2, 100), np.zeros(100)))

# simulate_state_space_with_g_func_usedependent(p, rot)

# fit_state_space_with_g_func()
# fit_state_space_with_g_func_2_state()
# fit_state_space_with_g_func_grp()
# fit_state_space_with_g_func_2_state_grp()

start_time = time.time()
fit_state_space_with_g_func_grp_bootstrap()
end_time = time.time()
print("Execution Time = " + str(end_time - start_time))

# fit_state_space_with_g_func_2_state_grp_bootstrap()

# inspect_fits_fancy() # TODO: something is goofy
# inspect_fits()
# inspect_fits_grp()
inspect_fits_boot()
