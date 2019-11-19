import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
import os as os
import re as re


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
    g = np.zeros((12, 1))

    theta_values = np.linspace(0.0, 330.0, 12) - 150.0
    theta_train_ind = np.where(theta_values == 0.0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    def g_func(theta, theta_mu, sigma):
        if sigma != 0:
            G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
        else:
            G = np.zeros((12, 1))
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
        if np.isnan(rot[i]):
            x[:, i + 1] = beta * x[:, i]
        else:
            x[:, i + 1] = beta * x[:, i] - alpha * delta[i] * G
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


def fit_state_space_with_g_func():
    # 1. guess at parameter values
    # 2. generate predictions from guess
    # 3. compute goodness of fit - (simulated data - actual data)
    # 4. store param-values and goodness of fit
    # 5. do this multiple times
    # 6. find the best fitting param values after searching through lots of them

    # os.chdir("D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed")
    # data_dir = "D:\Studium\Auslandsstudium\TuitionWaver_Master\Masterthesis\Analysis\Exp_Variance\Exp_Variance_MissingsReplaced\SubjData\processed"
    f_name = 'master_data.csv'

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

        f_name_p = "./fits/fit_" + str(sub[i])
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

    f_name = 'master_data.csv'
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

        params = np.loadtxt("./fits/fit_" + str(sub[i]))
        x_pred = simulate_state_space_with_g_func(params, rot)

        fig = plt.figure(figsize=(6, 6))
        # plt.plot(rot, '-k')
        # plt.plot(x_obs[:, 5], alpha=1, linestyle="-")
        for ii in range(12):
            plt.scatter(np.arange(0, x_obs.shape[0]), x_obs[:, ii])
        plt.plot(x_pred)
        # plt.show()

        plt.savefig('./figures/fit_' + str(sub[i]) + ".png")
        plt.close()


def inspect_fits_fancy():
    f_name = 'master_data.csv'
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
        params = np.loadtxt("./fits/fit_" + str(sub[i]))
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
    plt.scatter(np.arange(0, x_obs_1.shape[0]), x_obs_1[:, ii], c='r', alpha = 0.2)
    plt.plot(x_pred_1_mean[:, ii], '-r')
    plt.ylim(-30, 30)

    plt.scatter(np.arange(0, x_obs_2.shape[0]), x_obs_2[:, ii], c='g', alpha = 0.2)
    plt.plot(x_pred_2_mean[:, ii], '-g')
    plt.ylim(-30, 30)

    plt.scatter(np.arange(0, x_obs_3.shape[0]), x_obs_3[:, ii], c='b', alpha = 0.2)
    plt.plot(x_pred_3_mean[:, ii], '-b')
    plt.ylim(-30, 30)

    plt.show()


# simulate_state_space_with_g_func()
fit_state_space_with_g_func()
inspect_fits_fancy()
inspect_fits()
