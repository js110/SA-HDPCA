import argparse

import os

import sys



import numpy as np

import pandas as pd



ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:

    sys.path.insert(0, ROOT)



from src.data import load_gas_sensor, load_uci_har

from src.init_av import random_init

from src.kmeans_dp import dp_kmeans

from src.preprocess import preprocess





def load_dataset(name: str, clip_B: float):

    if name == "har":

        X, _ = load_uci_har("./UCI HAR Dataset")

        k = 6

    elif name == "gas":

        X, _, _, _ = load_gas_sensor("./gas")

        k = 6

    else:

        raise ValueError("Unsupported dataset")

    Z = preprocess(X, clip_B=clip_B)

    return Z, k





def run_once(Z, k, eps_iter, seed, clip_B, T, *, init_c, budget_mode, eps_schedule=None, feedback_params=None):

    rng = np.random.default_rng(seed)

    res = dp_kmeans(

        Z,

        init_centroids=init_c,

        k=k,

        T=T,

        eps_iter=eps_iter,

        clip_B=clip_B,

        rng=rng,

        eps_schedule=eps_schedule,

        budget_mode=budget_mode,

        feedback_params=feedback_params,

        proxy_points=Z,

    )

    return res





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["har", "gas"], required=True)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eps", type=float, default=1.0)

    parser.add_argument("--T", type=int, default=20)

    parser.add_argument("--clip_B", type=float, default=3.0)

    args = parser.parse_args()



    Z, k = load_dataset(args.dataset, clip_B=args.clip_B)

    T = args.T

    eps_tot = float(args.eps)

    eps_const = [eps_tot / T] * T

    init_c = random_init(Z, k, np.random.default_rng(args.seed))



    res_const = run_once(

        Z,

        k,

        eps_tot,

        args.seed,

        args.clip_B,

        T,

        init_c=init_c,

        budget_mode="static",

        eps_schedule=eps_const,

    )

    fb_params = {

        "warmup": 3,

        "beta": 0.8,

        "gamma": 1.0,

        "drift_clip": 2.0,

        "eps_min_ratio": 0.9,

        "eps_max_ratio": 1.4,

        "p": 2.0,

        "adj_clip_low": 0.85,

        "adj_clip_high": 1.15,

    }

    res_fb = run_once(

        Z,

        k,

        eps_tot,

        args.seed,

        args.clip_B,

        T,

        init_c=init_c,

        budget_mode="feedback_v3",

        feedback_params=fb_params,

    )

    eps_fb = [float(h["eps_t"]) for h in res_fb["history"]]



    print("Constant eps schedule:", eps_const)

    print("Feedback eps schedule:", eps_fb)

    print("Max |diff|:", float(np.max(np.abs(np.array(eps_const) - np.array(eps_fb)))))



    hist_const = pd.DataFrame(res_const["history"])

    hist_fb = pd.DataFrame(res_fb["history"])

    print("Noise scale counts (const):", hist_const["noise_scale_counts"].tolist())

    print("Noise scale counts (fb):", hist_fb["noise_scale_counts"].tolist())



    diff_eps = np.max(np.abs(hist_const["eps_t"] - hist_fb["eps_t"]))

    diff_noise = np.max(np.abs(hist_const["noise_scale_sums"] - hist_fb["noise_scale_sums"]))

    centroid_diff = float(np.linalg.norm(res_const["centroids"] - res_fb["centroids"]))



    if diff_eps == 0 and diff_noise == 0 and centroid_diff == 0:

        raise RuntimeError("Feedback schedule not affecting DP noise; check eps_t usage.")

    print("Sanity check passed: schedules/noise differ and affect centroids.")





if __name__ == "__main__":

    main()

