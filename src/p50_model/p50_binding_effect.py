# Demonstrate the effect of p50 on the binding of IRF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing import Pool

def get_IRF_bound_diff(irf, k1, k2, kp, h1, h2):
    # (i^h2 k2 kp)/((1 + i^h1 k1) (1 + i^h2 k2) (1 + i^h2 k2 + kp))
    prob_diff = (irf**h2 * k2 * kp) / ((1 + irf**h1 * k1) * (1 + irf**h2 * k2) * (1 + irf**h2 * k2 + kp))
    return prob_diff

def get_IRF_bound(irf, k1, k2, kp, h1, h2):
    #     (i^h2 k2 + i^(h1 + h2) k1 k2 + 
    #  i^h1 k1 (1 + kp p))/((1 + i^h1 k1) (1 + i^h2 k2 + kp p))
    p=1
    prob_bound = (irf**h2 * k2 + irf**(h1 + h2) * k1 * k2 + irf**h1 * k1 * (1 + kp * p)) / ((1 + irf**h1 * k1) * (1 + irf**h2 * k2 + kp * p))
    return prob_bound

def main():
    num_processes = 4
    irf = np.linspace(0, 1, 101)
    k1 = np.logspace(-3, 3, 3)
    k2 = np.logspace(-3, 3, 3)
    kp = np.logspace(-3, 3, 3)
    h1 = np.arange(1, 4)
    h2 = np.arange(1, 4)
    
    # meshgrid to get all combinations
    K1, K2, KP, H1, H2, IRF = np.meshgrid(k1, k2, kp, h1, h2, irf)

    df = pd.DataFrame({
        "IRF": IRF.flatten(),
        "K1": K1.flatten(),
        "K2": K2.flatten(),
        "KP": KP.flatten(),
        "H1": H1.flatten(),
        "H2": H2.flatten()
    })
    
    with Pool(num_processes) as p:
        bnd = p.starmap(get_IRF_bound_diff, zip(df['IRF'], df['K1'], df['K2'], df['KP'], df['H1'], df['H2']))

    df['p50_binding_effect'] = bnd
    df['par_set'] = np.repeat(np.arange(0, len(k1)*len(k2)*len(kp)*len(h1)*len(h2)), len(irf))
    # print(df.head())

    # Plot
    p = sns.relplot(x='IRF', y='p50_binding_effect', units='par_set', kind='line', data=df, alpha=0.5, estimator=None)
    plt.savefig('p50_binding_effect.png')

    higher_rows = df[df['IRF'].isin([0.25, 1.0])].groupby('par_set').filter(lambda x: x['p50_binding_effect'].iloc[0] - x['p50_binding_effect'].iloc[1]>0.1)
    # print(higher_rows)
    df_filtered = df[df['par_set'].isin(higher_rows['par_set'].unique()) & df['IRF'].isin([0.25, 1.0])]
    df_filtered.to_csv('pars_giving_higher_bound_LPS.csv', index=False)
    
    df_binding_high = df.copy()
    df_binding_high = df_binding_high[df_binding_high['par_set'].isin(higher_rows['par_set'].unique())]
    with Pool(num_processes) as p:
        bnd = p.starmap(get_IRF_bound, zip(df_binding_high['IRF'], df_binding_high['K1'], df_binding_high['K2'], df_binding_high['KP'], df_binding_high['H1'], df_binding_high['H2']))

    df_binding_high['IRF_binding'] = bnd


    p = sns.relplot(x='IRF', y='IRF_binding', units='par_set', kind='line', data=df_binding_high, alpha=0.5, estimator=None)
    plt.savefig('IRF_binding_high_effect_pars.png')

    with Pool(num_processes) as p:
        bnd = p.starmap(get_IRF_bound, zip(df["IRF"], df["K1"], df["K2"], df["KP"], df["H1"], df["H2"]))
    df_binding = df.copy()
    df_binding["IRF_binding"] = bnd

    p = sns.relplot(x="IRF", y="IRF_binding", units="par_set", kind="line", data=df_binding, alpha=0.5, estimator=None)
    plt.savefig("IRF_binding_all_pars.png")

    # p = sns.relplot(x='IRF', y='p50_binding_effect', units='par_set', kind='line', data=df_filtered, alpha=0.5, estimator=None)
    # plt.show()
    # plt.close()

    # Melt all parameters
    higher_rows_filtered = higher_rows[higher_rows['IRF'] == 1.0]
    df_melted = pd.melt(higher_rows_filtered, id_vars=['par_set'], value_vars=['K1', 'K2', 'KP', 'H1', 'H2'], var_name='Parameter', value_name='Value')


    # p = sns.relplot(x="Parameter", y="Value", units='par_set', kind='line', data=df_melted, alpha=0.5, estimator=None)
    # plt.show()


if __name__ == '__main__':
    main()