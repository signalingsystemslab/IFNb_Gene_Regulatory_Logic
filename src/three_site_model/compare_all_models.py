import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import seaborn as sns
import os

def get_rmsd(params_path):
    df = pd.read_csv(params_path)
    rmsd = df["rmsd"].values
    return rmsd

def get_path(force_t=False,c=False,h1=3,h2=1):
    path = "parameter_scan"
    if force_t:
        path += "_force_t/results"
    else:
        path += "/results"
    h_val = "%d_%d_1" % (h1, h2)
    if h_val != "3_1_1":
        path += "_h_%s" % h_val
    if c:
        path += "_c_scan"
    if force_t:
        path += "/three_site_force_t_best_fits_pars.csv"
    else:
        path += "/three_site_best_fits_pars.csv"
    return path

def plot_rmsd_boxplot(rmsd_df_input, name, figures_dir):
    # sort the dataframe by condition
    rmsd_df = rmsd_df_input.sort_values("Condition")
    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        fig, ax = plt.subplots(figsize=(10,6))
        # col = sns.color_palette("rocket", n_colors=2)[1]
        sns.boxplot(data=rmsd_df, x="Condition", y="rmsd",color="white")

        ax.set_ylabel("RMSD")

        # Remove x-axis labels
        ax.set_xticklabels([])
        # Remove x-axis title
        ax.set_xlabel("")
        # # Remove x-axis ticks
        # ax.set_xticks([])

        # Create a table of h values
        table_data = rmsd_df[[r"$h_1$", r"$h_2$", "c", r"$t_{IRF}$ same"]].drop_duplicates().values.tolist()
        table_data = np.array(table_data).T.astype(str)
        table_data[2] = np.where(table_data[2] == "1", "T", "F")
        table_data[3] = np.where(table_data[3] == "1", "T", "F")
        table = plt.table(cellText=table_data, cellLoc='center', loc='bottom', rowLabels=[r"$h_1$", r"$h_2$", "c",r"$t_{IRF}$ same"], bbox=[0, -0.35, 1, 0.3])

        colors = sns.color_palette("rocket", n_colors=4)
        alpha = 0.5
        colors = [(color[0], color[1], color[2], alpha) for color in colors]
        # Loop through the cells and change their color based on their text
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                cell = table[i, j] 
                if table_data[i][j] == "T":
                    cell.set_facecolor(colors[0])
                elif table_data[i][j] == "3":
                    cell.set_facecolor(colors[1])
                elif table_data[i][j] == "1":
                    cell.set_facecolor(colors[2])
                else:
                    cell.set_facecolor(colors[3])

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.18)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.savefig("%s/%s_rmsd_boxplot_optimized.png" % (figures_dir, name))

def main():
    figures_dir = "parameter_scan"
    force_ts = [False, True]
    cs = [False, True]
    h1s = [3,1]
    h2s = [1,3]
    # conditions = ["%d_%d_%s_%s" % (h1, h2, c, f) for h1 in h1s for h2 in h2s for c in cs for f in force_ts]
    # print(conditions)
    
    args = [(f, c, h1, h2) for h1 in h1s for h2 in h2s for c in cs for f in force_ts]
    paths = [get_path(*arg) for arg in args]
    for p in paths:
        if not os.path.exists(p):
            print("Removing %s" % p)
            index = paths.index(p)
            paths.remove(p)
            args.pop(index)

    conditions = ["%d_%d_%s_%s" % (arg[2], arg[3], arg[1], arg[0]) for arg in args]
    for c in conditions:
        print(c)

    print("There are %d paths" % len(paths))

    with Pool(4) as p:
        rmsd = p.map(get_rmsd, paths)

    rmsd = np.array(rmsd)
    print(rmsd.shape)
    df = pd.DataFrame({r"$t_{IRF}$ same": [arg[0] for arg in args], 
                       "c": [arg[1] for arg in args],
                        r"$h_1$": [arg[2] for arg in args],
                        r"$h_2$": [arg[3] for arg in args]})
    df["Condition"] = df[r"$h_1$"].astype(str) + "_" + df[r"$h_2$"].astype(str) + "_" + df["c"].astype(str) + "_" + df[r"$t_{IRF}$ same"].astype(str)
    df["Condition"] = pd.Categorical(df["Condition"], conditions)
    for i in range(rmsd.shape[1]):
        df["rmsd_%d" % i] = rmsd[:,i]
    # df = df.melt(id_vars=["force_t", "c", "h1", "h2"], var_name="par_set", value_name="rmsd")
    df = df.melt(id_vars=[r"$t_{IRF}$ same", "c", r"$h_1$", r"$h_2$","Condition"], var_name="par_set", value_name="rmsd")
    print(df)
    print("There are %d conditions" % len(conditions))
    plot_rmsd_boxplot(df, "parameter_scan", figures_dir)




if __name__ == "__main__":
    main()