# Generate synthetic data based on p50 training data
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

def generate_synthetic_dataset(training_data, err, seed, number, unscaled=False, clevel=0.95):
    # generate synthetic data based on training data and standard error
    lps_wt_loc = training_data.loc[(training_data["Stimulus"]=="LPS") & (training_data["Genotype"]=="WT")].index[0]
    lps_irf_2ko = training_data.loc[(training_data["Stimulus"]=="LPS") & (training_data["Genotype"]=="irf3irf7KO")].index[0]
    pic_irf_2ko = training_data.loc[(training_data["Stimulus"]=="polyIC") & (training_data["Genotype"]=="irf3irf7KO")].index[0]
    pic_irf_3ko = training_data.loc[(training_data["Stimulus"]=="polyIC") & (training_data["Genotype"]=="irf3irf5irf7KO")].index[0]

    IRF_vals = training_data["IRF"].values
    NFkB_vals = training_data["NFkB"].values
    IFNb_vals = training_data["IFNb"].values

    LPS_WT_val = 0
    LPS_IRF_2KO_val = 1
    PIC_IRF_2KO_val = 0
    PIC_IRF_3KO_val = 1
    
    std_dev = err

    
    i=0
    
    if number % 10 == 0:
        print("Generating synthetic data set %d" % number)
    if unscaled:
        # require that the synthetic data has LPS WT > LPS IRF 2KO (IRF) and PIC IRF 2KO > PIC IRF 3KO (IRF)
        while LPS_WT_val < LPS_IRF_2KO_val or PIC_IRF_2KO_val < PIC_IRF_3KO_val:
            rng = np.random.default_rng(seed*100 + i)
            IRF_synthetic = rng.normal(IRF_vals, std_dev)
            IRF_synthetic = np.clip(IRF_synthetic, 0, 1)
            IRF_synthetic = np.round(IRF_synthetic, 3)
            IRF_synthetic[IRF_vals == 0] = 0

            NFkB_synthetic = rng.normal(NFkB_vals, std_dev)
            NFkB_synthetic = np.clip(NFkB_synthetic, 0, 1)
            NFkB_synthetic = np.round(NFkB_synthetic, 3)
            NFkB_synthetic[NFkB_vals == 0] = 0

            IFNb_synthetic =  IFNb_vals

            LPS_WT_val = IRF_synthetic[lps_wt_loc] 
            LPS_IRF_2KO_val = IRF_synthetic[lps_irf_2ko]
            PIC_IRF_2KO_val = IRF_synthetic[pic_irf_2ko]
            PIC_IRF_3KO_val = IRF_synthetic[pic_irf_3ko]
            i += 1
            if i == 100:
                print("Failed to generate synthetic data set with IRF and NFkB values that match the training data")
                sys.exit(1)
    else:
        fraction = err
        z_score = stats.norm.ppf((1 + clevel)/2) # number of standard deviations from the mean that corresponds to the confidence level
        IRF_std_devs = fraction * IRF_vals / z_score # width = fraction * mean *2, stdev = width/(2*z_score)
        NFkB_std_devs = fraction * NFkB_vals / z_score

        while LPS_WT_val < LPS_IRF_2KO_val or PIC_IRF_2KO_val < PIC_IRF_3KO_val:
            rng = np.random.default_rng(seed*100 + i)
            IRF_synthetic = rng.normal(IRF_vals, IRF_std_devs)
            IRF_synthetic = np.clip(IRF_synthetic, 0, 1)
            IRF_synthetic = np.round(IRF_synthetic, 3)

            NFkB_synthetic = rng.normal(NFkB_vals, NFkB_std_devs)
            NFkB_synthetic = np.clip(NFkB_synthetic, 0, 1)
            NFkB_synthetic = np.round(NFkB_synthetic, 3)

            # verify all IRF_synthetic and NFkB_synthetic values are real numbers
            if np.isnan(IRF_synthetic).any() or np.isnan(NFkB_synthetic).any():
                print("At least one synthetic value is NaN. Position of a NaN value in IRF_synthetic: ", np.where(np.isnan(IRF_synthetic)))

            # if any(IRF_synthetic > IRF_vals + IRF_vals*(fraction+0.01)) or any(IRF_synthetic < IRF_vals - IRF_vals*(fraction+0.01)):
            #     print("IRF_synthetic values are farther than expected from IRF_vals:")
            #     loc = np.where((IRF_synthetic > IRF_vals + IRF_vals*(fraction+0.01)) | (IRF_synthetic < IRF_vals - IRF_vals*(fraction+0.01)))
            #     print("True value: %.2f, Synthetic value: %.2f. Fraction: %.2f" % (IRF_vals[loc], IRF_synthetic[loc], fraction))

            # if any(NFkB_synthetic > NFkB_vals + NFkB_vals*(fraction+0.01)) or any(NFkB_synthetic < NFkB_vals - NFkB_vals*(fraction+0.01)):
            #     print("NFkB_synthetic values are farther than expected from NFkB_vals:")
            #     loc = np.where((NFkB_synthetic > NFkB_vals + NFkB_vals*(fraction+0.01)) | (NFkB_synthetic < NFkB_vals - NFkB_vals*(fraction+0.01)))
            #     print("True value: %.2f, Synthetic value: %.2f Fraction: %.2f" % (NFkB_vals[loc], NFkB_synthetic[loc], fraction))

            IFNb_synthetic =  IFNb_vals

            LPS_WT_val = IRF_synthetic[lps_wt_loc] 
            LPS_IRF_2KO_val = IRF_synthetic[lps_irf_2ko]
            PIC_IRF_2KO_val = IRF_synthetic[pic_irf_2ko]
            PIC_IRF_3KO_val = IRF_synthetic[pic_irf_3ko]
            i += 1
            if i == 100:
                print("Failed to generate synthetic data set with IRF and NFkB values that match the training data")
                sys.exit(1)

    other_cols = training_data.copy().drop(columns=["IRF", "NFkB", "IFNb", "Dataset"])
    dataset_name = "synthetic_%d" % (number)
    synthetic_data = pd.DataFrame({"IRF": IRF_synthetic, "NFkB": NFkB_synthetic, "IFNb": IFNb_synthetic,
                                   **other_cols, "Dataset": dataset_name})
    return synthetic_data

def generate_synthetic_data(training_data, std_err, num_datasets, original_seed):
    synthetic_data = training_data.copy()
    seed = original_seed
    for i in range(num_datasets):
        seed += 1
        synthetic_data = pd.concat([synthetic_data, generate_synthetic_dataset(training_data, std_err, seed, i, unscaled=False, clevel=0.99)])
    return synthetic_data

def main():
    training_data = pd.read_csv("../data/p50_training_data.csv")
    dataset_name = "original"
    training_data["Dataset"] = dataset_name

    # Generate synthetic data
    num_datasets = 99
    seed = 5

    errors_tested = [1, 2, 5, 10, 20, 30, 40]
    # errors_tested = np.arange(1, 11, 1)
    # errors_tested = np.concatenate((errors_tested, [0.1, 0.5]))
    for e in errors_tested:
        std_err = e/100
        print("Generating synthetic data with standard error %.1f percent" % e, flush=True)
        synthetic_data = generate_synthetic_data(training_data, std_err, num_datasets, seed)
        # print(synthetic_data.head(len(training_data)))
        # print(synthetic_data.tail(len(training_data)))

        # Print range of IRF and NFkB values for each genotype and stimulus
        syn_data_grouped = synthetic_data.groupby(["Genotype", "Stimulus"])
        print("Fraction = %.2f" % std_err)
        for name, group in syn_data_grouped:
            training_row = training_data.loc[(training_data["Genotype"]==name[0]) & (training_data["Stimulus"]==name[1])]
            print("%s IRF: (%.2f) %.3f +/- %.3f, NFkB: (%.2f) %.3f +/- %.3f" % (name, training_row["IRF"].values[0], group["IRF"].mean(), 
                                                                                group["IRF"].max()-group["IRF"].min(), 
                                                                                training_row["NFkB"].values[0], group["NFkB"].mean(),
                                                                                group["NFkB"].max()-group["NFkB"].min()))
        summary = syn_data_grouped.agg({"IRF": ["mean", "std"], "NFkB": ["mean", "std"]})
        summary.to_csv("results/synthetic_data_summary_e%.1fpct.csv" % (std_err*100))
        

        synthetic_data.to_csv("../data/p50_training_data_plus_synthetic_e%.1fpct.csv" % (std_err*100), index=False)

        # Plot all data where Genotype != p50KO, x axis is IRF, y axis is NFkB, color is IFNb
        print("Plotting scatterplot for standard error %.2f" % std_err, flush=True)
        pal = sns.cubehelix_palette(as_cmap=True, start=.6, rot=-.4, dark=0.2, light=0.9, hue=0.6)
        fig, ax = plt.subplots(figsize=(5,5))
        p = sns.scatterplot(data=synthetic_data[synthetic_data["Genotype"]!="p50KO"], x="NFkB", y="IRF", hue="IFNb", ax=ax,
            palette = pal)
        p.set_xlim(0, 1)
        p.set_ylim(0, 1)
        p.set_xlabel(r"NF$\kappa$B")
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), frameon=False, loc="center left", title=r"IFN$\beta$")
        # sns.despine()
        p.set_title("%.1f percent" % (std_err*100))
        plt.savefig("IRF_NFkB_IFNb_scatterplot_e%.1fpct.png" % (std_err*100), dpi=300, bbox_inches = "tight")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(5,5))
        p = sns.scatterplot(data=synthetic_data[synthetic_data["Genotype"]=="p50KO"], x="NFkB", y="IRF", hue="IFNb", ax=ax,
            palette = pal)
        p.set_xlim(0, 1)
        p.set_ylim(0, 1)
        p.set_xlabel(r"NF$\kappa$B")
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), frameon=False, loc="center left", title=r"IFN$\beta$")
        # sns.despine()
        p.set_title("%.1f percent" % (std_err*100))
        plt.savefig("IRF_NFkB_IFNb_scatterplot_p50KO_e%.1fpct.png" % (std_err*100), dpi=300, bbox_inches = "tight")
        plt.close()

    np.savetxt("error_percentages_tested.txt", errors_tested)
        
        


if __name__ == "__main__":
    main()