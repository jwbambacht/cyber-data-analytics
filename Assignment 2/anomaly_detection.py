import familiarization as fam
import anomaly_detection as dtc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def compute_lof(data, signals, max_neighbours):  
    lof_curvs = [ [] for _ in range(len(signals)) ]     
    neighbour_array = range(1,max_neighbours)
    count_signal = 0
    lof_data = [ [] for _ in range(len(data['P_J306'])) ]
    for signal in signals:
        pref_lof = []
        for i in range(len(data[signal])):
            lof_data[i] = [data[signal][i]]
        for n in neighbour_array:
            outliers_detected = 0
            clf = LocalOutlierFactor(n_neighbors=n)
            clf.fit_predict(lof_data)
            res = clf.negative_outlier_factor_
            for r in res:
                # TODO: Define threshold of outlier detection             
                if r < -1.5:
                    outliers_detected += 1
            pref_lof.append(outliers_detected)

        lof_curvs[count_signal] = pref_lof
        count_signal += 1
    return lof_curvs

def plot_lof(lof_curvs, signals, max_neighbours, title):
    f,ax = plt.subplots(figsize=(22,3))
    neighbour_array = range(1,max_neighbours)
    x = np.array(neighbour_array)
    count = 0
    for elem in lof_curvs:
        y = np.array(elem)
        # convert to pandas dataframe
        d = {'Number of neighbours used': x, 'Outliers detected': y}
        pd_data = pd.DataFrame(d)
        # plot using lineplot
        sns.set(style='darkgrid')
        ax.set_xticks(neighbour_array)
        ax.set_yscale('log')
        ax.set_title(title)
        g = sns.lineplot(x="Number of neighbours used", y="Outliers detected", data=pd_data, label=signals[count])
        count += 1