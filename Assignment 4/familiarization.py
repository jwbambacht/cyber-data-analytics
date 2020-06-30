import numpy as np
import matplotlib.pyplot as plt


def bar_plot(dataset1, dataset2, columns, dataset_name1, dataset_name2):

    barWidth = 0.25

    # set height of bar
    bars1 = dataset1.sum()
    bars2 = dataset2.sum()

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    fig, ax = plt.subplots(figsize=(11,4))

    # Make the plot
    plt.bar(r1, bars1, color='blue', width=barWidth, edgecolor='white', label=dataset_name1)
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label=dataset_name2)

    # Add xticks on the middle of the group bars
    plt.xlabel('Feature index',size='16')
    plt.ylabel('Frequency',size='16')
    plt.xticks([r + barWidth for r in range(len(bars1))], columns)
    # plt.title("Top 10 features with largest difference between " + dataset_name1 + " and " + dataset_name2 + " data")
    plt.xticks(rotation=45)
    # Create legend & Show graphic
    plt.legend(loc='upper right',fontsize='x-large')
    plt.show()