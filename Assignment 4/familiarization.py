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

    # Make the plot
    plt.bar(r1, bars1, color='blue', width=barWidth, edgecolor='white', label=dataset_name1)
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label=dataset_name2)

    # Add xticks on the middle of the group bars
    plt.xlabel('Feature')
    plt.ylabel('Frequency')
    plt.xticks([r + barWidth for r in range(len(bars1))], columns)
    plt.title("Top k features with largest difference between " + dataset_name1 + " and " + dataset_name2 + " data")
    # Create legend & Show graphic
    plt.legend()
    plt.show()