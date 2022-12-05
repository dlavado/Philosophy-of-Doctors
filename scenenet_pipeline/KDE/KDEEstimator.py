
# %%
import numpy as np
import sys
sys.path.insert(0, '..')
import pandas as pd
import laspy as lp
import matplotlib.pyplot as plt
import EDA.EDA_utils as eda
from KDEClassifier import KDEClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, train_test_split


def plot_2D(xdata, ydata, x_label, y_label, title, legend, rgb=[], show=True, save=False):
    """
    Plots the data given in the vectors xdata and ydata\n
    Parameters
    ----------
    xdata - 1darray: 
        single array with the values for the x-axis
    ydata - ndarray:
        array with sets of values for the y-axis;
    legend - str: 
        legend for each plot in order as the data in ydata was given;
    rgb - array: 
        strings that define the color and type of plot for the lines in ydata
    pre-condition: len(ydata) == len(legend)
    """
    for i in range(len(ydata)):
        if len(rgb) > 0:
            plt.plot(xdata, ydata[i], rgb[i], marker="o")
        else:
            plt.plot(xdata, ydata[i], marker="o")
    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig("plots/"+title+".png")





"""
from the calculated bandwidths tested with cross validation, the most adequate was:
    Best bandwith: {'bandwidth': 2.0100000000000002}
    With accuracy = 0.7088121507379281
"""
def find_bandwidth(xyz, classes, hs = None):
    X_train, X_test, Y_train, Y_test = train_test_split(xyz, classes, test_size=0.2)
    print(f"Shape X_train: {X_train.shape}\nShape X_test: {X_test.shape}\nShape Y_train:{Y_train.shape}\nShape Y_test:{Y_test.shape}\n")
    if hs is None:
        hs = np.linspace(1.95, 2.05, 10)
    
    grid = GridSearchCV(estimator = KDEClassifier(), 
                        param_grid= {'bandwidth': hs},
                        cv=5,
                        verbose=2)

    grid.fit(X_train[::100], Y_train[::100])

    scores = grid.cv_results_["mean_test_score"]

    plot_2D(hs, [scores], 'bandwidth', 'accuracy', 'KDE model performance', ['acc'])

    print(f"Best bandwith: {grid.best_params_}\nWith accuracy = {grid.best_score_}")
    return grid.best_params_['bandwidth']


# %%
if __name__ == "__main__":

    tower_files = eda.get_tower_files(False)

    pcd_xyz, classes = eda.las_to_numpy(lp.read(tower_files[0]))
    pcd_tower, _ = eda.select_object(pcd_xyz, classes, [eda.POWER_LINE_SUPPORT_TOWER])
    towers = eda.extract_towers(pcd_tower, visual=False)

    #crop_tower_xyz, crop_tower_classes = eda.crop_two_towers(pcd_xyz, classes, towers[0], towers[-1])
    crop_tower_xyz, crop_tower_classes = eda.crop_tower_radius(pcd_xyz, classes, towers[0])
    down_xyz, down_classes = eda.downsampling(eda.np_to_ply(crop_tower_xyz), crop_tower_classes, samp_per=1)

    eda.visualize_ply([eda.np_to_ply(down_xyz)])

    
    X_train, X_test, y_train, y_test = train_test_split(down_xyz, down_classes, test_size=0.2)

    # %%
    kde = KernelDensity(bandwidth=2.01).fit(X_train)
    log_density = kde.score_samples(X_test)

    # %%
    for i in range(20):
        print(f"{i}: {np.mean(np.exp(log_density[np.where(y_test==i)]))}")

    # %%
    import cloudpickle
    with open(f'{eda.PICKLE_DIR + "kdeEstimator"}.pickle', 'wb') as handle:
        cloudpickle.dump(kde, handle)


# %%
