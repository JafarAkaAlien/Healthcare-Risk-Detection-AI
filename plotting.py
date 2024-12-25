import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
def plot_decision_regions(X, y, classifier, resolution=0.02):
 # setup marker generator and color map
   markers = ('o', 's', '^', 'v', '<')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])
 # plot the decision surface
   x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
   x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
   lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   lab = lab.reshape(xx1.shape)
   plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())
   for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.3,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Cancer Risk: {cl}',
                    edgecolor='black')