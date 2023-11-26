from cc_tk.plot.classification import plot_confusion
import numpy as np
confusion_matrix = np.array([[15, 3, 1], [2, 10, 0], [0, 0, 5]])
plot_confusion(confusion_matrix, fmt=".2f")
