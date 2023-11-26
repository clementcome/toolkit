import matplotlib.pyplot as plt
import numpy as np
from cc_tk.plot.classification import plot_confusion

confusion_matrix = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                            [0.0, 0.8, 0.1, 0.0, 0.1],
                            [0.0, 0.0, 0.7, 0.2, 0.1],
                            [0.0, 0.0, 0.1, 0.9, 0.0],
                            [0.1, 0.0, 0.0, 0.0, 0.9]])

plot_confusion(confusion_matrix, fmt=".2f")