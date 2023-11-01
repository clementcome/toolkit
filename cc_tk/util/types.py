"""Type definitions for cc_tk."""
from typing import Union

import numpy as np
import pandas as pd

ArrayLike1D = Union[np.ndarray, pd.Series]
ArrayLike2D = Union[np.ndarray, pd.DataFrame]
