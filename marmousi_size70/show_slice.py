import numpy as np
import matplotlib.pyplot as plt
from utils import *
from data.show import *

Data_path = 'D:/Data/Marmousi/marmousi_70_70/train_data/seismic/seismic62.npy'
loaded_array = np.load(Data_path)
for i in range(5):
    pain_openfwi_seismic_data(loaded_array[2][i][:][:])
    # y = extract_contours(x)
    # plt.figure(figsize=(10, 4))
    # plt.imshow(y)
    # plt.show()

