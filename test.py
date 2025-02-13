import numpy as np
a = np.array([0,0,0,0])
max_indices = np.where(a == np.max(a))[0]
chosen_index = np.random.choice(max_indices)
print(chosen_index)