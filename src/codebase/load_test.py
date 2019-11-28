from metrics import *
import numpy as np

Y = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-64/transfer_epoch_number-600--transfer_model_seed-1--transfer_repr_phase-Test/npz/Y.npz")
A = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-64/transfer_epoch_number-600--transfer_model_seed-1--transfer_repr_phase-Test/npz/A.npz")
Y_hat = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-64/transfer_epoch_number-600--transfer_model_seed-1--transfer_repr_phase-Test/npz/Y_hat.npz")

Y = Y['X']
A = A['X']
Y_hat = Y_hat['X']

print(DP(Y_hat, A))
print(DeltaEO(Y, Y_hat, A))
print(DeltaErr(Y, Y_hat, A))