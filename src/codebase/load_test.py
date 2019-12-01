from metrics import *
import numpy as np

for epoch in [0, 200, 400, 600, 800]:
	Y = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-RegularizedContinuousYGan--model_fair_coeff-0_1/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Valid/npz/Y.npz" %(epoch))
	A = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-RegularizedContinuousYGan--model_fair_coeff-0_1/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Valid/npz/A.npz" %(epoch))
	Y_hat = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-RegularizedContinuousYGan--model_fair_coeff-0_1/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Valid/npz/Y_hat.npz" %(epoch))

	Y = Y['X']
	A = A['X']
	Y_hat = Y_hat['X']

	print("Epoch: ", epoch)
	print(Y_hat)
	print("MSE: ", MSE(Y, Y_hat))
	print("DP: ", DP(Y_hat, A))
	print("Delta_EO: ", DeltaEO(Y, Y_hat, A))
	print("DeltaErr: ", DeltaErr(Y, Y_hat, A))

	print("MSE + DP: ", MSE(Y, Y_hat) + DeltaErr(Y, Y_hat, A))
	print("==========================")