from metrics import *
import numpy as np

for epoch in [0, 200, 400, 600, 800]:
	Y = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_01/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Test/npz/Y.npz" %(epoch))
	A = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_01/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Test/npz/A.npz" %(epoch))
	Y_hat = np.load("/Users/andrewli/Documents/laftr2/laftr/experiments/full_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_01/transfer_epoch_number-%d--transfer_model_seed-1--transfer_repr_phase-Test/npz/Y_hat.npz" %(epoch))

	Y = Y['X']
	A = A['X']
	Y_hat = Y_hat['X']

	print("Epoch: ", epoch)
	print("Error: ", 1 - accuracy(Y, Y_hat))
	print("Accuracy: ", accuracy(Y, Y_hat))
	print("inverted DP: ", DP(A, Y_hat))
	print("Log regression coeff: ", LogRegressionCoeff(Y_hat, A))
	print("Bin variance: ", BinVariance(Y_hat, A))
	print("Error + bin variance: ", 1-accuracy(Y, Y_hat) + BinVariance(Y_hat, A))
	print("==========================")