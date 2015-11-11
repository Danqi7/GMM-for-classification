import sys
import csv
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt



#gmmest estimates parameters of a 1-dimenstional GMM
#Input:
# - X			: N 1-dimensinal data points (a 1-by-N vector)
# - mu_init 	: initial means of K Gaussian components (a 1-by-K vector)
# - sigmasq_init: initial variances of K Gaussian components (a 1-by-K vector)
# - wt_init		: initial weights of K Gaussian components (a 1-by-K vector that sums to 1)
# - its 		: number of iterations
#Output:
# - mu 			: means of Gaussian components (a 1-by-K vector)
# - sigmasq     : variances of Gaussian components (a 1-by-K vector)
# - wt 			: weights of Gaussian components (a 1-by-K vector that sums to 1)
# - L			: log likelihood
def gmmest(X, mu_init, sigmasq_init, wt_init, its):
	mu = np.copy(mu_init)
	sigmasq = np.copy(sigmasq_init)
	wt = np.copy(wt_init)

	new_mu = np.copy(mu)
	new_sigmasq = np.copy(sigmasq)
	new_wt = np.copy(wt)
	
	for iteration in range(its):
		for i in range(len(wt)):
			# r of Gaussian i for each data
			responsibilities = []
			for x in X:
				top = stats.norm(mu[i],sigmasq[i]).pdf(x) * wt[i]
				bottom = 0
				for ii in range(len(wt)):
					bottom += stats.norm(mu[ii],sigmasq[ii]).pdf(x) * wt[ii]
				responsibilities.append(float(top) / float(bottom))

			#Big R for Gaussian i
			responsibility = sum(responsibilities)

			#update weight, mean and variance
			new_wt[i] = float(responsibility) / float(len(X))


			sum_mu = 0
			sum_var = 0
			for index in range(len(X)):
				sum_mu += responsibilities[index] * X[index]
				sum_var += responsibilities[index] * ((X[index]-mu[i])**2)
			new_mu[i] = float(sum_mu) / float(responsibility)		
			new_sigmasq[i] = (float(sum_var) / float(responsibility)) ** 0.5

		#update mu, sigmasq, and weights
		mu = np.copy(new_mu)
		sigmasq = np.copy(new_sigmasq)
		wt = np.copy(new_wt)


	L = result_prob(X, mu, sigmasq, wt)
	return mu, sigmasq, wt, L






#compute probability of observed data points for the input GMM
def result_prob(X, mu, sigmasq, wt):
	L = 0
	
	for x in X:
		point_prob = 0
		for i in range(len(wt)):
			point_prob += (stats.norm(mu[i],sigmasq[i]).pdf(x) * wt[i])

		L += math.log(point_prob)

	#print(L)	
	return L


def build_models1(X):
	#by observing the hist of data
	mu_init = np.array([7, 25])
	sigmasq_init = np.array([5, 3])
	wt_init = np.array([0.7, 0.3])
	its = 20

	print("initial L")
	print(result_prob(X, mu_init, sigmasq_init, wt_init))

	L = []
	mu = np.copy(mu_init)
	sigmasq = np.copy(sigmasq_init)
	wt = np.copy(wt_init)
	
	#firt iteration
	result = gmmest(X, mu_init, sigmasq_init, wt_init, its)
	mu = np.array(result[0][:])
	sigmasq = np.array(result[1][:])
	wt = np.array(result[2][:])
	L.append(result[3])

	#print(result[3])

	#rest of iterations
	for i in range(its-1):
		result = gmmest(X, mu, sigmasq, wt, 1)
		mu = np.array(result[0][:])
		sigmasq = np.array(result[1][:])
		wt = np.array(result[2][:])
		L.append(result[3])
		#print(result[3])

	#print(L)
	return result, L




def build_models2(X):
	mu_init = [-13, -4, 50]
	sigmasq_init = [5, 14, 30] 
	wt_init = [0.2, 0.4, 0.4]
	its = 20

	L = []
	mu = np.copy(mu_init)
	sigmasq = np.copy(sigmasq_init)
	wt = np.copy(wt_init)

	#firt iteration
	result = gmmest(X, mu_init, sigmasq_init, wt_init, its)
	mu = np.array(result[0][:])
	sigmasq = np.array(result[1][:])
	wt = np.array(result[2][:])
	L.append(result[3])

	#rest of iterations
	for i in range(its-1):
		result = gmmest(X, mu, sigmasq, wt, 1)
		mu = np.array(result[0][:])
		sigmasq = np.array(result[1][:])
		wt = np.array(result[2][:])
		L.append(result[3])

	
	return result, L


def main():
	rfile = sys.argv[1]

	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')

	X  = []
	Y = []

	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(int(row[1]))

	X = np.array(X)
	Y = np.array(Y)

	class1 = np.array(X[np.nonzero(Y == 1)[0]])
	class2 = np.array(X[np.nonzero(Y == 2)[0]])

	print("computing...")
	#build GMM for two classes
	model1 = build_models1(class1)
	#model2 = build_models2(class2)

	print("Here are the models!")

	print(model1[0])
	#print(model2[0])


	plt.plot(range(1,20+1), model1[1], 'ro')
	#plt.plot(range(1,20+1), model2[1], 'ro')
	plt.show() 



if __name__ == "__main__":
	main()
