class myMultivariateGaussianClassifier:
	def __init__(self):
		self.epsilon = 0.0
		self.feature_importances_ = None
		self.mu = None
		self.sq_sigma = None

	def _get_Gaussian_proba_estimate(self, X):
		n_features = X.shape[1]
		P_X = np.zeros([X.shape[0], n_features])
		for feat_ix in range(n_features):
			P_X[:,feat_ix] = ((1.0 / (((2 * np.pi) ** 0.5) * (self.sq_sigma[feat_ix] ** 0.5))) *
								np.exp(-((X[:, feat_ix] - self.mu[feat_ix]) ** 2) / (2 * self.sq_sigma[feat_ix])))
		#print "myMultivariateGaussianClassifier._get_Gaussian_proba_estimate: P_X[:5,:] = "
		#pp.pprint(P_X[:5])
		p_X = np.cumprod(P_X, axis=1)[:, n_features-1]
		#print "myMultivariateGaussianClassifier._get_Gaussian_proba_estimate: p_X[:5,:] = "
		#pp.pprint(p_X[:5])
		return p_X

	def fit(self, X, y):
		n_features = X.shape[1]
		self.feature_importances_ = np.zeros(n_features)
		#print "myMultivariateGaussianClassifier.fit: n_samples = {0:,}; n_features = {1:,}".format(n_samples, n_features)
		y_eq_zero_X = X[y == 0]
		y_eq_ones_X = X[y == 1]
		#print "myMultivariateGaussianClassifier.fit: y_eq_zero_n_samples = {0:,}; y_eq_ones_n_samples = {1:,}".format(y_eq_zero_X.shape[0], y_eq_ones_X.shape[0])
		self.mu = np.mean(y_eq_zero_X, axis=0)
		#print "myMultivariateGaussianClassifier.fit: mu = "
		#pp.pprint(self.mu)
		self.sq_sigma = np.var(y_eq_zero_X, axis=0)
		#print "myMultivariateGaussianClassifier.fit: sq_sigma = "
		#pp.pprint(self.sq_sigma)
		y_eq_zero_p_X = self._get_Gaussian_proba_estimate(y_eq_zero_X)
		y_eq_ones_p_X = self._get_Gaussian_proba_estimate(y_eq_ones_X)
		p_X = self._get_Gaussian_proba_estimate(X)
		dist_p_X = np.zeros([len(range(10, 110, 10)), 6])
		for pos, percentile in enumerate(range(10, 110, 10)):
			dist_p_X[pos, 0] = percentile
			dist_p_X[pos, 1] = np.percentile(p_X, percentile)
			dist_p_X[pos, 2] = np.percentile(y_eq_zero_p_X, percentile)
			dist_p_X[pos, 3] = np.percentile(y_eq_ones_p_X, percentile)
			predict_y = p_X < dist_p_X[pos, 3]
			dist_p_X[pos, 4] = sum(predict_y)	# len(predict_y == True) does not work
			dist_p_X[pos, 5] = metrics.f1_score(y, predict_y)

		#print "myMultivariateGaussianClassifier.fit: dist_p_X = "
		#pp.pprint(dist_p_X)

		#print "myMultivariateGaussianClassifier.fit: f1_score: min = {0:0.4f}; max = {1:0.4f}".format(np.min(dist_p_X[:,5]), np.max(dist_p_X[:,5]))
		epsilon_pos = np.argmax(dist_p_X[:,5])
		self.epsilon = dist_p_X[epsilon_pos, 3]
		#print "myMultivariateGaussianClassifier.fit: percentile = {0}; epsilon = {1}".format(dist_p_X[epsilon_pos, 0], self.epsilon)
		return self

	def score(self, X, y):
		predict_y = self._get_Gaussian_proba_estimate(X) < self.epsilon
		score = metrics.accuracy_score(y, predict_y)
		return score

	def predict(self, X):
		p_X = self._get_Gaussian_proba_estimate(X)
		predict_y = p_X < self.epsilon
		return predict_y
