import numpy as np

def covarMatrix(x):
    """Helper to generate covar matrix"""
    return np.matrix(x - np.mean(x, axis=0)[np.newaxis, :]).T * np.matrix(x - np.mean(x, axis=0)[np.newaxis, :])

class OptimalPortfolioDiversification():
    """Finding the optimal portfolio weights by principals of CAPM"""
    def __init__(self,
                 est_return_matrix,
                 risk_free_return,
                lagrange=False):
        """Initializor"""

        try:
            """Analysing the return matrix and assign"""
            self.est_return_matrix = est_return_matrix
            self.risk_free_return = risk_free_return
            self.number_obs = len(self.est_return_matrix) #number of observations in each category
            self.number_groups = len(est_return_matrix[0])

            """generating the avg return matrices"""
            self.avgReturnVector = np.sum(self.est_return_matrix, axis=0) / self.number_obs
            self.avgReturnDeviationFromRiskfree = np.matrix((self.avgReturnVector- self.risk_free_return)[np.newaxis, :])
            self.avgReturnMatrix = np.matrix(self.avgReturnVector)
        except:
            raise ValueError('Issue with your return matrices')

        """Just adding a ones vector"""
        self.ones_matrix = np.matrix(np.ones(self.number_groups))


        """Trying the unique solution. If non-invertability, the lagrange will be automatically called"""
        self.varCovar()
        if self.inverseTrue and not lagrange:
            try:
                self.minVarPortfolio()
            except:
                raise ValueError('Cant create the min variance portfolio')
            try:
                self.tangentPortfolio()
            except:
                raise ValueError('Cant create the tangent portfolio')
        elif not self.inverseTrue or lagrange:
            #call te lagrange function
            self.lagVarCovar()
            self.minVarLagrange()
            self.tangentPortfolioLagrange()





    def varCovar(self):
        """calculating the variance-covariance matrix based on the return matrix"""

        self.covariance_matrix = covarMatrix(self.est_return_matrix)
        self.cov_matrix_determinant = np.linalg.det(self.covariance_matrix)
        self.inverseTrue = False
        try:
            self.inv_cov_matrix = np.linalg.inv(self.covariance_matrix)
            self.inverseTrue = True
        except:
            print "ERROR: Determinant equals zero, matrix not invertable. Proceeding with Lagrange instead and, you should test for invertability"
            try:
                self.lagVarCovar()
            except:
                raise ValueError('Cant call lagrange function from the failed unique solution')


    def minVarPortfolio(self):
        """Calculating the minimum variance portfolio weights, expected return and volatility(std. deviation), based on the the variance-covariance matrix.
        Only applicable if the variaince-covariance matrix is invertable. If not, move on to lagrange and test for applicability"""

        try:
            self.inv_cov_matrix = self.inv_cov_matrix
        except:
            raise ValueError("Inverse Variance-Covariance matrix not assigned. Call var_covar function")
        try:
            try:
                _min_variance_vector = self.inv_cov_matrix * self.ones_matrix
            except:
                _min_variance_vector = self.inv_cov_matrix * self.ones_matrix.T
            _sum_of_min_var_vector = np.sum(_min_variance_vector)
            self.normalized_min_variance_vector = _min_variance_vector / _sum_of_min_var_vector #portfolio weights for minimum variance portfolio
            try:
                self.expected_return_of_min_var_portfolio = self.avgReturnMatrix * self.normalized_min_variance_vector
                self.stdOfMinVariancePortfolio = np.array(((self.covariance_matrix * self.normalized_min_variance_vector).T * np.array(self.normalized_min_variance_vector)))**(0.5)
            except:
                raise ValueError('Cant assign the exp return and std of the min varaince portfolio')
        except:
             raise ValueError('Some matrices have been calucaled wrongly. Perhaps its the covar. Check shapes, oth sides shoud be uqula to number of varibles')


    def tangentPortfolio(self):
        """Calculating the effecient/tangent portfolio weights, expected return and volatility(std. deviation), based on the the variance-covariance matrix.
        Only applicable if the variaince-covariance matrix is invertable. If not, move on to lagrange and test for applicability"""

        try:
            self.inv_cov_matrix = self.inv_cov_matrix
        except:
            raise ValueError("Inverse Variance-Covariance matrix not assigned. Call var_covar function")
        try:
            try:
                _tangency_vector = self.inv_cov_matrix * self.avgReturnDeviationFromRiskfree
            except:
                _tangency_vector = self.inv_cov_matrix * self.avgReturnDeviationFromRiskfree.T
            _sum_of_tangency_vector = np.sum(_tangency_vector)
            self.normalized_tangency_vector = _tangency_vector / _sum_of_tangency_vector #Portfolio weights for tangency portfolio
        except:
            raise ValueError('Cant assign the tangency vector')

        try:
            self.expected_return_of_tangency_portfolio = self.avgReturnMatrix * self.normalized_tangency_vector

            self.stdOfTangencyPortfolio = np.array(((self.covariance_matrix * self.normalized_tangency_vector).T * np.array(self.normalized_tangency_vector)))**(0.5)
        except:
            raise ValueError('Cant assign the exp return and std of the tangency portfolio')


    def lagVarCovar(self):
        """Function for prepping the matrices for other functions"""
        self.covariance_matrix = covarMatrix(self.est_return_matrix)
        lag_weight_matrix = np.zeros(self.number_groups)
        lag_weight_matrix = np.hstack((lag_weight_matrix, np.array([1]))) #One is the lambda (remember lagrange function)
        self.lag_weight_matrix = np.matrix(lag_weight_matrix).T


    def minVarLagrange(self):
        """Calculating the minimum variance portfolio by lagrange method"""

        try:
            self.covariance_matrix 
        except:
            raise ValueError("Variance-Covariance matrix not assigned. Call lag_var_covar function")

        try:
            lag_covariance_matrix = self.covariance_matrix * 2
            lag_covariance_matrix = np.hstack((lag_covariance_matrix, np.matrix(np.ones(self.number_groups)).T))
            new_row = np.ones(self.number_groups)
            new_row = np.hstack((new_row, np.array([0])))
            self.lag_covariance_matrix = np.vstack((lag_covariance_matrix, new_row))
        except:
            raise ValueError('Assignen the lagrange covariance matrix for Min Var is a problem')

        self.lag_cov_matrix_determinant = np.linalg.det(self.lag_covariance_matrix)
        if self.lag_cov_matrix_determinant == 0:
            raise ValueError("ERROR: Determinant equals zero, matrix not invertable.")
        else:
            self.lagInvCovarianceMatrix = np.linalg.inv(self.lag_covariance_matrix)

        self.lagNormMinVarVector = self.lagInvCovarianceMatrix * self.lag_weight_matrix #Portfolio weights

        self.lagExpReturnMinVar = self.avgReturnMatrix * self.lagNormMinVarVector[:self.number_groups]
        self.lagStdOfMinVar = np.array(((self.covariance_matrix * self.lagNormMinVarVector[:self.number_groups]).T * np.array(self.lagNormMinVarVector[:self.number_groups])))**(0.5)




    def tangentPortfolioLagrange(self):
        """Calculating the minimum variance portfolio by lagrange method"""

        try:
            self.covariance_matrix = self.covariance_matrix
        except:
            raise ValueError("Variance-Covariance matrix not assigned. Call lag_var_covar function")

        lagCovarTangMatrix = np.hstack((self.covariance_matrix, -(self.avgReturnDeviationFromRiskfree.T)))
        new_row = np.ones(self.number_groups)
        new_row = np.hstack((new_row, np.array([0])))
        self.lagCovarTangMatrix = np.vstack((lagCovarTangMatrix, new_row))
        self.lag_cov_matrix_eff_determinant = np.linalg.det(self.lagCovarTangMatrix)
        if self.lag_cov_matrix_eff_determinant == 0:
            raise ValueError("ERROR: Determinant equals zero, matrix not invertable.")
        else:
             self.lagInvCovarTangMatrix = np.linalg.inv(self.lagCovarTangMatrix)
        self.lagNormTangVector = self.lagInvCovarTangMatrix * self.lag_weight_matrix

        self.lagExpReturnTang = self.avgReturnMatrix * self.lagNormTangVector[:self.number_groups]
        self.lagStdOfTang = np.array(((self.covariance_matrix * self.lagNormTangVector[:self.number_groups]).T * np.array(self.lagNormTangVector[:self.number_groups])))**(0.5)
