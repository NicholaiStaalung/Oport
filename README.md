"# Optimal asset portfolios" 

By principles of the Capital Assets Pricing Model (CAPM), the script returns the normalized minimum variance and tangency portfolios. These are the optimal portfolio weights for the two portfolios.

Script is initilized with OptimalPortfolioDiversification(), adding a numpy 2d array with asset returns (Stationary) as est_return_matrix arg and a Risk free return (Often just 0).

Portfolio weights are stored in (self.normalized_min_variance_vector, self.normalized_tangency_vector)

Expected return for the next period and given portfolio is stored in (self.expected_return_of_min_var_portfolio, self.expected_return_of_tangency_portfolio)

Standard deviation (Risk) for the next period and given portfolio is stored in (self.stdOfMinVariancePortfolio, self.stdOfTangencyPortfolio)