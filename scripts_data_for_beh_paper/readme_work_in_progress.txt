curve fit params and what they mean:

r_squared: 1 - (ss_res / ss_tot)
nrmse: 
    rmse = np.sqrt( (ss_res) / len(y_targ) )
    nrmse = rmse / ( np.nanmax(y_targ) - np.nanmin(y_targ) )
AIC: akaike information criterion
subj_center: subjective center (i.e., x-value at which each person arrives at their own midpoint between highest and lowest ratings)
PSE: objective center. x-value at which people cross the 50% rating threshold
sigma: variance parameter from a sigmoid fit
bias_xmin: predicted y at the lowest parameter we measure
bias_xmax: predicted y at the highest parameter we measure
bias_lower: The distance between 0 and the lowest possible y-value in the curve (when the sigmoid flattens out. This is typically <= bias_xmin)
bias_upper: The distance between 1 and the highest possible y-value in the curve (when the sigmoid flattens out on the upper end. This is typically <= bias_xmax)
flipped: boolean saving whether the behavior was normal (flipped=False) or flipped from normal (flipped=True. E.g. fight perception at low charge speed and vice versa).
range_subt = 1-bias_xmin-bias_xmax
bias = bias_xmin/(bias_xmin+bias_xmax)