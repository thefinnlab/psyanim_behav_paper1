def sigmoid(x, sigma, center):#, flip_sigmoid=False):
    """
    Sigmoid function.

    Parameters:
    - x (array-like): Input values.
    - sigma (float): Controls the steepness of the curve, where sigma = 1/slope.
    - center (float): The midpoint of the sigmoid curve.

    Returns:
    - array-like: Sigmoid function values.
    """
    import numpy as np

    # if flip_sigmoid == False:
    return 1 / (1 + np.exp(-((x - center) / sigma)))
    # else:
    #     return 1 - (1 / (1 + np.exp(-((x - center) / sigma))))

def linear(x, slope, intercept):#, flip_sigmoid=False):
    """
    Sigmoid function.

    Parameters:
    - x (array-like): Input values.
    - sigma (float): Controls the steepness of the curve, where sigma = 1/slope.
    - center (float): The midpoint of the sigmoid curve.

    Returns:
    - array-like: Sigmoid function values.
    """
    import numpy as np

    # if flip_sigmoid == False:
    # return 1 / (1 + np.exp(-((x - center) / sigma)))
    return intercept + slope * x
    # else:
    #     return 1 - (1 / (1 + np.exp(-((x - center) / sigma))))

def sum_of_squared_error(params, x, y):#,flip_sigmoid):
    """
    Cost function using sum-of-squared-error.

    Parameters:
    - params (tuple): Parameters of the sigmoid function.
    - x (array-like): Input values.
    - y (array-like): Target values.

    Returns:
    - float: Sum of squared errors.
    """
    
    import numpy as np

    sigma, center = params
    predictions = sigmoid(x, sigma, center)#,flip_sigmoid)
    return np.sum((predictions - y)**2)

def sum_of_squared_error_linear(params, x, y):#,flip_sigmoid):
    """
    Cost function using sum-of-squared-error.

    Parameters:
    - params (tuple): Parameters of the sigmoid function.
    - x (array-like): Input values.
    - y (array-like): Target values.

    Returns:
    - float: Sum of squared errors.
    """
    import numpy as np

    slope, intercept = params
    predictions = linear(x, slope, intercept)#,flip_sigmoid)
    # sigma, center = params
    # predictions = sigmoid(x, sigma, center)#,flip_sigmoid)
    return np.sum((predictions - y)**2)


def get_aic(params,x,y):

    import numpy as np

    # slope, intercept = params
    sigma, center = params
    predictions = sigmoid(x, sigma, center)
    ss_err = np.sum((predictions - y)**2)
    n = len(x)
    sigma = np.std(predictions - y)**2
    # logL = -(n/2)*np.log(2*np.pi) -(n/2)*np.log(sigma**2) - (ss_err/(2*(sigma**2))) # full v
    logL = -(n/2)*np.log(ss_err/n) # However, this can be simplified for AIC purposes to:
    
    k = 2 # params
    aic = 2*k - 2*logL
    
    return aic

def get_aic_linear(params,x,y):

    import numpy as np
    slope, intercept = params
    predictions = linear(x, slope, intercept)
    ss_err = np.sum((predictions - y)**2)
    n = len(x)
    sigma = np.std(predictions - y)**2
    # logL = -(n/2)*np.log(2*np.pi) -(n/2)*np.log(sigma**2) - (ss_err/(2*(sigma**2))) # full v
    logL = -(n/2)*np.log(ss_err/n) # However, this can be simplified for AIC purposes to:
    
    k = 2 # params
    aic = 2*k - 2*logL
    
    return aic


def normalized_error(y_targ, y_pred):
    """
    Calculate the normalized error.
    
    Parameters:
    - y_targ (list of float): Observed values.
    - y_pred (list of float): Predicted values corresponding to the observed values.
    
    Returns:
    - float: The normalized error.
    """
    import numpy as np


    # Calculate the residual sum of squares
    ss_res = np.sum([(y_targ_i - y_pred_i) ** 2 for y_targ_i, y_pred_i in zip(y_targ, y_pred)])
    
    # Calculate the RMSE
    rmse = np.sqrt( (ss_res) / len(y_targ) )

    # Calculate the normalized error
    nrmse = rmse / ( np.nanmax(y_targ) - np.nanmin(y_targ) )
    
    return nrmse


def prediction_r_squared(y_targ, y_pred):
    """
    Calculate the R-squared, coefficient of determination.
    
    Parameters:
    - y_targ (list of float): Observed values.
    - y_pred (list of float): Predicted values corresponding to the observed values.
    
    Returns:
    - float: The R-squared value.
    """
    import numpy as np


    # Calculate the residual sum of squares
    ss_res = np.sum([(y_targ_i - y_pred_i) ** 2 for y_targ_i, y_pred_i in zip(y_targ, y_pred)])
    
    # Calculate the total sum of squares
    ss_tot = np.sum([(y_targ_i - np.nanmean(y_targ)) ** 2 for y_targ_i in y_targ])
    
    # Calculate the R-squared value
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared
    

def r_squared(y_targ, y_pred):
    """
    Calculate the R-squared, coefficient of determination.
    
    Parameters:
    - y_targ (list of float): Observed values.
    - y_pred (list of float): Predicted values corresponding to the observed values.
    
    Returns:
    - float: The R-squared value.
    """
    import numpy as np

    # Calculate the residual sum of squares
    res = y_targ - y_pred
    ss_res = np.sum([(res_i - np.nanmean(res)) ** 2 for res_i in res])
    
    # Calculate the total sum of squares
    ss_tot = np.sum([(y_targ_i - np.nanmean(y_targ)) ** 2 for y_targ_i in y_targ])
    
    # Calculate the R-squared value
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def fit_linear(x_pred, y_targ, rescale=False, padding=False, n_rep=10, param_bounds=None, param_details=None):
    """
    Fit the sigmoid function with multiple random initializations.

    Parameters:
    - x_pred (array-like): Input values for prediction.
    - y_targ (array-like): Target values.
    - rescale (bool, optional): Scale x_pred and y_targ to [0,1] before fitting. Default is False.
    - padding (bool, optional): Add padding to x_pred and y_targ indicating that the plateaus are reached. Default is False.
    - n_rep (int, optional): Number of random initializations. Default is 10.
    - param_bounds (list of tuples, optional): Optional bounds for parameters. Default is None.
    - param_details (str, optional): File name to save the fitting progress. Default is None.

    Returns:
    - tuple: Best-fitting parameters and goodness of fit
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    best_cost = np.inf

    # Create a DataFrame to save the fitting progress
    # params_name = ['iteration', 'init_sigma', 'init_center', 'est_sigma', 'est_center', 'ss_res', 'nrmse', 'r_squared']
    params_name = ['iteration', 'init_slope', 'init_intercept', 'est_intercept','est_slope', 'ss_res', 'nrmse', 'r_squared','aic']
    params = pd.DataFrame(np.zeros((n_rep, len(params_name))), columns=params_name)

    if rescale:
        x = ( x_pred - np.nanmin(x_pred) ) / ( np.nanmax(x_pred) - np.nanmin(x_pred) )
        y = ( y_targ - np.nanmin(y_targ) ) / ( np.nanmax(y_targ) - np.nanmin(y_targ) )
        bias_lower, bias_upper = np.nanmin(y_targ), ( 1 - np.nanmax(y_targ) )
        # bias_lower, bias_upper = y_targ[0], ( 1 - y_targ[-1] )
    else:
        x = x_pred.copy()
        y = y_targ.copy()
        bias_lower, bias_upper = 0, 0

    xfit, yfit = x.copy(), y.copy()

    for i in range(n_rep):
        params['iteration'][i] = i+1

        # Random initialization within optional bounds
        if param_bounds:
            initial_params = [np.random.uniform(low=lb, high=ub) for lb, ub in param_bounds]
        else:
            initial_params = [np.random.uniform(0, 10), np.random.uniform(np.min(xfit), np.max(yfit))]
        # params['init_sigma'][i], params['init_center'][i] = initial_params
        params['init_slope'][i], params['init_intercept'][i] = initial_params

        # Fit the curve
        # result = minimize(sum_of_squared_error, x0=initial_params, args=(xfit, yfit,flip_sigmoid), bounds=param_bounds)
        result = minimize(sum_of_squared_error_linear, x0=initial_params, args=(xfit, yfit), bounds=param_bounds) # result.x = slope and intercept
        # params['est_sigma'][i], params['est_center'][i] = result.x
        params['est_slope'][i], params['est_intercept'][i] = result.x

        # Calculate the goodness of fit
        y_pred = bias_lower + (1 - bias_lower - bias_upper) * linear(x, *result.x)#,flip_sigmoid)
        params['ss_res'][i] = sum_of_squared_error_linear(result.x, x, y)#,flip_sigmoid)
        params['aic'][i] = get_aic_linear(result.x,x,y)
        params['nrmse'][i] = normalized_error(y_targ, y_pred)
        params['r_squared'][i] = r_squared(y_targ, y_pred)

        # Check if the current fit is the best so far
        if result.fun < best_cost:

            best_cost = result.fun

            # if so, save the best-fitting parameters and goodness of fit
            # sigma, center = result.x
            slope, intercept = result.x
            # best_params = center, sigma, bias_lower, bias_upper
            best_params = slope, intercept, bias_lower, bias_upper
            
            goodness_of_fit = params['ss_res'][i], params['nrmse'][i], params['r_squared'][i],params['aic'][i]

            # save other parameters: y(x=center), y(x=0) - the lower bias, y(x=1) - the upper bias
            # center_y = bias_lower + (1 - bias_lower - bias_upper) * sigmoid(center, *result.x)#,flip_sigmoid)
            # bias_xmin = bias_lower + (1 - bias_lower - bias_upper) * sigmoid(np.nanmin(x_pred), *result.x)#,flip_sigmoid)
            # bias_xmax = 1 - (bias_lower + (1 - bias_lower - bias_upper) * sigmoid(np.nanmax(x_pred), *result.x))#,flip_sigmoid))
            # center_y = bias_lower + (1 - bias_lower - bias_upper) * linear(x, *result.x)#,flip_sigmoid)
            bias_xmin = bias_lower + (1 - bias_lower - bias_upper) * linear(np.nanmin(x_pred), *result.x)#,flip_sigmoid)
            bias_xmax = 1 - (bias_lower + (1 - bias_lower - bias_upper) * linear(np.nanmax(x_pred), *result.x))#,flip_sigmoid))
            
            # obj_center = inverse_sigmoid(0.5, (sigma, center, bias_lower, (1-bias_lower-bias_upper)))
            obj_center = inverse_linear(0.5, (slope, intercept))
            # other_params = (center_y, obj_center, bias_xmin, bias_xmax)
            other_params = (obj_center, bias_xmin, bias_xmax)
        
    # Save the fitting progress if needed
    if param_details:
        params.to_csv(param_details)

    return best_params, goodness_of_fit, other_params #, flip_sigmoid

def fit_sigmoid(x_pred, y_targ, rescale=False, padding=False, n_rep=10, param_bounds=None, param_details=None):
    """
    Fit the sigmoid function with multiple random initializations.

    Parameters:
    - x_pred (array-like): Input values for prediction.
    - y_targ (array-like): Target values.
    - rescale (bool, optional): Scale x_pred and y_targ to [0,1] before fitting. Default is False.
    - padding (bool, optional): Add padding to x_pred and y_targ indicating that the plateaus are reached. Default is False.
    - n_rep (int, optional): Number of random initializations. Default is 10.
    - param_bounds (list of tuples, optional): Optional bounds for parameters. Default is None.
    - param_bounds (str, optional): File name to save the fitting progress. Default is None.

    Returns:
    - tuple: Best-fitting parameters and goodness of fit
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize


    best_cost = np.inf

    # Create a DataFrame to save the fitting progress
    params_name = ['iteration', 'init_sigma', 'init_center', 'est_sigma', 'est_center', 'ss_res', 'nrmse', 'r_squared','aic']
    params = pd.DataFrame(np.zeros((n_rep, len(params_name))), columns=params_name)

    # Scale x_pred and y_targ to [0,1] before fitting
    if rescale:
        x = ( x_pred - np.nanmin(x_pred) ) / ( np.nanmax(x_pred) - np.nanmin(x_pred) )
        y = ( y_targ - np.nanmin(y_targ) ) / ( np.nanmax(y_targ) - np.nanmin(y_targ) )
        bias_lower, bias_upper = np.nanmin(y_targ), ( 1 - np.nanmax(y_targ) )
        # bias_lower, bias_upper = y_targ[0], ( 1 - y_targ[-1] )
    else:
        x = x_pred.copy()
        y = y_targ.copy()
        bias_lower, bias_upper = 0, 0

    # Add padding to x_pred and y_targ indicating that the plateaus are reached
    if padding:
        # Sort the arrays before applying the padding
        idxs = np.argsort(x)
        xfit, yfit = x[idxs], y[idxs]
        # Get max and min x value and their y value
        idx_min = np.where(x == np.nanmin(x))[0][0]
        idx_max = np.where(x == np.nanmax(x))[0][0]

        x_step = 1/6
        xfit = np.hstack(((xfit[idx_min] - 2 * x_step), (xfit[idx_min] - x_step), xfit,
                       (xfit[idx_max] + 2 * x_step), (xfit[idx_max] + x_step)))
        yfit = np.hstack((yfit[idx_min], yfit[idx_min], yfit, yfit[idx_max], yfit[idx_max]))
    else:
        xfit, yfit = x.copy(), y.copy()


    for i in range(n_rep):
        params['iteration'][i] = i+1

        # Random initialization within optional bounds
        if param_bounds:
            initial_params = [np.random.uniform(low=lb, high=ub) for lb, ub in param_bounds]
        else:
            initial_params = [np.random.uniform(0, 10), np.random.uniform(np.min(xfit), np.max(yfit))]
        params['init_sigma'][i], params['init_center'][i] = initial_params

        # Fit the curve
        # result = minimize(sum_of_squared_error, x0=initial_params, args=(xfit, yfit,flip_sigmoid), bounds=param_bounds)
        result = minimize(sum_of_squared_error, x0=initial_params, args=(xfit, yfit), bounds=param_bounds)
        params['est_sigma'][i], params['est_center'][i] = result.x

        # Calculate the goodness of fit
        y_pred = bias_lower + (1 - bias_lower - bias_upper) * sigmoid(x, *result.x)#,flip_sigmoid)
        params['ss_res'][i] = sum_of_squared_error(result.x, x, y)#,flip_sigmoid)
        params['aic'][i] = get_aic(result.x,x,y)
        params['nrmse'][i] = normalized_error(y_targ, y_pred)
        params['r_squared'][i] = r_squared(y_targ, y_pred)

        # Check if the current fit is the best so far
        if result.fun < best_cost:
            best_cost = result.fun

            # if so, save the best-fitting parameters and goodness of fit
            sigma, center = result.x
            # if flip_sigmoid:
            #     center = 1 - center
            best_params = center, sigma, bias_lower, bias_upper
            goodness_of_fit = params['ss_res'][i], params['nrmse'][i], params['r_squared'][i],params['aic'][i]

            # save other parameters: y(x=center), y(x=0) - the lower bias, y(x=1) - the upper bias
            center_y = bias_lower + (1 - bias_lower - bias_upper) * sigmoid(center, *result.x)#,flip_sigmoid)
            bias_xmin = bias_lower + (1 - bias_lower - bias_upper) * sigmoid(np.nanmin(x_pred), *result.x)#,flip_sigmoid)
            bias_xmax = 1 - (bias_lower + (1 - bias_lower - bias_upper) * sigmoid(np.nanmax(x_pred), *result.x))#,flip_sigmoid))
            # ...and the x(y=0.5) - the objective
            # if (bias_xmin >= 0.5) & (bias_xmax <= 0.5):
            #     # obj_center = center - 0.5 * sigma * (1-bias_lower-bias_upper)
            #     if ~flip_sigmoid:
            #         obj_center = 0
            #     else:
            #         obj_center = 1
            # elif (bias_xmax >= 0.5) & (bias_xmin <= 0.5):
            #     # obj_center = center + 0.5 * sigma * (1-bias_lower-bias_upper)
            #     if ~flip_sigmoid:
            #         obj_center = 1
            #     else:
            #         obj_center = 0
            # else:
            obj_center = inverse_sigmoid(0.5, (sigma, center, bias_lower, (1-bias_lower-bias_upper)))
            # if flip_sigmoid:
            #     obj_center = 1 - obj_center
            other_params = (center_y, obj_center, bias_xmin, bias_xmax)
        
    # Save the fitting progress if needed
    if param_details:
        params.to_csv(param_details)

    return best_params, goodness_of_fit, other_params#, flip_sigmoid


def inverse_sigmoid(y, params):
    """
    Calculate the inverse of the sigmoid function (logit function) to find the x value given a y value.
    (Calculate at what amount of evidence does a subject flip from social to non-social (y = 0.5))

    Parameters:
    - y (float): The y value for which to find the corresponding x value.
    - params (tuple): Parameters of the sigmoid function.

    Returns:
    - float: The x value corresponding to the given y value in the sigmoid function.
    """
    import numpy as np

    sigma, center, intercept, amplitude = params
    x = center - sigma * np.log( (amplitude / (y - intercept)) - 1 )

    return x


def inverse_linear(y, params):
    """
    Calculate the inverse of the sigmoid function (logit function) to find the x value given a y value.
    (Calculate at what amount of evidence does a subject flip from social to non-social (y = 0.5))

    Parameters:
    - y (float): The y value for which to find the corresponding x value.
    - params (tuple): Parameters of the sigmoid function.

    Returns:
    - float: The x value corresponding to the given y value in the sigmoid function.
    """
    import numpy as np

    slope, intercept = params
    # x = center - sigma * np.log( (amplitude / (y - intercept)) - 1 )
    x = (y-intercept)/slope
    return x


def plot_curve_fit(ax, x_pred, y_targ, params, color, text_loc):#,flip_sigmoid):
    """
    Plot data points and the fitted curve

    Parameters:
    - ax (matplotlib.axes.Axes): Axes for plotting.
    - x_pred (array-like): Input values for prediction.
    - y_targ (array-like): Target values.
    - params (tuple): Fitted parameters.
    - color (str): Color for plotting.
    - text_loc (tuple): Location of goodness of fit text.
    """
    import numpy as np

    text_x, text_y = text_loc

    center, sigma, bias_lower, bias_upper = params
    intercept = bias_lower
    amplitude = 1 - bias_lower - bias_upper

    # Plot the original data points
    ax.plot(x_pred, y_targ, '.', alpha=0.5, color=color)

    # Plot the fitted curve
    x_values = np.linspace(np.nanmin(x_pred), np.nanmax(x_pred), 100)
    y_values = intercept + amplitude * sigmoid(x_values, sigma, center)#,flip_sigmoid)
    ax.plot(x_values, y_values, '-', color=color)
    
    # Add annotation
    y_pred = intercept + amplitude * sigmoid(x_pred, sigma, center)#,flip_sigmoid)
    r_sq = r_squared(y_targ, y_pred)
    nrmse = normalized_error(y_targ, y_pred)
    ax.text(text_x, text_y, "R2=%.3f\nNE=%.3f" % (r_sq, nrmse), color=color)

    ax.set_xlim(np.nanmin(x_pred)-0.05, np.nanmax(x_pred)+0.05)
    ax.set_ylim(np.nanmin(y_targ)-0.05, np.nanmax(y_targ)+0.05)


def plot_curve_fit_linear(ax, x_pred, y_targ, params, color, text_loc):#,flip_sigmoid):
    """
    Plot data points and the fitted curve

    Parameters:
    - ax (matplotlib.axes.Axes): Axes for plotting.
    - x_pred (array-like): Input values for prediction.
    - y_targ (array-like): Target values.
    - params (tuple): Fitted parameters.
    - color (str): Color for plotting.
    - text_loc (tuple): Location of goodness of fit text.
    """
    import numpy as np

    text_x, text_y = text_loc

    # center, sigma, bias_lower, bias_upper = params
    slope, intercept, bias_lower, bias_upper = params
    # intercept = bias_lower
    # amplitude = 1 - bias_lower - bias_upper

    # Plot the original data points
    ax.plot(x_pred, y_targ, '.', alpha=0.5, color=color)

    # Plot the fitted curve
    x_values = np.linspace(np.nanmin(x_pred), np.nanmax(x_pred), 100)
    # y_values = intercept + amplitude * sigmoid(x_values, sigma, center)#,flip_sigmoid)
    y_values = linear(x_values, slope, intercept)#,flip_sigmoid)
    ax.plot(x_values, y_values, '-', color=color)
    
    # Add annotation
    # y_pred = intercept + amplitude * sigmoid(x_pred, sigma, center)#,flip_sigmoid)
    y_pred = linear(x_pred, slope, intercept)
    r_sq = r_squared(y_targ, y_pred)
    nrmse = normalized_error(y_targ, y_pred)
    ax.text(text_x, text_y, "R2=%.3f\nNE=%.3f" % (r_sq, nrmse), color=color)

    ax.set_xlim(np.nanmin(x_pred)-0.05, np.nanmax(x_pred)+0.05)
    ax.set_ylim(np.nanmin(y_targ)-0.05, np.nanmax(y_targ)+0.05)
