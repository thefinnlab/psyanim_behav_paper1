def fix_obj_center(df):
    import numpy as np
    
    for i,row in df.iterrows():
        if row['flipped'] == False: # only deal with regular curves
            if np.isnan(row['obj_center']): # full curve is in the upper or lower half 
                #(not fixed in the curvefit script)
                if (row['bias_xmin'] >= 0.5) & (row['bias_xmax'] <= 0.5): # upper half
                    df.at[i, 'obj_center'] = 0
                elif (row['bias_xmax'] >= 0.5) & (row['bias_xmin'] <= 0.5): # lower half
                    df.at[i, 'obj_center'] = 1

            if np.isinf(row['obj_center']):
                if (row['bias_xmin'] >= 0.5) & (row['bias_xmax'] <= 0.5): # upper half
                    df.at[i, 'obj_center'] = 0
                elif (row['bias_xmax'] >= 0.5) & (row['bias_xmin'] <= 0.5): # lower half
                    df.at[i, 'obj_center'] = 1

            if row['obj_center'] > 1:
                df.at[i, 'obj_center'] = 1
                
            if row['obj_center'] < 0:
                df.at[i, 'obj_center'] = 0

        elif row['flipped'] == True: # if flipped, assign obj_center as np.nan
            df.at[i, 'obj_center'] = np.nan
             
    return df

def add_new_params(df):

    import numpy as np 
    
    df['range'] = np.nan #1-df['bias_xmin']-df['bias_xmax']
    df['bias'] =  np.nan #df['bias_xmin']-df['bias_xmax']

    for i,row in df.iterrows():
        
        if (row['bias_xmin'] <= 0) & (row['bias_xmax'] <= 0):
            df.at[i, 'bias_xmin'] = 0
            df.at[i, 'bias'] = 0 # no fraction of bias comes from either end
            df.at[i,'range']= 1

        elif (row['bias_xmin'] < 0) | (row['bias_xmax'] < 0):
            if (row['bias_xmin'] < 0) & (row['bias_xmax'] >= 0):
                df.at[i, 'bias_xmin'] = 0
                df.at[i, 'bias'] = 0 # no fraction of bias comes from the lower end
                df.at[i,'range'] = 1 - df.at[i,'bias_xmax']
    
            elif (row['bias_xmin'] >= 0) & (row['bias_xmax'] < 0):
                df.at[i, 'bias_xmax'] = 0
                df.at[i, 'bias'] = 1 # no fraction of bias comes from the upper end
                df.at[i,'range'] = 1 - df.at[i,'bias_xmin']

        elif (row['bias_xmin'] >= 0) & (row['bias_xmax'] >= 0):
                if (row['bias_xmin'] == 0) & (row['bias_xmax'] == 0):
                    df.at[i,'range'] = 1
                    df.at[i,'bias'] =  0
                else:
                    df.at[i,'range'] = 1-df.at[i,'bias_xmin']-df.at[i,'bias_xmax']
                    df.at[i,'bias'] =  df.at[i,'bias_xmin']/(df.at[i,'bias_xmin'] + df.at[i,'bias_xmax'])
                 #df.at[i,'bias_xmin']-df.at[i,'bias_xmax'] (bias_diff)
    return df            

def get_p(p):
    if p < .001:
        return "p < .001"
    elif p < .01:
        return f"p={p:.3f}"
    else:
        return f"p={p:.2f}"

