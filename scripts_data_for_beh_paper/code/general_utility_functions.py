def get_p(p):
    if p < .001:
        return "p < .001"
    elif p < .01:
        return f"p={p:.3f}"
    else:
        return f"p={p:.2f}"


def return_minmax(df,param):
    # print min nax values of entered params to sanity check curve fit params, most should be within 0 and 1
    print(f'{param}: min={df[param].describe()["min"]:.2f}, max={df[param].describe()["max"]:.2f}')



def explore_correlations(df, xcol,ycol,ax,col):

    sns.regplot(data=df,x=xcol,y=ycol,ax=ax,marker = '.', color=col)
    x,y = df[xcol], df[ycol]
    rows = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y))
    x = x[rows]
    y = y[rows]
    x = x[rows]
    y = y[rows]
    r,p = pearsonr(x,y)
    if p < .05:
        color = 'k'
        fontweight = 'bold'
    else:
        color = 'k'
        fontweight = 'normal'
    ax.annotate(f'r_p={r:.2f} ({get_p(p)})',xy=(0.05,.2),xycoords='axes fraction', color = color, fontweight = fontweight, ha='left',va='top')

    r,p = spearmanr(x,y)
    if p < .05:
        color = 'k'
        fontweight = 'bold'
    else:
        color = 'k'
        fontweight = 'normal'
    ax.annotate(f'r_s={r:.2f} ({get_p(p)})',xy=(0.05,.1),xycoords='axes fraction', color = color, fontweight = fontweight, ha='left',va='top')
