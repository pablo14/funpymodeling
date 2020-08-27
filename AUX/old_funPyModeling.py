def status(data):
    if isinstance(data, pd.Series):
        data2=pd.DataFrame(data)
    else:
        data2=data
        
    if(isinstance(data, np.ndarray)):
        data2=pd.DataFrame(data2)

        
    # total de rows
    tot_rows=len(data2)
    
    # total de nan
    d2=data2.isnull().sum().reset_index()
    d2.columns=['variable', 'q_nan']
    
    # percentage of nan
    d2[['p_nan']]=d2[['q_nan']]/tot_rows
    
    # num of zeros
    d2['q_zeros']=(data2==0).sum().values

    # perc of zeros
    d2['p_zeros']=d2[['q_zeros']]/tot_rows

    # total unique values
    d2['unique']=data2.nunique().values
    
    # get data types per column
    d2['type']=[str(x) for x in data2.dtypes.values]
    
    return(d2)


def corr_pair(data, method='pearson'):
    d_cor=data.corr(method)

    d_cor2=d_cor.reset_index() # generates index as column

    d_long=d_cor2.melt(id_vars='index') # to long format, each row 1 var

    d_long.columns=['v1', 'v2', 'R']
    
    d_long[['R2']]=d_long[['R']]**2
    
    d_long2=d_long.query("v1 != v2") # don't need the auto-correlation

    return(d_long2)


def num_vars(data, exclude_var=None):
    num_v = data.select_dtypes(include=['int64', 'float64']).columns
    if exclude_var is not None: 
        num_v=num_v.drop(exclude_var)
    return num_v


def profiling_num(data):
    # ask for series/array or dataframe
    if(len(data.shape)==1):
        d=pd.DataFrame({data.name:data})
    
    # explicit keep the num vars
    d=data[num_vars(data)]
    
    des1=pd.DataFrame({'mean':d.mean().transpose(), 
                   'std_dev':d.std().transpose()})

    des1['variation_coef']=des1['std_dev']/des1['mean']
    
    d_quant=d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().add_prefix('p_')
    
    des2=des1.join(d_quant, how='outer')
    
    des_final=des2.copy()
    
    des_final['variable'] = des_final.index
    
    des_final=des_final.reset_index(drop=True)
    
    des_final=des_final[['variable', 'mean', 'std_dev','variation_coef', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95', 'p_0.99']]
    
    return des_final


def freq_tbl_logic(var, name):
    cnt=var.value_counts()
    df_res=pd.DataFrame({'frequency': var.value_counts(), 'percentage': var.value_counts()/len(var)})
    df_res.reset_index(drop=True)
    
    df_res[name] = df_res.index
    
    df_res=df_res.reset_index(drop=True)
    
    df_res['cumulative_perc'] = df_res.percentage.cumsum()/df_res.percentage.sum()
    
    df_res=df_res[[name, 'frequency', 'percentage', 'cumulative_perc']]
    
    return df_res


def todf(data):
    if isinstance(data, list):
        data=np.array(data)

    if(len(data.shape))>2:
        raise Exception("I live in flattland! (can't handle objects with more than 2 dimensions)") 

    if isinstance(data, pd.Series):
        data2=pd.DataFrame({data.name: data})
    elif isinstance(data, np.ndarray):
        if(data.shape==1):
            data2=pd.DataFrame({'var': data})
        else:
            data2=pd.DataFrame(data)
    else: 
        data2=data
        
    return data2



def freq_tbl(data):
    data=todf(data)
    
    cat_v=cat_vars(data)
    
    if(len(cat_v)>1):
        for col in cat_v:
            print(freq_tbl_logic(data[col], name=col))
            print('\n----------------------------------------------------------------\n')
        return cat_v
    else:
        return freq_tbl_logic(data.iloc[:,0], name=data.columns[0])
    
    
    
def coord_plot(data, cluster_var):
    # 1- group by cluster, get the means
    x_grp=data.groupby(cluster_var).mean()
    x_grp[cluster_var] = x_grp.index 
    x_grp=x_grp.reset_index(drop=True)
    x_grp # data con las variables originales
    
    # 2- normalizing the data min-max
    x_grp_no_tgt=x_grp.drop(cluster_var, axis=1)

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(x_grp_no_tgt)
    x_grp_mm=mm_scaler.transform(x_grp_no_tgt)

    # 3) convert to df
    df_grp_mm=pd.DataFrame(x_grp_mm, columns=x_grp_no_tgt.columns)

    df_grp_mm[cluster_var]=x_grp[cluster_var] # variables escaladas

    # plot
    parallel_coordinates(df_grp_mm, cluster_var, colormap=plt.get_cmap("Dark2"))
    plt.xticks(rotation=90)
    
    return [x_grp, df_grp_mm]