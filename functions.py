import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt


def get_table_download_link(file):
    data = pd.read_csv(file, sep=";")
    csv_file = data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="metrics_description.txt">Download description metrics file</a>'


def limits(x, dict_ranges, col):
    if x > dict_ranges[col][1]:
        return dict_ranges[col][1]
    else:
        if x < dict_ranges[col][0]:
            return dict_ranges[col][0]
        else:
            return x

def radarchart_pyplot(data, teams, cols, ranges_cols, percentile): 
    
    from functions_radarchart import ComplexRadar
    
    df_filter = data[['Squad'] + cols]
    
    df_teams_filter = df_filter[df_filter.Squad.isin(teams)]
    ranges_cols_selected = [ranges_cols[s] for s in cols]
    
    if percentile:
        
        df_plot = pd.DataFrame({'Squad': df_filter['Squad'].tolist()})
        
        for c in df_filter.columns[1:]:
            p_c = round((df_filter[c]-df_filter[c].min())\
                        / (df_filter[c].max()-df_filter[c].min())*100)
            p_c.reset_index(drop=True, inplace=True)
            df_plot = pd.concat([df_plot, p_c], axis=1)
            
        for c in df_plot.columns[1:]:
            df_plot[c] = df_plot[c].astype(int)
            
        df_plot = df_plot[df_plot.Squad.isin(teams)]
            
    else:
        df_plot = df_teams_filter.copy()
        
        for c in df_plot.columns[1:]:
            df_plot[c] = df_plot[c].apply(
                lambda x: limits(x, ranges_cols, c))
            
    fig = plt.figure(figsize=(6,6))
    radar = ComplexRadar(fig, tuple(cols), ranges_cols_selected)
    for t in teams:
        df_teams_t = list(
            df_plot[df_plot.Squad == t].\
                iloc[:,1:].values[0])
        radar.plot(tuple(df_teams_t), label = t)
        radar.fill(tuple(df_teams_t), alpha = 0.2)
        
    return fig, df_teams_filter


def calculate_percentiles(
        df_from_radar, df_total, 
        metrics_selection, teams_selection, select_info_table, dict_metrics, 
        select_percentiles):
    df_radar_T = df_from_radar.copy()
    df_radar_T.index = df_radar_T.Squad
    df_radar_T = df_radar_T.T.iloc[1:,:]
    for c in df_radar_T.columns:
        df_radar_T[c] = df_radar_T[c].astype(float)
    # calculo de percentiles
    df_percentiles = pd.DataFrame()
    for c in list(df_radar_T.index):
        percentil_c = round((df_total[c] - df_total[c].min())\
                / (df_total[c].max() - df_total[c].min())*100)
        df_percentiles = pd.concat([df_percentiles, 
                                    percentil_c], axis=1)
    df_percentiles.index = df_total['Squad']
    df_percentiles = df_percentiles[df_percentiles.index.isin(
        teams_selection)]
    df_percentiles_T = df_percentiles.T
    df_percentiles_T = df_percentiles_T[df_radar_T.columns]
    df_percentiles_T.rename(
        columns={c:c+ " p" for c in df_percentiles_T.columns}, 
        inplace=True)
    for c in df_percentiles_T.columns:
        df_percentiles_T[c] = df_percentiles_T[c].astype(int)
    df_full = pd.concat([df_radar_T, df_percentiles_T], axis=1)
    one_dict = {c: (c, 'Value') 
                for c in df_full.columns[:len(teams_selection)]}
    one_dict.update({c: (c[:-2], 'Percentile') 
                    for c in df_full.columns[len(teams_selection):]})
   
    df_full.rename(columns = one_dict, inplace=True)
    if select_info_table: 
        dict_keys = {y: dict_metrics[y] for y in df_percentiles.columns}
        df_full.index = df_full.index.map(dict_keys)
    # order df
    l = []
    i = 0
    while i < len(teams_selection):
        l.append(i)
        l.append(i+len(teams_selection))
        i+=1
    df_full = df_full[[df_full.columns[j] for j in l]]
    if select_percentiles:
        df_full = df_full[[c for c in df_full.columns if 'Percentile' in c]]
    df_full.columns = pd.MultiIndex.from_tuples(df_full.columns)
    return df_full
    
    


def scatterplot_stats(df, stats, competition, dict_metrics, tendencia, 
                      colors, minus, color_rectangle):
    import plotly.express as px
    import os
    from sklearn.linear_model import LinearRegression
    import plotly.graph_objects as go
    from sklearn.metrics import r2_score
    
    df_filter = df[df.Competition.isin(competition)]

    df_stats = df_filter[['Squad', 'Competition'] + stats]
    dict_stats = {y: dict_metrics[y] for y in stats}
    df_stats.rename(columns=dict_stats, inplace=True)
    stats = list(dict_stats.values())
    
    # linearRegression
    lr = LinearRegression()
    lr.fit(df_stats[[stats[0]]], df_stats[stats[1]])
    coef_x = lr.coef_[0]
    ct = lr.intercept_
    lr_predict = lr.predict(df_stats[[stats[0]]])
    r2 = round(r2_score(df_stats[stats[1]], lr_predict), 2)
    y_tendencia = [ct + j*coef_x for j in list(df_stats[stats[0]].values)]
    
    fig = px.scatter(df_stats, x=stats[0], y=stats[1], 
                     hover_data=['Squad'])
    if tendencia:
        fig.add_trace(go.Scatter(x=df_stats[stats[0]], 
                                 y=y_tendencia,
                                 mode='lines', 
                                 name='Trendline (r2 = ' + str(r2) + ')', 
                                 line=dict(color='royalblue')))

    max_x = 0.1*(df_stats.iloc[:,-2].max() - df_stats.iloc[:,-2].min())
    max_y = 0.1*(df_stats.iloc[:,-1].max() - df_stats.iloc[:,-1].min())
    for i in range(len(df_stats)):
        t = df_stats.iloc[i,0]
        with open(os.getcwd() + "/teams/" + t + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # Add the prefix that plotly will want when using the string as source
        encoded_image = "data:image/png;base64," + encoded_string
        fig.add_layout_image(
            dict(
                source = encoded_image,
                xref = "x", yref = "y", 
                x = df_stats.iloc[i,-2], y = df_stats.iloc[i,-1], 
                sizex = max_x, sizey = max_y, 
                sizing = "contain", layer = 'above',
                xanchor = "center", yanchor = "middle"))
    fig.update_traces(textposition='top center')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
    
    if color_rectangle:
        
        # inferior derecha
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = list(np.repeat(df_stats[stats[1]].min()-minus, 2)), 
                                 fill = 'none', line_color = colors[1], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y=list(np.repeat(df_stats[stats[1]].mean(), 2)),
                                 fill='tonexty', line_color = colors[1], mode='lines', showlegend=False))
        # inferior izquierda
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = list(np.repeat(df_stats[stats[1]].min()-minus, 2)), 
                                 fill = 'none', line_color = colors[0], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y=list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill='tonexty', line_color = colors[0], mode='lines', showlegend=False))
        # superior izquierda
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill = 'none', line_color = colors[2], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = [df_stats[stats[1]].max()+minus, df_stats[stats[1]].max()+minus], 
                                 fill = 'tonexty', line_color = colors[2], mode='lines', showlegend=False))
        # superior derecha
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill = 'none', line_color = colors[3], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = [df_stats[stats[1]].max()+minus, df_stats[stats[1]].max()+minus], 
                                 fill = 'tonexty', line_color = colors[3], mode='lines', showlegend=False))
                                 
        fig.update_yaxes(range=[df_stats[stats[1]].min()-minus, df_stats[stats[1]].max()+minus], row=1, col=1)
        fig.update_xaxes(range=[df_stats[stats[0]].min()-minus, df_stats[stats[0]].max()+minus], row=1, col=1)
        
    else:
        
        fig.add_shape(type='line', 
                      x0=df_stats[stats[0]].mean(), 
                      y0=df_stats[stats[1]].min(),
                      x1=df_stats[stats[0]].mean(),
                      y1=df_stats[stats[1]].max(),
                      line=dict(dash='dot', width=1))
        fig.add_shape(type='line', 
                      x0=df_stats[stats[0]].min(), 
                      y0=df_stats[stats[1]].mean(),
                      x1=df_stats[stats[0]].max(),
                      y1=df_stats[stats[1]].mean(),
                      line=dict(dash='dot', width=1))
    
    return fig



def hiearchical_clustering(df_dendogram):
    
    import plotly.figure_factory as ff
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
    from scipy.spatial.distance import pdist
    import scipy.cluster.hierarchy as sch
    
    scaler = StandardScaler()
    df_preprocess = scaler.fit_transform(df_dendogram.iloc[:,2:])
    
    list_coph = []
    methods = {'single': 'Nearest Point Algorithm (single)', 
               'complete': 'Farthest Point Algorithm (complete)', 
               'ward': 'Incremental Algorithm (ward)', 
               'average': 'UPGMA Algorithm (average)', 
               'weighted': 'WPGMA Algorithm (weighted)', 
               'median': 'WPGMC Algorithm (median)', 
               'centroid': 'UPGMC Algorithm (centroid)'}
    
    for m in list(methods.keys()):
        Z = linkage(df_preprocess, m)
        c, coph_dists = cophenet(Z, pdist(df_preprocess))
        list_coph.append(c)
    results = pd.DataFrame(
            zip(list(methods.values()), list_coph), 
            columns = ['Algorithm and Distance Method', 'Cophenetic Correlation'])
    results = results.sort_values(
            by='Cophenetic Correlation', ascending=False)
    results.reset_index(drop=True, inplace=True)
    
    best = [m for m, c in zip(list(methods.keys()), list_coph) if c == max(list_coph)]
    
    names = list(df_dendogram.Squad)
    fig = ff.create_dendrogram(df_preprocess, orientation='bottom', labels=names, 
                               linkagefun=lambda x: sch.linkage(x, best[0]))
    fig.update_layout(width=800, height=500, 
                      title = 'Dendogram using ' + \
                      " ".join(methods[best[0]].split(" ")[:-1]))
                      #paper_bgcolor='rgb(143,188,143)')
    return fig, results


def plotly_PCA(df_clustering, pca_method, pct_variance):
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    
    df_clustering = df_clustering.reset_index(drop=True)
    df_clustering_values = df_clustering.iloc[:,2:]
    
    scaler = StandardScaler()
    df_preprocess = scaler.fit_transform(df_clustering_values)
    
    if pca_method == 'Eigenvalues above the mean':
        S = np.cov(df_preprocess.T)
        autovalores, autovectores = np.linalg.eigh(S)
        nPCA = (sorted(autovalores) > autovalores.mean()).sum()
        colors = ['royalblue'] * len(autovalores)
        colors[nPCA-1] = 'indianred'
        fig = go.Figure(
                data=[go.Bar(x=list(range(1,len(autovalores))), 
                             y=sorted(autovalores, reverse=True), 
                             showlegend=False, marker_color = colors), 
                      go.Scatter(x=list(range(1,len(autovalores))), 
                                 y=[autovalores.mean() for i in range(len(autovalores))],
                                 showlegend=False, line=dict(
                                         color='indianred', dash='dash', width=1))])
        fig.update_layout(barmode='group', title='Above-average eigenvalues', 
                          xaxis_title='CPs', yaxis_title='Eigenvalues')
        
    else:
        pca = PCA()
        pca.fit_transform(df_preprocess)
        nPCA = np.count_nonzero(
                pca.explained_variance_ratio_.cumsum() <= pct_variance/100) + 1
        colors = ['royalblue'] * (len(pca.explained_variance_)+1)
        colors[nPCA-1] = 'indianred'
        fig = go.Figure([go.Scatter(x=list(range(1,len(pca.explained_variance_)+1)), 
                                   y=pca.explained_variance_ratio_.cumsum(), 
                                   showlegend=False,
                                   mode='lines+markers', marker_color = colors,
                                   line=dict(color='royalblue')), 
                         go.Scatter(x=list(range(1,len(pca.explained_variance_)+1)), 
                                    y=[round(pct_variance/100,2) 
                                    for i in range(len(pca.explained_variance_)+1)],
                                    showlegend=False, 
                                    line = dict(color = 'indianred', dash='dash', width=1))])
        fig.update_layout(title='Cumulative variance explained by the components', 
                          xaxis_title='CPs', yaxis_title='Cumulative explained variance')
    return fig
    


def pca_kmeans_clustering(df_clustering, pca_method, pct_variance, selection, n):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from kneed import KneeLocator
    from sklearn.cluster import KMeans
    import plotly.express as px
    
    df_clustering = df_clustering.reset_index(drop=True)
    df_clustering_values = df_clustering.iloc[:,2:]
    
    # PCA
    scaler = StandardScaler()
    df_preprocess = scaler.fit_transform(df_clustering_values)
    
    if pca_method == 'Eigenvalues above the mean':
        S = np.cov(df_preprocess.T)
        autovalores, autovectores = np.linalg.eigh(S)
        nPCA = (sorted(autovalores) > autovalores.mean()).sum()
    else:
        pca = PCA()
        pca.fit_transform(df_preprocess)
        nPCA = np.count_nonzero(
                pca.explained_variance_ratio_.cumsum() <= pct_variance/100) + 1
        
    pca = PCA(n_components=nPCA)
    pca.fit(df_preprocess)  
    explained_pca = pca.explained_variance_ratio_
    scores_pca = pca.transform(df_preprocess)
    
    # KMeans
    if selection:
        nClusters = n
        method_cluster = 'Manual Selection'
    else:
        dict_metric_cluster = dict()
        for n_cluster in range(2, 10):
            kmeans = KMeans(n_clusters = n_cluster, 
                            random_state = 1).fit(scores_pca)
            metric = kmeans.inertia_
            dict_metric_cluster[n_cluster] = metric
        nClusters = KneeLocator(list(dict_metric_cluster.keys()), 
                                list(dict_metric_cluster.values()), 
                                curve='convex', direction='decreasing').elbow
        if nClusters is None:
            method_cluster = 'Silhouette Method'
            nClusters = max(dict_metric_cluster, key=dict_metric_cluster.get)
        else:
            method_cluster = 'Elbow Method'
    kmeans = KMeans(n_clusters=nClusters, random_state=1).fit(scores_pca)
    dataset_final = pd.concat(
        [df_clustering, pd.DataFrame(scores_pca)], axis=1)
    dataset_final.columns = list(df_clustering.columns) + \
        ['Comp'+str(i) for i in range(1,nPCA+1)]
    dataset_final['Cluster'] = kmeans.labels_
    dataset_final['Cluster'] = dataset_final['Cluster'].map(str)
    
    # Plot
    fig = px.scatter(dataset_final, x='Comp1', y='Comp2', 
                     color='Cluster', text='Squad')
    fig.update_layout(xaxis_title = 'Comp1 (' + str(round(explained_pca[0]*100, 2)) + ' %)', 
                      yaxis_title = 'Comp2 (' + str(round(explained_pca[1]*100, 2)) + ' %)', 
                      title_text = str(nClusters) + ' clusters using ' + \
                          method_cluster + ' and ' + str(nPCA) + ' CPs')
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)#, paper_bgcolor='rgb(143,188,143)')
    return fig

def color_info(s):
    return ['background-color: lightgray' for v in s]


def swarmplot(df, metric, team, dict_metrics):
    import plotly.express as px
    df_filter = df[['Squad', 'Competition', metric]]
    limits = [round(np.percentile(df_filter[metric], i), 2)
              for i in range(20,120,20)] + [team]
    df_filter['Percentile'] = df_filter[metric].apply(
        lambda x: str([j for j in limits[:-1] if x <= j][0]))
    df_filter.loc[df_filter.Squad == team, 'Percentile'] = team
    colors = ['red', 'coral', 'gold', 'lightgreen', 'forestgreen', 'black']
    
    #plot
    fig = px.strip(data_frame = df_filter, x = metric, color = 'Percentile', 
                   hover_name = 'Squad', hover_data = ['Competition', metric], 
                   color_discrete_sequence = colors, 
                   color_discrete_map = {str(i):j for i,j in zip(limits, colors)}, 
                   stripmode = 'overlay', template = 'plotly')
    fig.update_layout(showlegend = False, xaxis_title = metric, 
                      title = dict_metrics[metric])
    fig.update_yaxes(showticklabels = False)
    return fig


def similar_team(
        df_similar, 
        selected_team, 
        selected_number, 
        view_description,
        dict_metrics):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go
    import os 
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from IPython.core.display import HTML
    
    df_similar_values = df_similar.iloc[:,2:]
    scaler = StandardScaler()
    df_similar_std = scaler.fit_transform(df_similar_values)
    
    df_cosine = pd.DataFrame(cosine_similarity(pd.DataFrame(df_similar_std)))
    df_cosine.columns = df_similar['Squad']
    df_cosine.index = df_similar['Squad']
    
    for c in df_cosine.columns:
        df_cosine[c] = (df_cosine[c] - np.min(df_cosine[c])) / np.ptp(df_cosine[c])
    
    #Filter by team
    n_similar_teams = df_cosine[selected_team].sort_values(
        ascending=False)[:selected_number+1]
    n_similar_teams_df = n_similar_teams.reset_index()
    
    n_similar_teams_df = n_similar_teams_df.merge(
        df_similar, on = 'Squad', how = 'left')

    n_similar_teams_df[selected_team] = n_similar_teams_df[selected_team].apply(
        lambda x: round(100*x, 2))
    n_similar_teams_df.rename(columns={selected_team: '% Similarity'}, inplace=True)
    n_similar_teams_df = n_similar_teams_df[['Squad', 'Competition'] + 
                                           [c for c in n_similar_teams_df.columns
                                            if c not in ['Squad', 'Competition']]]
    
    n_similar_teams_df_copy = n_similar_teams_df.copy()
    img_list = []
    for c in list(n_similar_teams_df_copy['Squad']):
        with open(os.getcwd() + "/teams/" + c + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_image = r'<img src="data:image/png;base64,%s" width="50"/>' % encoded_string
            img_list.append(encoded_image)
    n_similar_teams_df_copy['img'] = img_list
    
    n_similar_teams_df_copy = n_similar_teams_df_copy[['img', 'Squad'] + \
                                                      list(n_similar_teams_df_copy.columns[1:-1])]
        
    if view_description:
        dict_keys = {y: dict_metrics[y] for y in n_similar_teams_df_copy.columns 
                     if y in dict_metrics.keys()}
        n_similar_teams_df_copy.rename(columns=dict_keys, inplace=True)
    #drop img title
    n_similar_teams_df_copy.rename(columns={'img':''}, inplace=True)    
    df_stats = pd.DataFrame()
    for i in range(len(n_similar_teams_df_copy)):
        df_i = n_similar_teams_df_copy.iloc[i,:].T
        df_stats = pd.concat([df_stats, df_i], axis=1)
        
    df_stats.columns = df_stats.iloc[0,:]
    df_stats_tabla = df_stats.iloc[1:,:]
    
    html = (
        df_stats_tabla.style
        .format("{:.2f}", subset = (df_stats_tabla.index[2:], df_stats_tabla.columns))
        .background_gradient(cmap='RdYlGn', axis = 1, low = .25, high = .25, 
                             subset = (df_stats_tabla.index[2:], df_stats_tabla.columns))
        .apply(color_info, axis = 1, 
               subset = (df_stats_tabla.index[:2], df_stats_tabla.columns))
        .set_properties(**{'font-family': 'Calibri', 'text-align': 'center', 
                           'border-width':'thin', 'border-color':'black', 
                           'border-style':'solid', 'border-collapse':'collapse'})
        .render()
        )

    return n_similar_teams_df, HTML(html)


def color_percentiles_zones(val):
    if val < 25:
        color = 'red'
    else:
        if val < 50:
            color = 'darkorange'
        else:
            if val < 75:
                color = 'gold'
            else:
                color = 'green'
    return color


def draw_pitch(team, list_colors, list_values, metric_selected):
    
    line = "black"
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig,ax = plt.subplots(figsize=(10.4,6.8))
    plt.xlim(-1,121)
    plt.ylim(-4,91)
    ax.axis('off')

    # lineas campo
    ly1 = [0,0,90,90,0]
    lx1 = [0,120,120,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)

    # area grande
    ly2 = [25,25,65,65] 
    lx2 = [120,103.5,103.5,120]
    plt.plot(lx2,ly2,color=line,zorder=5)

    ly3 = [25,25,65,65] 
    lx3 = [0,16.5,16.5,0]
    plt.plot(lx3,ly3,color=line,zorder=5)

    # porteria
    ly4 = [40.5,40.7,48,48]
    lx4 = [120,120.2,120.2,120]
    plt.plot(lx4,ly4,color=line,zorder=5)

    ly5 = [40.5,40.5,48,48]
    lx5 = [0,-0.2,-0.2,0]
    plt.plot(lx5,ly5,color=line,zorder=5)

    # area pequeña
    ly6 = [36,36,54,54]
    lx6 = [120,114.5,114.5,120]
    plt.plot(lx6,ly6,color=line,zorder=5)

    ly7 = [36,36,54,54]
    lx7 = [0,5.5,5.5,0]
    plt.plot(lx7,ly7,color=line,zorder=5)

    # lineas y puntos
    vcy5 = [0,90] 
    vcx5 = [60,60]
    plt.plot(vcx5,vcy5,color=line,zorder=5)

    plt.scatter(109,45,color=line,zorder=5)
    plt.scatter(11,45,color=line,zorder=5)
    plt.scatter(60,45,color=line,zorder=5)

    # circulos
    circle1 = plt.Circle((109.5,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle2 = plt.Circle((10.5,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle3 = plt.Circle((60,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

    # rectangulos
    rec1 = plt.Rectangle((103.5,30), 16, 30,ls='-',color="white", zorder=1,alpha=1)
    rec2 = plt.Rectangle((0,30), 16.5, 30,ls='-',color="white", zorder=1,alpha=1)
    #rec3 = plt.Rectangle((-1,-1), 122, 92,color=pitch,zorder=1,alpha=1)
    
    # colors
    zone1 = plt.Rectangle((0, 0), 40, 90, color=list_colors[0], zorder=1, alpha=0.5)
    zone2 = plt.Rectangle((40, 0), 40, 90, color=list_colors[1], zorder=1, alpha=0.5)
    zone3 = plt.Rectangle((80, 0), 40, 90, color=list_colors[2], zorder=1, alpha=0.5)

    #ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)
    
    # zones
    ax.add_artist(zone1)
    ax.add_artist(zone2)
    ax.add_artist(zone3)
    
    # text
    plt.text(17, 80, str(list_values[0]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    plt.text(46, 80, str(list_values[1]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    plt.text(97, 80, str(list_values[2]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    
    # flecha
    plt.arrow(55, -3, dx = 10, dy = 0, linewidth = 1.5, head_width=1)
    
    #image
    img = mpimg.imread('teams/'+team+'.png')
    plt.imshow(img, zorder=0, extent = [35, 85, 20, 70])
    
    plt.title(metric_selected, fontweight = 'semibold')
    
    return fig

    
def color_percentiles(val):
    if val < 25:
        color = 'red'
    else:
        if val < 50:
            color = 'darkorange'
        else:
            if val < 75:
                color = 'gold'
            else:
                color = 'green'
    return color


def plot_zones(df_zone, team_zone_select, metric_selected):
    df_zone_percent = df_zone.copy()
    for c in [c for c in df_zone_percent.columns if '90' in c]:
        df_zone_percent[c] = round((df_zone_percent[c] - df_zone_percent[c].min()) \
                               / (df_zone_percent[c].max() - df_zone_percent[c].min())*100)
        df_zone_percent[c] = df_zone_percent[c].astype(int)
    df_zone_percent = df_zone_percent[['Squad'] + \
                                      [c for c in df_zone_percent.columns 
                                       if '90' in c]]
    df_zone_percent.rename(columns={c: c[:-3] 
                                    for c in df_zone_percent.columns[1:]}, 
                           inplace=True)
    df_zone_pct = df_zone[[c for c in df_zone.columns[:-3]]]

    i_team = [i for i in range(len(df_zone_percent)) 
              if df_zone_percent.iloc[i,0] == team_zone_select][0]
    colors_team = list(df_zone_percent.iloc[i_team,1:].\
                       apply(color_percentiles_zones).values)
    values_team = list(df_zone_pct.iloc[i_team,1:].values)
    fig = draw_pitch(team_zone_select, colors_team, values_team, metric_selected)
    return fig
    
    

    


    
    
    
    
    