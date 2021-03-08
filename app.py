# Librerias
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

from functions import *

st.title('Welcome to  clubes2021! :wave:')
st.write(':calendar: *Update date: 08/03/2021*')

def load_data(ruta, encoding):
    return pd.read_csv(ruta, sep=";", encoding=encoding)

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
    return 'color: %s' % color


def main():
    
    df = load_data('data/FBREF_clubes2020_data.csv', encoding='latin')
    df_description_metrics = load_data('data/summarize_field.csv', encoding='latin')
    df_description_metrics.index = df_description_metrics['Original']
    dict_description_metrics = df_description_metrics['Description'].to_dict()
    
    page = st.sidebar.radio("Team Performance", 
                            ("Radar Chart and Information", 
                             "Scatter Plot Stats",
                             "Heatmap Zones Stats",
                             "Similar Teams"))
    
    if page == 'Radar Chart and Information':
        
        st.markdown("""___""")
        
        st.write(':soccer: Comparison of the performance of the teams of the \
                 5 major leagues using the radar chart plot that allows us to \
                 visualize a large number of statistics at the same time.')
        
        teams = list(df['Squad'])
        competitions = list(df['Competition'].unique())
        
        competition_selection = st.selectbox(
                'Competitions', ['All Competitions'] + competitions)
        
        if competition_selection != 'All Competitions':
            default_dict = {'La Liga': ['Barcelona', 'Real Madrid'], 
                            'Premier League': ['Liverpool', 'Manchester City'], 
                            'Ligue 1': ['PSG', 'Lyon'], 
                            'Bundesliga': ['Bayern Munich', 'Dortmund'], 
                            'Serie A': ['Juventus', 'Inter']}
            teams_competition = list(
                    df[df.Competition == competition_selection]['Squad'])
            default_teams = default_dict[competition_selection]
        else:
            teams_competition = teams
            default_teams = ['Bayern Munich', 'PSG']
        
        teams_selection = st.multiselect(
                'Teams', options = teams_competition, 
                default = default_teams)
        
        st.write(':mag: Select metrics to compare the behavior of the selected \
                 teams and understand their performance in the three phases of \
                 the game. Download of the file that contains the description \
                 of each metric.')
        
        if competition_selection != 'All Competitions':
            df_competition = df[df.Competition == competition_selection]
        else:
            df_competition = df.copy()
        
        st.markdown(get_table_download_link('data/description_field.csv'), 
                    unsafe_allow_html=True)
        
        with st.beta_expander("Radar Information"):
            
            st.write(':memo: Radar is a way of visualizing a large number of \
                     stats at one time. Traditional radar boundaries represent \
                     the top 5% and bottom 5% of all statistical production \
                     (teams of selected competition).')
            
            st.write(':new: In addition to traditional radars, it is now possible \
                     to display metrics as percentiles for that position. This \
                     removes the 5% / 95% limits from classic radars and replaces \
                     the normal stats with the percentiles for the entire population \
                     at that position.')
                     
            radio_radar = st.radio('Radar', ['Traditional Radar', 
                                              'Percentile Radar'])
            if radio_radar == 'Percentile Radar':
                percentile = True
                ranges_cols = {c: (0,100) for c in df_competition.columns[2:]}            
            else:
                percentile = False
                ranges_cols = {c: (np.percentile(df_competition[c].values, 5), 
                                   np.percentile(df_competition[c].values, 95)) 
                               for c in df_competition.columns[2:]}
        
        cols_selection = st.multiselect(
                'Metrics', options = list(df.columns[2:]), 
                default = ['Gls/90', 'Ast/90', 'KP/90', 'PPA/90', 
                           'Press%', 'Passes%', 'Poss%', 'Tkl/90', 'Recov/90',
                           'GA/90', 'PSxG+/-', 'SoTA/GA', 'Aerial%', 'Dribbles%', 
                           'Touches/90', 'SoT/G', 'Points/90'])
        plot_radar, df_radar = radarchart_pyplot(
                df_competition, teams_selection, cols_selection, 
                ranges_cols, percentile)
        st.pyplot(plot_radar)
        df_radar = df_radar.reset_index(drop=True)
        
        st.subheader('Information and analysis by percentile')
        
        select_info_1 = st.checkbox(
            'Full name of the metric in the table') 
        select_info_1_1 = st.checkbox(
            'View only percentiles')
        
        df_percentile = calculate_percentiles(
            df_radar, df_competition, cols_selection, teams_selection, 
            select_info_1, dict_description_metrics, select_info_1_1)
        
        #with st.beta_expander("Information and analysis by percentiles"):
        st.write(
                df_percentile.style.applymap(
                    color_percentiles, 
                    subset = [c for c in df_percentile.columns
                              if 'Percentile' in c]) \
                    .format(
                        {c: "{:.2f}" for c in df_percentile.columns 
                         if 'Value' in c}))
                
        st.write(':computer: The calculation of the percentile is considering \
                     the teams of the selected competition.')
        st.image('img/escala.PNG', width=150)
        
    else:
        if page == 'Scatter Plot Stats':
            
            st.markdown("""___""")
            
            st.write(':soccer: Performance analysis using a scatter diagram \
                     of the teams in the 5 major leagues based on selected \
                     statistics.')
            
            competitions = list(df['Competition'].unique())
            
            competition = st.multiselect('Competitions',
                                         options = competitions, 
                                         default = competitions)
            
            st.write(':bar_chart: Default scatter diagrams are proposed for each \
                     category of metrics selected, as well as the option of \
                     making the graphs themselves by manually selecting the \
                     metrics to compare.')
            
            dashboard = st.selectbox(
                    'Graphic option', 
                    ['Predefined', 'Customized'])
            
            categories = ['All Stats', 'Standard Stats', 'Goalkeeping', 'Shooting', 
                          'Passing', 'Goal and Shot Creation', 'Defensive Actions', 
                          'Possession', 'Miscellaneous Stats']
            
            category_dict = {'Standard Stats': ['Poss%', 'Gls', 'Ast', 'xG', 'xA', 
                                                'Gls/90', 'Ast/90', 'xG/90', 'xA/90', 
                                                'SoT/G', 'SoTA/GA'], 
                             'Goalkeeping': ['GA', 'OG', 'GA/90', 'Save%', 'CS%', 
                                             'SoTA/90', 'PSxG', 'PSxG/SoT', 'xGA/90', 
                                             'PSxG+/-', 'SoTA/GA'], 
                             'Shooting': ['Gls', 'Sh', 'Sh/90', 'SoT/90', 'xG', 
                                          'xG/90', 'SoT/G', 'Gls-xG'],
                             'Passing': ['Passes%', 'ShortPasses%', 'MediumPasses%', 
                                         'LongPasses%', 'PassesCompleted/90', 
                                         'PassesAttempted/90', 'ShortPassesCompleted/90', 
                                         'MediumPassesCompleted/90', 
                                         'LongPassesCompleted/90', 
                                         'TotDistPasses/90',
                                         'PrgDistPasses/90','KP/90', 
                                         'FinalThirdPasses/90', 'PPA/90', 
                                         'CrsPA/90'], 
                             'Goal and Shot Creation': ['Gls/90', 'Sh/90',
                                                        'SCA/90', 'GCA/90'], 
                             'Defensive Actions': ['Press%', 'Tkl/90', 'Blocks/90', 
                                                   'Int/90', 'Err/90',], 
                             'Possession': ['Poss%', 'Touches/90', 'Dribbles%', 
                                            'Gls/90', 'Points/90'],
                             'Playing Time': ['xG/90', 'xGA/90', 'Points/90'],
                             'Miscellaneous Stats': ['Aerial%', 'Fls/90', 
                                                     'Int/90', 'Tkl/90', 
                                                     'Recov/90']}
                             
            customized_stats = {'Standard Stats': ['SoT/G', 'SoTA/GA'], 
                                'Goalkeeping': ['GA', 'PSxG+/-'], 
                                'Shooting': ['Gls', 'G-xG'], 
                                'Passing': ['FinalThirdPasses/90', 'KP/90'], 
                                'Goal and Shot Creation': ['GCA/90', 'Gls/90'], 
                                'Defensive Actions': ['Press%', 'Int/90'], 
                                'Possession': ['Poss%', 'Points/90'], 
                                'Playing Time': ['Points/90', 'xG/90'],
                                'Miscellaneous Stats': ['Fls/90', 'Tkl/90']}
                             
            
            if dashboard == 'Customized':
        
                category_stats = st.selectbox('Category', categories)
                
                if category_stats == 'All Stats':
                    stats_selection = list(df.columns[2:])
                else:
                    stats_selection = category_dict[category_stats]
                
                st.write(':mag: Select metrics of the previously \
                         selected category to observe the performance of the clubs \
                         of the chosen competitions. The first metric selected will \
                         be the X axis and the second will be the Y axis.')
            
                stats = st.multiselect(
                        'Metrics', options = stats_selection)
                
                if len(stats) == 2:
                    
                    with st.beta_expander("Graphic Information"):
                        
                        st.write(':chart_with_downwards_trend: The dashed line \
                                 on each axis indicates the mean value of each metric.')
                        
                        st.write(':chart_with_upwards_trend: A linear trendline \
                                 is a best-fit straight line that usually shows \
                                 that something is increasing or decreasing \
                                 at a steady rate. r2 metric expresses \
                                 what fraction of the variability of your \
                                 dependent variable (Y) is explained \
                                 by your independent variable (X).')
                    
                        select_tendency = st.checkbox('View Linear Trendline')  
            
                    st.plotly_chart(
                        scatterplot_stats(df, stats, competition, 
                                          dict_description_metrics, 
                                          select_tendency, "", "", False), 
                        use_container_width=True)
                    
            else:
                
                category_stats = st.selectbox('Category', categories[1:])
                
                colors = {'Standard Stats': ['darkorange', 'red', 'green', 'gold'], 
                          'Goalkeeping': ['darkorange', 'red', 'green', 'gold'], 
                          'Shooting': ['red', 'darkorange', 'gold', 'green'], 
                          'Passing': ['red', 'darkorange', 'gold', 'green'],
                          'Goal and Shot Creation': ['red', 'darkorange', 'gold', 'green'],
                          'Defensive Actions': ['red', 'darkorange', 'gold', 'green'],
                          'Possession': ['red', 'darkorange', 'gold', 'green'], 
                          'Miscellaneous Stats': ['darkorange', 'red', 'green', 'gold']}
                
                minus_values = {'Standard Stats': 0.5, 
                                'Goalkeeping': 1,
                                'Shooting': 2, 
                                'Passing': 1,
                                'Goal and Shot Creation': 0.2,
                                'Defensive Actions': 0.5,
                                'Possession': 0.5, 
                                'Miscellaneous Stats': 0.5}
                
                with st.beta_expander("Graphic Information"):
                    
                    st.write(':chart_with_downwards_trend: The dashed line \
                             on each axis indicates the mean value of each metric.')
                        
                    st.write(':chart_with_upwards_trend: A linear trendline \
                             is a best-fit straight line that usually shows \
                             that something is increasing or decreasing \
                             at a steady rate. r2 metric expresses \
                             what fraction of the variability of your \
                             dependent variable (Y) is explained \
                             by your independent variable (X).')
                             
                    st.write(':art: We divide the graph into 4 zones based on \
                             the average of the two variables and we give each \
                             quadrant a color based on the value of the two variables \
                             indicating the performance of the team.')
                    st.image('img/escala2.PNG', width=150)
                    
                    select_tendency = st.checkbox('View Linear Trendline')
                
                st.plotly_chart(scatterplot_stats(
                        df, customized_stats[category_stats], 
                        competition, 
                        dict_description_metrics, select_tendency, 
                        colors[category_stats], minus_values[category_stats], True), 
                    use_container_width=True)
                
        else:
                        
            st.markdown("""___""")
            
            if page == 'Heatmap Zones Stats':
                
                st.write(':chart_with_upwards_trend: Analysis of the distribution and location, for the selected team, \
                         of the metrics: takles, pressures and touches, by area of the \
                        field (Z1, Z2, Z3) and their comparison (percentile calculation) \
                        with the rest of the teams in the selected competition.')
                
                df_zone = load_data('data/FBREF_clubes2020_zones.csv', encoding='latin')
                
                competition_zones = st.selectbox(
                    'Competition', list(df_zone['Competition'].unique()))
                
                df_zone_competition = df_zone[df_zone.Competition == competition_zones]
                
                team_zones = st.selectbox('Team', list(df_zone_competition['Squad']))
                
                with st.beta_expander("Graphic Information"):
                    
                    st.write('For each metric:')
                    st.write(':100: The percentage in each zone indicates the \
                               percentage of team actions that occurred in that zone \
                               with respect to the total.')
                    st.write(':art: The colors by zone indicate the percentile with \
                              respect to the rest of the teams of the value of \
                              the metric, by 90 minutes, in the zone.')
                    st.image('img/escala.PNG', width=150)

                for metric_zone in ['Takles', 'Pressure', 'Touches']:
                    df_zone_metric = df_zone_competition[
                        ['Squad'] + [c for c in df_zone_competition.columns 
                                     if metric_zone in c]]
                    st.pyplot(plot_zones(df_zone_metric, team_zones, metric_zone))

                
            else:
                
                if page == 'Similar Teams':
                
                    st.write(':bulb: **Similar Team Search.** \
                         A team is selected as the subject of the search and \
                        the algorithm then produces a list of teams with similar \
                        statistical profiles, ranked on a scale of 0-100, with \
                        100 being an exact match.')
                    st.write(':bulb: **Similarity Algorithms.** \
                         Algorithms that group teams from selected \
                         competitions so that teams from the same group have \
                         similar metrics.')
                         
                    radio_option = st.radio('Algorithm', 
                                        ['Similar Team Search', 'Clustering Teams'])
                         
                    if radio_option == 'Similar Team Search':
                    
                        team_similar = st.selectbox('Team', df.Squad)
                        df_team_similar = df[df.Squad == team_similar]
                    
                        competitions_similar = st.multiselect(
                            'Competitions', 
                            options = list(df.Competition.unique()), 
                            default = ['La Liga', 'Premier League', 
                                       'Bundesliga', 'Serie A', 'Ligue 1'])
                    
                        df_similar_comp = df[df.Competition.isin(competitions_similar)]
                        df_similar_comp = pd.concat([df_team_similar, 
                                                     df_similar_comp], 
                                                    axis=0)
                        df_similar_comp.drop_duplicates(inplace=True)
                        df_similar_comp.reset_index(drop=True, inplace=True)
                        
                        with st.beta_expander("Swarmplot"):
                            metric_swarmplot = st.selectbox(
                                'Metric', list(df_similar_comp.columns[2:]))
                            swarmplot_graph = swarmplot(
                                df_similar_comp, metric_swarmplot, team_similar, 
                                dict_description_metrics)
                            st.plotly_chart(swarmplot_graph)
                    
                        n_similars = st.slider(
                            'Number of similar teams', 1, 10, 7)
                    
                        metrics_selection = st.selectbox(
                            'Metrics', ['Filtered', 'All metrics'])
                    
                        if metrics_selection == 'Filtered':
                            metrics_similar = st.multiselect(
                                '', options = list(df_similar_comp.columns[2:]), 
                                default = ['Gls/90', 'SoT/G', 'PSxG+/-',
                                           'Ast/90', 'GCA/90', 'KP/90', 'PPA/90', 
                                           'Press%', 'Passes%', 'Poss%', 'Tkl/90', 
                                           'Aerial%', 'xGA/90', 'Err/90', 'Fls/90', 
                                           'Dribbles%'])
                            df_metrics_filtered = df_similar_comp[['Squad','Competition'] \
                                                                  + metrics_similar]
                        else:
                            metrics_similar = list(df_similar_comp.columns[2:])
                            df_metrics_filtered = df_similar_comp
                            
                        with st.beta_expander("Graphic Information"):
                            st.write(':art: We will use the following color \
                                     scale for each selected statistic.')
                            st.image('img/escala1.PNG', width=300)
                            select_info = st.checkbox(
                                'Full name of the metric in the table')   
                    
                        df_similar_t, html_tabla = similar_team(
                            df_metrics_filtered, team_similar, n_similars, 
                            select_info, dict_description_metrics)
                    
                        raw_html = html_tabla._repr_html_()
                        components.html(raw_html, height=600, scrolling=True)
                            
                        
                        # plot radar chart
                        if metrics_selection == 'Filtered':
                            with st.beta_expander("Radar Comparision", 
                                                  expanded=False):
                                df_similar_t = df_similar_t.sort_values(
                                    by='% Similarity', ascending=False)

                                teams_radar = df_similar_t['Squad'].tolist()
                                df_plot_similar = df_metrics_filtered[
                                    df_metrics_filtered.Squad.isin(teams_radar)]
                                df_plot_similar.reset_index(drop=True, inplace=True)
                    
                                ranges = {c: (np.percentile(
                                        df_metrics_filtered[c].values, 5), 
                                    np.percentile(
                                        df_metrics_filtered[c].values, 95)) 
                                          for c in df_metrics_filtered.columns[2:]}
                        
                                plot_radar_similar, _ = radarchart_pyplot(
                                    df_plot_similar, df_plot_similar['Squad'].tolist(), 
                                    metrics_similar, ranges, False)
                                st.pyplot(plot_radar_similar)
                    
                    else:
                    
                        st.write(':books: Algorithm that groups similar teams into \
                             groups called clusters. The endpoint is a set of \
                             clusters, where each cluster is distinct from each \
                             other cluster, and the teams within each cluster\
                             are broadly similar to each other.')
                
                        competitions = list(df['Competition'].unique())
            
                        competition = st.multiselect('Competitions',
                                                     options = competitions, 
                                                     default = 'La Liga')
                
                        algorithm = st.selectbox('Algorithm', 
                                                 ['Hiearchical Clustering', 
                                                  'KMeans Clustering + PCA'])
                                     
                        if algorithm == 'Hiearchical Clustering':
                    
                            df_dendogram = df[df.Competition.isin(competition)]
                    
                            st.write(':pencil: **Cophenetic Correlation Coefficient.** \
                                 It is used to compare the results of clustering the \
                                 same dataset using different distance calculation methods \
                                 that compute the distance between two clusters. \
                                 The closer the value is to 1, the more accurately \
                                 the clustering solution reflects your data.')
                    
                            hiearchical_fig, hiearchical_distance = \
                                hiearchical_clustering(df_dendogram)
                            hiearchical_distance.index = list(
                                range(1,len(hiearchical_distance)+1))
                            st.table(hiearchical_distance)
                            st.plotly_chart(hiearchical_fig, 
                                            use_container_width=True)
                    
                        else:
                            if algorithm == 'KMeans Clustering + PCA':
                        
                                df_clustering = df[df.Competition.isin(competition)]
                        
                                st.write(':books: **Princpal Component Analysis.** Technique\
                                     used to describe a data set in terms of new \
                                     uncorrelated variables (Dimensionality Reduction).')
                                st.write(':memo: **KMeans.** Clustering Method that aims to \
                                     partition a set of *n* observations into *k* groups \
                                     in which each observation belongs to the group \
                                     whose centroid is closest.')
                        
                                pca_method = st.selectbox(
                                        'Method to obtain optimal number of CPs', 
                                        ['Eigenvalues above the mean', 
                                         '% of explained variance accumulated'])
                        
                                if pca_method == '% of explained variance accumulated':
                            
                                    pct_variance = st.slider(
                                            'Maximum % of accumulated variance \
                                            explained by the CPs', 
                                            50, 100, 90)
                                else:
                                   pct_variance = ""
                                   
                                if st.checkbox('Manual selection of the number of clusters'):
                                    manual_selection = True
                                    n_clusters = st.slider(
                                        'Number of clusters', 2, 7, 5)
                                else:
                                    manual_selection = False
                                    n_clusters = None
                                
                                st.plotly_chart(
                                    pca_kmeans_clustering(df_clustering, 
                                                          pca_method, pct_variance, 
                                                          manual_selection, n_clusters), 
                                    use_container_width=True)
                        
                                st.write(':bulb: We can observe the graphical procedure \
                                         to understand the optimal number of selected CPs.')
                                
                                with st.beta_expander(
                                        "Principal Component Analysis (PCA) Process",
                                        expanded=False):
                                    st.plotly_chart(
                                            plotly_PCA(
                                                df_clustering, pca_method, 
                                                pct_variance), 
                                            use_container_width=True)

                                
                        
                    
            
if __name__ == "__main__":
    main()
    
st.markdown("""___""")  
st.write(':information_source: **Data Sources**')    
st.markdown(body="""
            > **Data Collection** \
            <a href="https://fbref.com/en/"> FBref</a>""", unsafe_allow_html=True)
st.markdown(body="""
            > **Contact** \
            <a href="https://www.linkedin.com/in/javier-fernandez-rodriguez/"> \
            Linkedin</a> \
            <a href="https://twitter.com/javferrod96"> Twitter</a>""", unsafe_allow_html=True)
