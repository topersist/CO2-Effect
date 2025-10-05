# -*- coding: utf-8 -*-
"""
CO2 Rise Directly Impairs Crop Nutritional Quality
Code S1: Main Code

By S.F. ter Haar*, P.M. van Bodegom, and L. Scherer
Institute of Environmental Sciences (CML), Leiden University, Leiden, The Netherlands
*email: s.f.ter.haar@cml.leidenuniv.nl

The data and code that support the findings of this study are openly available 
in Zenodo at http://doi.org/10.5281/zenodo.17271748

Created on Aug 18 2023
Updated Oct 5 2025

@author: SF ter Haar
"""
# %% import required libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools

#%% imports needed files to run the script
# input main database
dfOriginal = pd.read_csv('CO2_Rise_Crop_Quality_SI_Data_S1_Database.csv', 
                         header=7, index_col='Common Name')

# if you have the file saved locally, point to it here
# world = gpd.read_file("ne_110m_admin_0_countries.zip")
# world = world[['NAME', 'POP_EST', 'CONTINENT', 'ISO_A3', 'GDP_MD', 'geometry']]

# to be used if you don't have the file above (requires internet connection)
world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

# %% cleans the input database

df = dfOriginal.copy()

df.rename(columns={'Study Type': 'studytype', 
                   'Plot Type': 'plottype', 
                   'Photosynthetic Pathway': 'c3c4',
                   'Ambient CO2': 'aco2',
                   'Elevated CO2': 'eco2'}, 
          inplace = True)

# standardized responses to baseline ambient (aco2 = 350 ppm) and elevated 
# (eco2 = 550 ppm) CO2 level
df['r_b_interp'] = df['eco2'] - df['aco2'] + (350 - df['aco2']) * df['delta']
df['r_g_interp'] = df['eco2'] - df['aco2'] + (550 - df['aco2']) * df['delta']

# this is declared to accomodate values that give a negative ln
# solves this by adjusting to a +200 ppm CO2 (eco2 - aco2)
df['alogr'] = np.log(
    (df['eco2'] - df['aco2'] + (200 * df['delta'])) / (df['eco2']-df['aco2'])) 
df.loc[df['r_b_interp'] >= 0, 'alogr'] = np.log(
    df.loc[df['r_b_interp'] >= 0, 'r_g_interp'] 
    / df.loc[df['r_b_interp'] >= 0, 'r_b_interp'])

df['tissueAgg'] = 'b'
# classify plant parts
for com_name in df['Tissue'].unique():
    if (com_name == 'seed' 
        or com_name == 'fruit' 
        or com_name == 'fruit juice' 
        or com_name == 'pod' 
        or com_name == 'grain'):
        df.loc[df.Tissue==com_name, ['tissueAgg']] = 'reproductive'
    elif (com_name == 'root' or com_name == 'tuber'):
        df.loc[df.Tissue==com_name, ['tissueAgg']] = 'BG'
    elif (com_name == 'shoot' or com_name == 'stem'):
        df.loc[df.Tissue == com_name, ['tissueAgg']] = 'AG'
    else:
        print('Alert: unclassified plant part: ', com_name)

df['studyplottype'] = df['studytype'] + ' ' + df['plottype']
#%% EDA - numbers used in the manuscript
print(f"""
Statistics - Abstract
---------------------
{len(dfOriginal)} entries
{dfOriginal['n'].sum()} pairs
{dfOriginal.index.nunique()} crops
{dfOriginal['element'].nunique()} nutrients
""")

print(f"""
Statistics - Methods: Database Creation
---------------------
Non-snowball datasets:
    {len(dfOriginal.loc[dfOriginal['Origin'] != 'snowball'])} entries
    {dfOriginal.loc[dfOriginal['Origin'] != 'snowball']['n'].sum()} pairs
Snowball datasets:
    {len(dfOriginal.loc[dfOriginal['Origin'] == 'snowball'])} entries
    {dfOriginal.loc[dfOriginal['Origin'] == 'snowball']['n'].sum()} pairs
""")

print(f"""
Statistics - Methods: Response Rate Linearization
---------------------
Ambient CO\u2082 level: 
    Mean: {np.round(df['aco2'].mean(), 1)} ppm 
    Range: {np.round(df['aco2'].min(), 1)} ppm to {np.round(df['aco2'].max(), 1)}
    Largest peak at {np.round(100*dfOriginal['Added CO2'].value_counts()
                              .nlargest(1)).index[0]} ppm
""")

print(f"""
Statistics - Results: Unprecedented data coverage\n---------------------
{len(dfOriginal)} entries from {dfOriginal['Reference'].nunique()} articles
\nEntry contribution [%] by  
{np.round(dfOriginal["Reference"].value_counts()
          .nlargest(5) / len(dfOriginal) * 100, 1).to_string()}

Dietterich dataset contains 5 separate studies, which individually
make up [%] of dataset
{np.round(100 * dfOriginal.loc[dfOriginal['Origin'] == 'Dietterich']
          .index.value_counts() / len(dfOriginal), 1).to_string()}

{dfOriginal['element'].nunique()} nutrients. These are the most well studied [%]:
{np.round(df['element'].value_counts()
          .nlargest(5) / len(df) * 100, 1).to_string()}

Per: {np.round(df["c3c4"].value_counts() / len(df) * 100, 1).to_string()}

Per {np.round(df["studytype"].value_counts() / len(df) * 100, 1).to_string()}
""")

print(f"""
FACE+OTC is conducted in {dfOriginal['Country'].nunique()} countries
{df.index.nunique()} crops representing {df['Cultivar'].nunique()} cultivars \
from {df['Genus'].nunique()} genera \
contained in {df['Family'].nunique()} families

Most studied crops \n {np.round(100*df.index.value_counts()
                                .nlargest(5)/len(df), 1).to_string()}
""")

print(f"""Statistics - Results: Database Summary Table\n---------------------
{len(dfOriginal)} entries
{dfOriginal['n'].sum()} pairs
{np.round(df["c3c4"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["plottype"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["studytype"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["studyplottype"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["Country"].value_counts()/df["Country"].value_counts()
               .sum()*100, 1).to_string()}\n
{np.round(df["element"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["Tissue"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["tissueAgg"].value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df.index.value_counts()/len(df)*100, 1).to_string()}\n
{np.round(df["Family"].value_counts()/len(df)*100, 1).to_string()}
""")
#%% Reorganized dataframe for later use
# creates the combined proxy element category to include both N and protein, 
# but not together for the same sample
df_filt = pd.DataFrame()
for source in df['Reference'].unique(): 
    a=df.loc[df['Reference']==source]
    for com_name in a['Scientific Name'].unique():
        b=a.loc[a['Scientific Name']==com_name]
        if 'protein' in b['element'].unique():
            if 'N' in b['element'].unique():
                df_filt = pd.concat([df_filt, b.loc[b['element']=='N']], 
                                    ignore_index=False)
            else:
                df_filt = pd.concat([df_filt, b.loc[b['element']=='protein']], 
                                    ignore_index=False)
        else:
            if 'N' in b['element'].unique():
                df_filt = pd.concat([df_filt, b.loc[b['element']=='N']], 
                                    ignore_index=False)
df_filt['element'] = 'N proxy'
df = pd.concat([df, df_filt])

# separates out rice due to vastly different growing condition
df.loc[df.Genus=='Oryza', ['tissueAgg']] = 'reproductive - rice'
df.loc[df.Genus=='Oryza', ['Tissue']] = 'rice grain'

df = df.reset_index()
#%% funnel plot (Figure S1)
fig = plt.figure(figsize=(7, 5), dpi=600)
sns.scatterplot(data=df, x="alogr", y="n", color="#003473", alpha=0.3)
plt.axvline(x=df['alogr'].mean(), c="black", label="mean")
plt.ylabel('sample size [n]', fontsize=14)
plt.xlabel('effect size of adjusted log response', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('Figure S1.pdf', format='pdf', bbox_inches='tight', pad_inches = 0)
#%% comparing aCO2 and eCO2 distribution (Figure S2)
fig = plt.figure(figsize=(7, 5), dpi=600) 
gs = GridSpec(2, 2, height_ratios=[4, 0.3], width_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])
norm = plt.Normalize(dfOriginal["Year Published"].min(), 
                     dfOriginal["Year Published"].max())
scatter_plot = sns.scatterplot(data=dfOriginal, 
                               x="Ambient CO2", 
                               y="Elevated CO2", 
                               alpha=0.5,
                               size="n",
                               hue="Year Published", 
                               palette="viridis",
                               legend=False,
                               ax=ax1)
ax1.set_xlabel('ambient CO$_2$ [ppm]', fontsize=14)
ax1.set_ylabel('elevated CO$_2$ [ppm]', fontsize=14)
ax1.set_title('(a)', loc='left', weight='bold', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([]) 
cbar_ax = fig.add_subplot(gs[1, 0])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Year Published", fontsize=14)
cbar.ax.tick_params(labelsize=14)

ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(data=dfOriginal, x="Added CO2", color="#003473", ax=ax2)
ax2.set_xlabel('added CO$_2$ [ppm]', fontsize=14)
ax2.set_ylabel('count', fontsize=14)
ax2.set_title('(b)', loc='left', weight='bold', fontsize=14)
ax2.tick_params(axis='both', labelsize=14)

plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.savefig('Figure S2.pdf', format='pdf', bbox_inches='tight', pad_inches = 0)
#%%e world map of study locations (Figure S3)
df['Longitude'] = np.round(df['Longitude'], 0)
df['Latitude'] = np.round(df['Latitude'], 0)

location_counts = df.groupby(['Longitude', 
                              'Latitude']).size().reset_index(name='counts')

world = world[(world.POP_EST > 0) & (world.NAME != "Antarctica")]

gdf = gpd.GeoDataFrame(location_counts, 
    geometry=gpd.points_from_xy(location_counts.Longitude, 
                                location_counts.Latitude))

fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=600)
world.plot(ax=ax, color='lightgrey')
gdf.plot(ax=ax, markersize=gdf['counts']*1, color='#003473', alpha=0.5)
plt.xlim([-180, 180])
ax.tick_params(axis='both', labelsize=14)
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('Figure S3.pdf', format='pdf', bbox_inches='tight', pad_inches = 0)
#%% plot data over time by family (Figure 1)
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=600)
df['C-type Family']=df['c3c4']+" "+df['Family']
sns.histplot(
    data=df.sort_values(by=['Year Published']), 
    x="Year Published", 
    hue="C-type Family", 
    element="step", 
    multiple="stack", 
    cumulative=True, 
    stat="count")
sns.move_legend(ax, "upper left", fontsize=11)
plt.xlabel('Year Published', fontsize=14)
plt.ylabel('cumulative number of entries', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
plt.xlim([min(dfOriginal['Year Published']), max(dfOriginal['Year Published'])])
plt.savefig('Figure 1.pdf', format='pdf', bbox_inches='tight', pad_inches = 0.1)
#%% clean up columns post-EDA 
# drop unneeded columns post EDA
df = df.drop(columns=['Scientific Name', 
                      'Additional Info', 
                      'Elevated temperature', 
                      'Irrigation', 
                      'Sowing Time', 
                      'Phosphorous application', 
                      'Nitrogen application', 
                      'Ozone', 
                      'Reference', 
                      'Origin', 
                      'Year', 
                      'Year Published'])

# recategorize study types 
for com_name in df['studytype'].unique():
    if (com_name=='FACE' or com_name=='OTC' or com_name=='tunnel'):
        df.loc[df['studytype']==com_name, ['studytypeSimple']] = 'outdoor'
    elif (com_name=='chamber' or com_name=='greenhouse'):
        df.loc[df['studytype']==com_name, ['studytypeSimple']] = 'indoor'
    else:
        print('Alert: unclassified studytype: ', com_name)
        
df['studyplottypeSimple'] = df['studytypeSimple'] + ' ' + df['plottype']

dfW=df.set_index(['Common Name',
                  'Family', 
                  'Genus',
                  'Species',
                  'Cultivar',
                  'element',
                  'Tissue',
                  'studytype',
                  'plottype',
                  'c3c4',
                  'eco2',
                  'aco2',
                  'Added CO2',
                  'delta',
                  'alogr',
                  'tissueAgg',
                  'studytypeSimple',
                  ])['n'].repeat(df['n']).reset_index()
#%% bootstrapping
def powerBootstrap(x, sdp, n_bootstraps):
    """
    Runs a bootstrap analysis of the inputted dataframe and returns relevant
    statistics. Tests if the mean is different from 0 in a two-sided test.
    Take care to include only the response variable when passing to the 
    function. alpha and beta are 0.05. 
    
    If the dataset is small, its variance can be << the population variance, 
    resulting in an overestimation of power. This is solved using an a priori 
    estimate of the population variance. In our case, the standard deviation 
    of the entire dataset. 
    
    Parameters:
        df (array of floats): 
            data array containing the response values for the desired dataset
        delta (float): 
            the standard deviation of the entire dataset
        n_bootstraps (int): 
            number of bootstraps taken (recommended: 10,000)
            
    Returns:
        mean (float): 
            the mean of the bootstrapped interval
        ci_bootstrap (array): 
            95% confidence interval of the bootstrapped interval
        power (float): 
            the power of the bootstrapped interval
        pval (float): 
            the p-value using a two-sided z-test
        m (int): 
            number of samples used in calculation
        
    Examples:
          >>> [mean, CI, pow, p, m] = powerBootstrap(df['response'])
          returns the mean, confidence interval, power, p-value, and number
          of samples of the bootstrapped df
          
    Source: This function is an adapted translation of the function "pwr.boot" 
    from Irakli Loladze (2014) in  eLife doi: 10.7554/eLife.02245 (CC0 license)
    URL: http://github.com/loladze/co2.git
    """
    
    small = 20
    delta = 0.05
    nsamples = len(x)

    # standard deviation, uses that of whole dataset for small samples
    sdx = max(stats.tstd(x), sdp) if nsamples < small else stats.tstd(x)

    # Vectorized shift
    xshift = x + delta * (stats.tstd(x) / sdx)

    # Vectorized bootstrap indices:
    idx = np.random.randint(0, nsamples, size=(n_bootstraps, nsamples))

    # Bootstrap samples and means
    x_strap = x[idx]
    x_strap_mean = x_strap.mean(axis=1)

    x_strap_shift = xshift[idx]
    x_strap_shift_mean = x_strap_shift.mean(axis=1)

    mean = x_strap_mean.mean()

    # Use robust empirical CI
    if stats.kstest(x_strap_mean, 'norm')[1] < 0.05:
        # sample data is not normally distributed, so use percentiles
        ci_bootstrap = np.percentile(x_strap_mean, [2.5, 97.5])
    else:
        ci_bootstrap = stats.norm.interval(0.95, loc=np.mean(x_strap_mean), 
                                           scale=100*stats.sem(x_strap_mean))

    # Calculate power: proportion outside CI
    power = np.mean((x_strap_shift_mean < ci_bootstrap[0]) | 
                    (x_strap_shift_mean > ci_bootstrap[1]))

    # Two-tailed p-value
    shift_diff = x_strap_mean - mean
    tester2a = np.sum(shift_diff >= abs(mean))
    tester2b = np.sum(shift_diff <= -abs(mean))
    pval = min((tester2a + tester2b) / n_bootstraps, 1)

    return mean, ci_bootstrap, power, pval, nsamples

# bootstrapping
# calculates standard deviation for whole dataset, used for small samples
sdp = stats.tstd(dfW['delta'])

hierarchy_levels = ['c3c4', 'Family', 'Genus', 'Species', 'Cultivar']
tissues = ['Tissue', 'tissueAgg']
studies = ['studytype', 'studytypeSimple']
extra_vars = ['element', 'c3c4', 'studytype', 'studytypeSimple', 'plottype', 
              'Tissue', 'tissueAgg', 'Family', 'Genus', 'Common Name']

results_list = []
grouping_combos = []

grouping_combos.append(None) #entire dataframe
grouping_combos.append(['studytype', 'plottype']) 
grouping_combos.append(['studytypeSimple', 'plottype']) 

for var in extra_vars:
    if var == 'element':
        grouping_combos.append(['element']) 
    else:
        grouping_combos.append([var]) # individual class
        grouping_combos.append([var, 'element']) # class + element

for i in range(1, len(hierarchy_levels) + 1): # hierarchical + combos
    main = hierarchy_levels[:i]
    grouping_combos.append(main + ['element'])
    grouping_combos.append(main + ['element', 'Tissue'])
    grouping_combos.append(main + ['element', 'tissueAgg'])
    for tissue_col, study_col in itertools.product(tissues, studies):
        grouping_combos.append(main + ['element', 'plottype', tissue_col, 
                                       study_col])

for group_cols in grouping_combos:
    if group_cols is None:
        group_data = dfW.copy()

        if len(group_data) > 1 and group_data['delta'].nunique() > 1:
            ans = powerBootstrap(group_data['alogr'].values, sdp, 10000)
            results_list.append({
                'mean': np.round(ans[0], 4),
                '2.5': np.round(ans[1][0], 4),
                '97.5': np.round(ans[1][1], 4),
                'power': ans[2],
                'p-value': ans[3],
                'm': ans[4],
                'class': 'entire_dataset'
            })
        continue

    if 'Cultivar' in group_cols:
        # ensures that cultivar results are only given for species with data
        # covering multiple cultivars
        df_filtered = dfW.groupby('Common Name').filter(lambda x: x['Cultivar'].nunique() > 1)
    else:
        df_filtered = dfW

    grouped = df_filtered.groupby(group_cols)

    for group_keys, group_data in grouped:
        if len(group_data) > 1 and group_data['delta'].nunique() > 1:
            ans = powerBootstrap(group_data['alogr'].values, sdp, 10000)

            group_dict = {}

            for col, val in zip(group_cols, group_keys):
                if col in tissues:
                    group_dict['tissue'] = val
                elif col in studies:
                    group_dict['studytype'] = val
                else:
                    group_dict[col] = val

            group_dict.update({
                'mean': np.round(ans[0], 4),
                '2.5': np.round(ans[1][0], 4),
                '97.5': np.round(ans[1][1], 4),
                'power': ans[2],
                'p-value': ans[3],
                'm': ans[4],
                'class': '+'.join(group_cols)
            })

            results_list.append(group_dict)

bresults = pd.DataFrame(results_list)

bresults['2.5%'] = np.round(100*(np.exp(bresults['2.5'])-1), 4)
bresults['mean%'] = np.round(100*(np.exp(bresults['mean'])-1), 4)
bresults['97.5%'] = np.round(100*(np.exp(bresults['97.5'])-1), 4)

extra_text = [
    "CO2 Rise Directly Impairs Crop Nutritional Quality\n",
    "Data S3: Bootstrapping Results, see Data S4 for details\n\n"
    '"By S.F. ter Haar*, P.M. van Bodegom, and L. Scherer"\n',
    '"Institute of Environmental Sciences (CML), Leiden University, Leiden, The Netherlands"\n',
    "*email: s.f.ter.haar@cml.leidenuniv.nl\n\n",
]

filename = "CO2_Rise_Crop_Quality_SI_Data_S3_Results.csv"

with open(filename, "w") as f:
    f.writelines(extra_text)

bresults.to_csv(filename, mode="a", index=False)
#%% bootstrapping results plotted against power (Figure S4)
# Copying the DataFrame
bplot = bresults.copy()

# Plotting
fig = plt.figure(figsize=(7, 7), dpi=600)

bplot['mean%'] = 100*(np.exp(bplot['mean'])-1)
bplot['97.5%'] = 100*(np.exp(bplot['97.5'])-1)
bplot['2.5%'] = 100*(np.exp(bplot['2.5'])-1)

bplot['significance'] = 'no significant change'

plt.subplot(1,2,1)
# Plotting lines with conditional colors
for i in range(len(bplot)):
    color = 'grey'  # Default color
    
    # Check conditions for line color
    if bplot['mean%'].iloc[i] < 0 and bplot['p-value'].iloc[i] <= 0.05:
        color = "#E04A48"
        bplot.loc[i, 'significance'] = 'decrease'
    elif bplot['mean%'].iloc[i] > 0 and bplot['p-value'].iloc[i] <= 0.05:
        color="#003473"
        bplot.loc[i, 'significance'] = 'increase'
    
    plt.plot([bplot['mean%'].iloc[i], bplot['2.5%'].iloc[i], bplot['97.5%'].iloc[i]], 
             [bplot['power'].iloc[i], bplot['power'].iloc[i], bplot['power'].iloc[i]], 
             '-', c=color, zorder=1)

sns.scatterplot(data=bplot, x="mean%", y="power", 
                hue="significance", 
                palette=["#E04A48", "grey", "#003473"], 
                legend=True)

plt.legend(bbox_to_anchor=(1.1, -0.2), loc='lower center', ncol=3, fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel('power', fontsize=14)
plt.xlabel('% change', fontsize=14)
plt.title('(a)', loc='left', weight='bold', fontsize=14)
plt.ylim([-0.015, 1.015])
plt.axhline(y=0.4, linestyle='--', c="grey")
plt.axhline(y=0.8, linestyle='--', c="grey")
plt.axvline(x=0, c="black", label="x=0")

plt.subplot(1,2,2)
# Plotting lines with conditional colors
for i in range(len(bplot)):
    color = 'grey'  # Default color
    
    # Check conditions for line color
    if (bplot['mean%'].iloc[i] < 0 
        and bplot['2.5%'].iloc[i] < 0 
        and bplot['97.5%'].iloc[i] < 0):
        color = "#E04A48"
        bplot.loc[i, 'significance'] = 'decrease'
    elif (bplot['mean%'].iloc[i] > 0 
          and bplot['2.5%'].iloc[i] > 0 
          and bplot['97.5%'].iloc[i] > 0):
        color = "#003473"
        bplot.loc[i, 'significance'] = 'increase'
    
    plt.plot([bplot['mean%'].iloc[i], bplot['2.5%'].iloc[i], bplot['97.5%'].iloc[i]], 
             [bplot['power'].iloc[i], bplot['power'].iloc[i], bplot['power'].iloc[i]], 
             '-', c=color, zorder=1)

sns.scatterplot(data=bplot, x="mean%", y="power", hue="significance",
                palette=["#E04A48", "grey", "#003473"], legend=None)

plt.ylabel(' ')
plt.xlabel('% change', fontsize=14)
plt.title('(b)', loc='left', weight='bold', fontsize=14)
plt.ylim([-0.015, 1.015])
plt.xlim([-40, 40])
plt.tick_params(left=False, right=False , labelleft=False, labelsize=14) 
plt.axhline(y=0.4, linestyle='--', c="grey")
plt.axhline(y=0.8, linestyle='--', c="grey")
plt.axvline(x=0, c="black", label="x=0")
fig.tight_layout()

plt.savefig('Figure S4.pdf', bbox_inches='tight', pad_inches=0, format='pdf')
#%% response per major factor (Figure 2)
# per c3/c4
bp = bresults.copy()
bp = bp.loc[bp['power'] > 0.8]
bp = bp.loc[bp['m'] > 7].sort_values(by='mean%')

bsort = "c3c4"
bplot = bp.loc[bp['class']==bsort].copy()
fig = plt.figure(figsize=(6.8, 6.8), dpi=600)
gs = fig.add_gridspec(4,2)

plt.subplot(4,2,1)
bplot['group'] = bplot[bsort]

# Plot connecting lines
for index, row in bplot.iterrows():
    color = ('grey'
             if (row['p-value'] > 0.05 or row['power'] < 0.8) 
             else ('#E04A48' if row['mean'] < 0 else "#003473"))
    plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
             [row['group'], row['group'], row['group']], 
             '-', c=color, zorder=1)
    plt.scatter(row['mean%'], row['group'], color=color, marker='|')
    
plt.title('(a)', loc='left', weight='bold')
plt.yticks(fontsize=10)

# per element
fig.add_subplot(gs[0:2, 1])
bsort = "element"
bplot = bp.loc[bp['class']==bsort].copy()

handles = []
labels = []

# Plot connecting lines
for index, row in bplot.iterrows():
    linestyle = '-' if row['power'] > 0.8 else '--'
    if row['p-value'] > 0.05 or row['power'] < 0.8:
        color = 'grey'
        label = 'no significant change'
    elif row['mean'] < 0:
        color = '#E04A48'
        label = 'decrease'
    else:
        color = "#003473"
        label = 'increase'
    line, = plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                     [row['element'], row['element'], row['element']], 
                     linestyle, c=color, zorder=1)
    plt.scatter(row['mean%'], row['element'], color=color, marker='|', s=25)
    plt.rc('ytick', labelsize=5)
    
    # Collect legend handles and labels
    if label not in labels:
        handles.append(line)
        labels.append(label)
    
plt.title('(b)', loc='left', weight='bold')
plt.yticks(fontsize=8)
plt.figlegend(handles, labels, loc='lower center', 
              bbox_to_anchor=(0.5, 0.0), 
              ncol=3)

# per tissueAgg
plt.subplot(4,2,3)
bsort = "tissueAgg"
bplot = bp.loc[bp['class']==bsort].copy()
bplot['group'] = bplot['tissue']

# Plot connecting lines
for index, row in bplot.iterrows():
    if row['group'] == 'AG':
        row['group'] == 'above ground'
        row['group2']='above ground'
    elif row['group'] == 'BG':
        row['group2']='below ground'
    else:
        row['group2']=row['group']
    linestyle = '-' if row['power'] > 0.8 else '--'
    color = ('grey' 
             if (row['p-value'] > 0.05) 
             else ('#E04A48' if row['mean'] < 0 else "#003473"))
    plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
             [row['group2'], row['group2'], row['group2']], 
             linestyle, c=color, zorder=1)
    plt.scatter(row['mean%'], row['group2'], color=color, marker='|')
    
plt.title('(c)', loc='left', weight='bold')
plt.yticks(fontsize=10)

# per study type
bplot = bp.loc[bp['class']=="studytypeSimple"].copy()
plt.subplot(4,2,5)
bplot['group'] = bplot['studytype']

# Plot connecting lines
for index, row in bplot.iterrows():
    color = ('grey' 
             if (row['p-value'] > 0.05 or row['power'] < 0.8) 
             else ('#E04A48' if row['mean'] < 0 else "#003473"))
    plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
             [row['studytype'], row['studytype'], row['studytype']], 
             '-', c=color, zorder=1)
    plt.scatter(row['mean%'], row['studytype'], color=color, marker='|')
    
plt.title('(d)', loc='left', weight='bold')
plt.yticks(fontsize=10)

# per plot type
plt.subplot(4,2,6)
bplot = bp.loc[bp['class']=='plottype'].copy()
bplot['group']=bplot['plottype']

# Plot connecting lines
for index, row in bplot.iterrows():
    linestyle = '-' if row['power'] > 0.8 else '--'
    color = ('grey' 
             if (row['p-value'] > 0.05) 
             else ('#E04A48' if row['mean'] < 0 else "#003473"))
    plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
             [row['group'], row['group'], row['group']], linestyle
             , c=color, zorder=1)
    plt.scatter(row['mean%'], row['group'], color=color, marker='|')
    
plt.title('(e)', loc='left', weight='bold')
plt.yticks(fontsize=10)

# per study type
plt.subplot(4,2,7)
bsort = "studytype"
bplot = bp.loc[bp['class']==bsort].copy()
bplot['group']=bplot['studytype']

# Plot connecting lines
for index, row in bplot.iterrows():
    linestyle = '-' if row['power'] > 0.8 else '--'
    color = ('grey' if (row['p-value'] > 0.05) 
             else ('#E04A48' if row['mean'] < 0 else "#003473"))
    plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
             [row['group'], row['group'], row['group']], linestyle
             , c=color, zorder=1)
    plt.scatter(row['mean%'], row['group'], color=color, marker='|')
    
plt.xlabel('mean response [%]')
plt.title('(f)', loc='left', weight='bold')
plt.yticks(fontsize=10)

# per plot type
plt.subplot(4,2,8)
bsort = "studytypeSimple+plottype"
bplot = bp.loc[bp['class']==bsort].copy()
bplot['group'] = bp['studytype'] + " \n" + bp['plottype']

handles = []
labels = []

# Plot connecting lines
for index, row in bplot.iterrows():
    linestyle = '-' if row['power'] > 0.8 else '--'
    if row['p-value'] > 0.05 or row['power'] < 0.8:
        color = 'grey'
        label = 'no significant change'
    elif row['mean'] < 0:
        color = '#E04A48'
        label = 'decrease'
    else:
        color = "#003473"
        label = 'increase'
    line, = plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                     [row['group'], row['group'], row['group']], 
                     linestyle, c=color, zorder=1)
    plt.scatter(row['mean%'], row['group'], color=color, marker='|', s=25)
    plt.rc('ytick', labelsize=5)  
    
    # Collect legend handles and labels
    if label not in labels:
        handles.append(line)
        labels.append(label)
    
plt.xlabel('mean response [%]')
plt.title('(g)', loc='left', weight='bold')
plt.yticks(fontsize=10)

plt.subplots_adjust(hspace=0.50)
plt.subplots_adjust(wspace=0.40)
plt.savefig('Figure 2.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#%% per C-type and Family combined (Figure 3)
bp = bresults.copy()
bp = bp.loc[bp['class'] == 'c3c4+Family+element+tissueAgg']
bplot = bp.loc[bp['power'] > 0.8].copy()

plt.figure(figsize=(6.8, 6.8), dpi=600)

bplot['group_title'] = bplot['c3c4'] + " " + bplot['Family'] + " " + bplot['tissue']
bplot['group'] = bplot['Family'] + " " + bplot['c3c4'] + " " + bplot['tissue']

counter = 1
handles = []
labels = []

for grp in sorted(bplot['group'].unique()):
    if bplot['group'].nunique() < 5:
        plt.subplot(2, 2, counter)
    else:
        plt.subplot(int(round(bplot['group'].nunique() / 3)) + 1, 3, counter)

    bp = bplot.loc[bplot['group'] == grp]
    bp = bp.drop(bp[bp['element'] == 'N'].index)
    bp = bp.drop(bp[bp['element'] == 'protein'].index)
    bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
    bplot2 = bp.copy()

    # Plot connecting lines
    for index, row in bplot2.iterrows():
        if row['power'] < 0.4:
            linestyle = '--'
        elif row['power'] < 0.8:
            linestyle = '-.'
        else:
            linestyle = '-'
        if row['p-value'] > 0.05:
            color = 'grey'
            label = 'no significant effect'
        elif row['mean'] < 0:
            color = '#E04A48'
            label = 'decrease'
        else:
            color = "#003473"
            label = 'increase'

        line, = plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                         [row['element'], row['element'], row['element']], 
                         linestyle, c=color, zorder=1)
        plt.scatter(row['mean%'], row['element'], color=color, marker='|')

        # Collect legend handles and labels
        if label not in labels:
            handles.append(line)
            labels.append(label)
    
        plt.title(bplot2['group_title'].unique()[0], fontsize=10)
        if bplot2['element'].nunique()>14:
            plt.yticks(fontsize=5)
        elif bplot2['element'].nunique()>12:
            plt.yticks(fontsize=7)
        elif bplot2['element'].nunique()>12:
            plt.yticks(fontsize=6.5)
        elif bplot2['element'].nunique()>7:
            plt.yticks(fontsize=8)
        else:
            plt.yticks(fontsize=10)

    if bplot['group'].nunique() < 5:
        if counter > 2:
            plt.xlabel('mean response [%]', fontsize=10)
    else:
        if counter >= bplot['group'].nunique() - 2:
            plt.xlabel('mean response [%]', fontsize=10)
    counter += 1

# Add a single legend outside the subplots
plt.figlegend(handles, labels, loc='lower center', fontsize=10, 
              bbox_to_anchor=(0.5, 0.195), ncol=3)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, 
                    hspace=0.5, wspace=0.3)

plt.savefig('Figure 3.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#%% per specie combined (Figure 4)
dfTemp = dfOriginal.copy()
dfTemp['sci name'] = dfOriginal['Genus'].astype(str).str[0] + \
                     '. ' + dfOriginal['Species']

bsort = 'c3c4+Family+Genus+Species+element'
bp = bresults.copy()
bp = bp.drop(bp[bp['element'] == 'N'].index)
bp = bp.drop(bp[bp['element'] == 'protein'].index)
bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
bp = bp.loc[bp['class'] == bsort].copy()

plt.figure(figsize=(6.8, 6.8), dpi=600)
bplot = bp.loc[bp['power'] > 0.8].copy()
bplot['name'] = bplot['Genus'].astype(str).str[0] + '. ' + bplot['Species']

# Number of unique species
n_species = bplot['name'].nunique()

# Grid size: ceil(n_species/4) rows, 4 columns
n_rows = int(round(n_species / 4))
n_cols = 4

fig = plt.figure(figsize=(6.8, 6.8), dpi=600)
gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)

counter = 0
handles = []
labels = []
trigger = -1
for grp in sorted(bplot['name'].unique()):
    # make soybean (G. max) taller
    if grp.startswith("G. max"):  
        ax = fig.add_subplot(gs[counter // n_cols: counter // n_cols + 2,
                              counter % n_cols]) 
        trigger = counter + 4
    elif trigger == counter:
        counter += 1 
        ax = fig.add_subplot(gs[counter // n_cols, counter % n_cols])
    else:
        ax = fig.add_subplot(gs[counter // n_cols, counter % n_cols])
    
        
    bplot2 = bplot.loc[bplot['name'] == grp].copy()

    # Plot connecting lines
    for index, row in bplot2.iterrows():
        if row['p-value'] > 0.05 or row['power'] < 0.8:
            color = 'grey'
            label = 'no significant change'
        elif row['mean'] < 0:
            color = '#E04A48'
            label = 'decrease'
        else:
            color = "#003473"
            label = 'increase'
        
        if bplot2['element'].nunique() > 10:
            size = 5
        else:
            size = 15
        
        line, = ax.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                         [row['element'], row['element'], row['element']], 
                         '-', linewidth=size/10, c=color, zorder=1)
        ax.scatter(row['mean%'], row['element'], s=size, 
                   color=color, marker='|')
        
        if label not in labels:
            handles.append(line)
            labels.append(label)

    ax.set_title(grp + ' (' + dfTemp[dfTemp['sci name'].isin(bplot2['name'].unique())].index.unique()[0] + ')', 
                 fontsize=8)

    if bplot2['element'].nunique() > 14:
        ax.tick_params(axis='y', labelsize=6.5)
    elif bplot2['element'].nunique() > 12:
        ax.tick_params(axis='y', labelsize=4.5)
    elif bplot2['element'].nunique() > 10:
        ax.tick_params(axis='y', labelsize=5.5)
    elif bplot2['element'].nunique() > 7:
        ax.tick_params(axis='y', labelsize=6.5)
    else:
        ax.tick_params(axis='y', labelsize=8)

    if counter >= n_species - 3:
        ax.set_xlabel('mean response [%]', fontsize=7)
    
    counter += 1

# Add a single legend outside the subplots
fig.legend(handles, labels, loc='lower center', fontsize=8,
           bbox_to_anchor=(0.83, 0.13), ncol=1)

plt.savefig('Figure 4.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)  
#%% plots cultivars per study-plot type per element (Figure S5-S8)
bsort = 'c3c4+Family+Genus+Species+Cultivar+element+plottype+Tissue+studytype'
bp = bresults.copy()
bp = bp.loc[bp['class'] == bsort]
bp = bp.drop(bp[bp['element'] == 'N'].index)
bp = bp.drop(bp[bp['element'] == 'protein'].index)
bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
bp = bp.loc[bp['power']>=0.8]
bp['group'] = bp['Genus'].astype(str).str[0] + '. ' + bp['Species']
bp['name'] = bp['studytype'] + ' ' + bp['plottype'] + ' ' + bp['Cultivar']
bp2 = bp.copy()

for tst in sorted(bp.groupby('group')
                  .filter(lambda x: x['name'].nunique()>1)['group'].unique()):
    bp = bp2.loc[bp2['group'] == tst].copy()
    bp = bp.drop(bp[bp['element'] == 'N'].index)
    bp = bp.drop(bp[bp['element'] == 'protein'].index)
    bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
    
    
    bp = bp.groupby('element').filter(lambda x: x['name'].nunique()>1)
        
    # Define a global order for elements and make categorical
    global_elements = sorted(bp['name'].unique(), reverse=True)
    element_cats = pd.Categorical(global_elements, categories=global_elements, 
                                  ordered=True)
 
    plt.figure(figsize=(8, 10), dpi=600)

    bplot = bp.loc[bp['m'] > 7].sort_values(by='element')
    
    counter = 1
    handles = []
    labels = []

    for grp in sorted(bplot['element'].unique()):
        plt.subplot(-(len(bplot['element'].unique())//-4), 4, counter)
            
        bp = bplot.loc[bplot['element'] == grp].copy()
        
        existing_studytypes = set(bp['name'])
        missing_studytypes = set(global_elements) - existing_studytypes

        if len(existing_studytypes)<len(global_elements):
            nan_rows = pd.DataFrame({
                        'name': list(missing_studytypes),
                        'element': grp,  # Assuming the same element
                        'power': 0,
                        'p-value': 0,
                        'mean%': 0,
                        '97.5%': 0,
                        '2.5%': 0
                        })

            bp = pd.concat([bp, nan_rows], ignore_index=True)
        
        bplot2 = bp.copy().sort_values('name', ascending=False)

        # Plot connecting lines
        for index, row in bplot2.iterrows():            
            if row['power'] < 0.8:
                alph = 0
            else:
                linestyle = '-'
                alph = 1
                mark = '|'
            if row['p-value'] > 0.05:
                color = 'grey'
                label = 'no significant effect'
            elif row['mean%'] < 0:
                color = '#E04A48'
                label = 'decrease'
            else:
                color = "#003473"
                label = 'increase'

            line, = plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                             [row['name'], row['name'], row['name']], 
                             linestyle, alpha=alph, c=color, zorder=1)
            plt.scatter(row['mean%'], row['name'], 
                        color=color, alpha=alph, marker=mark)
            
            if (counter+3) % 4 == 0:
                plt.ylim(-0.5, len(global_elements) - 0.5)  
                plt.yticks(ticks=range(len(global_elements)), 
                           labels=global_elements)  
            else:
                plt.ylim(-0.5, len(global_elements) - 0.5)
                plt.yticks(ticks=range(len(global_elements)), labels=[])

            # Collect legend handles and labels
            if label not in labels:
                handles.append(line)
                labels.append(label)
        
            plt.title(grp)
            plt.rc('font', size=9)
            plt.rc('ytick', labelsize=8)
            
        if counter>(len(bplot['element'].unique())-4):
            plt.xlabel('mean\nresponse \n[%]')
        counter += 1

    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, 
                        hspace=0.5, wspace=0.3)

    # Add a single legend outside the subplots
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    figname = 'Figure S5-8 ' + tst + '.pdf'
    plt.savefig(figname, format='pdf', bbox_inches='tight', pad_inches = 0)

#%% plots study-plot type per element (S9 + bonus)
bsort = 'c3c4+Family+Genus+Species+element+plottype+Tissue+studytype'
bp = bresults.copy()
bp = bp.loc[bp['class'] == bsort]
bp = bp.drop(bp[bp['element'] == 'N'].index)
bp = bp.drop(bp[bp['element'] == 'protein'].index)
bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
bp = bp.loc[bp['power']>=0.8]
bp['group'] = bp['Genus'].astype(str).str[0] + '. ' + bp['Species']
bp['name'] = bp['studytype'] + ' ' + bp['plottype']
bp2 = bp.copy()

for tst in sorted(bp.groupby('group')
                  .filter(lambda x: x['name'].nunique()>1)['group'].unique()):
    bp = bp2.loc[bp2['group'] == tst].copy()
    bp = bp.drop(bp[bp['element'] == 'N'].index)
    bp = bp.drop(bp[bp['element'] == 'protein'].index)
    bp['element'] = bp['element'].replace('N proxy', 'N$^*$')
    bp = bp.groupby('element').filter(lambda x: x['name'].nunique()>1)
        
    # Define a global order for elements and make categorical
    global_elements = sorted(bp['name'].unique(), reverse=True)
    element_cats = pd.Categorical(global_elements, 
                                  categories=global_elements, 
                                  ordered=True)
 
    plt.figure(figsize=(8, 5.5), dpi=600)

    bplot = bp.loc[bp['m'] > 7].copy()
    bplot = bplot.sort_values(by='element')
    
    counter = 1
    handles = []
    labels = []

    for grp in sorted(bplot['element'].unique()):
        plt.subplot(-(len(bplot['element'].unique())//-4), 4, counter)
            
        bp = bplot.loc[bplot['element'] == grp].copy()
        
        existing_studytypes = set(bp['name'])
        missing_studytypes = set(global_elements) - existing_studytypes

        if len(existing_studytypes)<len(global_elements):
            nan_rows = pd.DataFrame({
                        'name': list(missing_studytypes),
                        'element': grp,  # Assuming the same element
                        'power': 0,
                        'p-value': 0,
                        'mean%': 0,
                        '97.5%': 0,
                        '2.5%': 0
                        })
            bp = pd.concat([bp, nan_rows], ignore_index=True)
        
        bplot2 = bp.copy().sort_values('name', ascending=False)

        # Plot connecting lines
        for index, row in bplot2.iterrows():            
            if row['power'] > 0.8:
                linestyle = '-'
                alph = 1
                mark = '|'
            else:
                alph = 0
            if row['p-value'] > 0.05:
                color = 'grey'
                label = 'no significant effect'
            elif row['mean%'] < 0:
                color = '#E04A48'
                label = 'decrease'
            else:
                color = "#003473"
                label = 'increase'

            line, = plt.plot([row['2.5%'], row['mean%'], row['97.5%']], 
                             [row['name'], row['name'], row['name']], 
                             linestyle, alpha=alph, c=color, zorder=1)
            plt.scatter(row['mean%'], row['name'], color=color, 
                        alpha=alph, marker=mark)
            
            # Consistent y-axis across plots with uniform spacing
            if (counter+3) % 4 == 0:
                plt.ylim(-0.5, len(global_elements) - 0.5)  
                plt.yticks(ticks=range(len(global_elements)), 
                           labels=global_elements)
            else:
                plt.ylim(-0.5, len(global_elements) - 0.5)
                plt.yticks(ticks=range(len(global_elements)), labels=[])

            # Collect legend handles and labels
            if label not in labels:
                handles.append(line)
                labels.append(label)
        
            plt.title(grp)
            plt.rc('font', size=9)
            plt.rc('ytick', labelsize=8)
            
        if counter>(len(bplot['element'].unique())-4):
            plt.xlabel('mean response [%]')
        counter += 1
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, 
                        hspace=0.5, wspace=0.3)
    
    # Add a single legend outside the subplots
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    figname = 'studyplot ' + tst + '.pdf'
    plt.savefig(figname, format='pdf', bbox_inches='tight', pad_inches=0)