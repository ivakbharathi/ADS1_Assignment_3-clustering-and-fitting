

# library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import norm
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
from scipy.stats import t

def read_file(filename):
    """
    # this function read csv data into panda dataframe
    # then clean data and return the data.
    """

    #Load data in Panda Dataframe and skip top 4 rows
    data = pd.read_csv(filename, skiprows = 4)

    #return data
    return data

def preprocess_agriculture_data(data):
    """
    Preprocess agriculture data for analysis.

    Parameters:
    - data: pandas DataFrame containing agriculture data.

    Returns:
    - preprocessed_data: Preprocessed DataFrame ready for analysis.
    """

    #Fill missing values
    data.fillna(0, inplace=True)

    # Drop unnecessary columns
    data = data.drop(["Country Code", "Indicator Code","Unnamed: 67"], axis = 1)

    preprocessed_data = data.copy()

    return preprocessed_data

#Filter Data for Relevant Indicators
indicators = ['Cereal yield (kg per hectare)',
              'Arable land (% of land area)',
              'Agricultural land (% of land area)',
              'Crop production index (2014-2016 = 100)',
              'Fertilizer consumption (kilograms per hectare of arable land)',
              'Rural population (% of total population)',
              'Agricultural machinery, tractors per 100 sq. km of arable land'

]

# Function to filter data for relevant indicators
def filter_data(data):
  """
    Extract data from a DataFrame based on a list of relevant indicators.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the data.

    Returns:
    - pd.DataFrame: DataFrame with data only for the specified indicators.

    """
  filtered_df = data[data["Indicator Name"].isin(indicators)]

  return filtered_df

#This funtion plot trend for specific indicator
def get_trend(data, indicator, ylabel, title):

  # Filtering data for the world indicator
  filter_data = data[(data["Indicator Name"] == indicator) & (data["Country Name"] == "World")]

  # Selecting data for all years from 1990 to 2020
  years = [str(year) for year in range(1990, 2021)]
  filter_data = filter_data[years]

  # Plotting the line for the world
  plt.figure(figsize=(14, 5))
  plt.plot(years, filter_data.iloc[0], linestyle='--', marker='o', color='green')

  # Set x-axis ticks at a 5-year interval
  plt.xticks(np.arange(0, len(years), 5), labels=years[::5])

  plt.xlabel('Year')
  plt.ylabel(ylabel)
  plt.title(title, fontsize=18)

  plt.show()

#Function to take transpose of the dataframe
def get_transpose(data):
  """
    Transpose the dataframe.

    Parameters:
    - data (pd.DataFrame): The dataset.

    Returns:
    - transposed_data (pd.DataFrame): Transposed dataframe.
    """

  # Set 'Country Name' and 'Indicator Name' as multi-level index
  data.set_index(['Country Name','Indicator Name'], inplace=True)


  # Transpose the data to calculate correlation between indicators
  transposed_data = data.T

  return transposed_data

# Define function to calculate confidence ranges
def err_ranges(params, covariance, x_data, model):
        n = len(x_data)
        dof = max(0, n - len(params))  # degrees of freedom

        # Calculate standard deviations of parameters
        param_std_dev = np.sqrt(np.diag(covariance))

        # Calculate t-statistic for 95% confidence interval
        t_value = 2.262

        # Calculate confidence ranges
        lower_bounds = params - t_value * param_std_dev
        upper_bounds = params + t_value * param_std_dev

        return lower_bounds, upper_bounds

# Define the model function
def simple_model(x, a, b, c):
        return a * x**2 + b * x + c

def perform_curve_fitting(df, country_name, indicator_name, years):
    """
    Perform curve fitting for a specific country, indicator, and years.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - country_name (str): Name of the country.
    - indicator_name (str): Name of the indicator.
    - years (list): List of years.

    Returns:
    - None (plots the curve fitting result).
    """

    # Extract relevant data from DataFrame
    country_data = df[(df['Country Name'] == country_name) & (df['Indicator Name'] == indicator_name)]
    x_data = country_data[years].values.flatten()
    y_data = np.arange(len(years))

    # Fit the model to the data using curve_fit
    params, covariance = curve_fit(simple_model, y_data, x_data)

    # Calculate confidence ranges
    lower_bounds, upper_bounds = err_ranges(params, covariance, y_data, simple_model)

    # Generate data for plotting the fitted curve
    x_fit = np.linspace(min(y_data), max(y_data), 1000)
    y_fit = simple_model(x_fit, *params)

    # Plot the data, the fitted curve, and the confidence range
    plt.scatter(y_data, x_data, label='Data')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
    plt.fill_between(x_fit, simple_model(x_fit, *lower_bounds), simple_model(x_fit, *upper_bounds),
                     color='orange', alpha=0.3, label='Confidence Range')

    plt.xlabel('Year')
    plt.ylabel(indicator_name)
    plt.title(f'Curve Fitting for {country_name}')
    plt.legend()
    plt.show()

# Extract data for the specified country
def extract_country(data, country):
  """
    Extract data for specific country.

    Parameters:
    - data (pd.DataFrame): The dataset.
    - country (str): Name of the specific country.

    Returns:
    - country_data (pd.DataFrame): The dataframe for the specified country.
    """
  country_data = data[country].reset_index()

  return country_data

def calculate_correlation(country_data, country, indicator_mapping=None):
    """
    Calculate the correlation of indicators for a specific country.

    Parameters:
    - country_data (pd.DataFrame): The dataset.
    - country (str): Code of the specific country.

    """

    # Rename indicators if indicator_mapping is provided
    if indicator_mapping:
        country_data.rename(columns=indicator_mapping, inplace=True)

    # Calculate the correlation matrix between indicators
    correlation_matrix = country_data.corr()

    # Create a heatmap for correlation between indicators
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap for ' + country, fontsize=18)
    plt.show()

def scale(df):
    """ Expects a dataframe and normalizes all
        columns to the 0-1 range. It also returns
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df_normalized = (df - df_min) / (df_max - df_min)

    return df_normalized, df_min, df_max

def overall_trends_analysis(data):
    """
    Utilize statistical analysis and visualization techniques to examine overall trends in crop production.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing crop production data.

    Returns:
    - None (Displays visualizations).
    """

    # Extract relevant columns for analysis
    relevant_columns = ['Country Name', 'Indicator Name', '2000', '2001', '2002', '2003',
                        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']

    # Filter data for relevant columns
    crop_production_data = data[relevant_columns]

    # Calculate total production for each year
    total_production_per_year = crop_production_data.groupby('Indicator Name').sum()

    # Plotting trends over the years
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=total_production_per_year.transpose(), markers=True)
    plt.title('Overall Trends in Crop Production Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Production')
    plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def top_countries(df, indicator):

  # Selecting data for indicator
  data = df[df["Indicator Name"] == indicator]

  # Sorting in descending order
  data = data.sort_values(by='2020', ascending=False)

  # Top 10 countries
  top_countries = data.head(10)

  # Bar chart data
  countries = top_countries['Country Name']
  indicator_data = top_countries['2020']

  # Plotting the bar chart
  plt.figure(figsize=(13, 8))
  plt.bar(countries, indicator_data, color='navy')
  plt.xlabel('Country')
  plt.ylabel(indicator)
  plt.xticks(rotation=45)
  plt.title(f'Top 10 Countries {indicator}\n2020', fontsize=18)
  plt.show()

def backscale(arr, df_min, df_max):
    """ Expects an array of normalized cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

def get_diff_entries(df1, df2, column):
    """ Compares the values of the column in df1 and the column with the same
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match. """

    # merge dataframes keeping all rows
    df_out = pd.merge(df1, df2, on=column, how="outer")
    print("total entries", len(df_out))
    # merge keeping only rows in common
    df_in = pd.merge(df1, df2, on=column, how="inner")
    print("entries in common", len(df_in))
    df_in["exists"] = "Y"

    # merge again
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")

    # extract columns without "Y" in exists
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()

    return diff_list

def perform_clustering(data, num_clusters, indicator_list):
    """
    Perform K-Means clustering on the given data.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the data for clustering.
    - num_clusters (int): Number of clusters to create.
    - indicator_list (list): List of indicators
    """

    # Selecting relevant data
    filtered_data = data[data["Indicator Name"].isin(indicator_list)]


    # Choose the columns
    selected_data = filtered_data.loc[:, '1980':'2018']

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(selected_data)

    # Perform K-Means clustering
    kmeans = KMeans(num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Calculate silhouette score
    silhouettescores = silhouette_score(data_scaled, clusters)
    print(f"Silhouette Score: {round(silhouettescores, 2)}")

    # Add cluster labels to the DataFrame
    filtered_data['Cluster'] = clusters

    # Visualize the results
    for i in range(max(clusters) + 1):
      cluster_data = data_scaled[clusters == i]
      plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X')

    # Round cluster center values
    cluster_centers_rounded = np.round(kmeans.cluster_centers_, 2)

    plt.title("Clustering of Data")
    plt.xlabel(indicator_list[0])
    plt.ylabel(indicator_list[1])
    plt.legend(title='Clusters')

    plt.show()

    return filtered_data, clusters, cluster_centers_rounded, silhouettescores, scaler.data_min_, scaler.data_max_

def compare_selected_countries(df, clusters, years, crop_indicator):
    """
    Compare selected countries within each cluster based on crop production.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - clusters (pd.Series): Series containing cluster labels for each country.
    - years (list): List of years to analyze.
    - crop_indicator (str): Name of the crop production indicator.

    Returns:
    - plt.Figure: Matplotlib figure containing the comparison plot.
    - selected_countries
    """

    # Choose one country from each cluster
    selected_countries = []
    for i in range(max(clusters) + 1):
        cluster_data = df[df['Cluster'] == i]
        selected_country = cluster_data.iloc[0]['Country Name']
        selected_countries.append(selected_country)

    # Print the selected countries
    print("Selected Countries from Each Cluster:")
    for i, country in enumerate(selected_countries):
        print(f"Cluster {i + 1}: {country}")

    # Analyze and compare the selected countries within each cluster
    for i, country in enumerate(selected_countries):
        cluster_data = df[df['Cluster'] == i]
        country_data = cluster_data[cluster_data['Country Name'] == country]

        # Select only the chosen years
        country_data_subset = country_data[['Country Name'] + years]

        # Perform analysis and visualization for each selected country within the cluster
        # Plotting a line chart for the selected years
        plt.plot(country_data_subset.columns[1:], country_data_subset.iloc[0, 1:], label=f'{country} - Cluster {i + 1}')

    # Customize the plot
    plt.title(f"Comparison of Selected Countries from each Clusters \n{crop_indicator}")
    plt.xlabel("Year")
    plt.ylabel(f"{crop_indicator} (Add unit if available)")

    # Place the legend to the right outside the plot
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return selected_countries

def plot_filtered_data(data, selected_countries, years, indicator, kind):
    """
    Plot a bar graph for specific countries, years, and indicator.
    Use for plotting countries selected from clusters for specific years and indicator
    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the data.
    - selected_countries (list): List of specific countries to include in the plot.
    - years (list): List of years for the x-axis.
    - indicator (str): Indicator to plot.
    - kind (str): Type of plot

    Returns:
    - None (Displays a bar graph).
    """

    # Filter data for selected countries, years, and indicator
    filtered_df = data[(data["Country Name"].isin(selected_countries)) & (data['Indicator Name'] == indicator)][['Country Name'] + years]

    # Transpose the DataFrame for correct plotting orientation
    filtered_df_transposed = filtered_df.set_index('Country Name').T

    # Plotting a bar graph
    filtered_df_transposed.plot(kind= kind, figsize=(10, 6))
    plt.title(indicator)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_topcountries(data, indicator):

  # Specify different colors for each bar
  bar_colors = ['blue', 'green', 'yellow', 'lightpink', 'lightseagreen']

  # Selecting data for Electric power consumption
  filtered_data = data[data["Indicator Name"] == indicator]

  # Sorting by electric power consumption in descending order
  filtered_data = filtered_data.sort_values(by='2020', ascending=False)

  # Top 5 countries
  filtered_data = filtered_data.head(5)

  # Data for the horizontal bar chart
  countries = filtered_data['Country Name']
  indicators_data = filtered_data['2020']


  # Plotting the horizontal bar chart
  plt.figure(figsize=(13, 8))
  plt.barh(countries, indicators_data, color=bar_colors)
  plt.xlabel(indicator)
  plt.ylabel('Country')
  plt.title(f'Top 10 Countries {indicator}\n2020')
  plt.show()

def compare_countries_within_cluster(data, indicator, cluster_label, years):
    """
    Compare countries within the same cluster.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the data.
    - cluster_label (int): Cluster label for which countries will be compared.
    - years (list): List of years for visualization.

    Returns:
    - None (Displays a comparison plot).
    """

    # Select countries belonging to the specified cluster
    cluster_data = data[data['Cluster'] == cluster_label]

    # Choose a subset of countries from the cluster for comparison
    selected_countries = cluster_data.sample(min(4, len(cluster_data)))['Country Name'].tolist()

    plt.figure(figsize=(14, 5))

    for country in selected_countries:
        country_data = data[(data['Country Name'] == country) & (data['Cluster'] == cluster_label)]

        # Select only the chosen years
        country_data_subset = country_data[['Country Name'] + years]

        # Perform analysis and visualization for each selected country within the cluster
        plt.plot(country_data_subset.columns[1:], country_data_subset.iloc[0, 1:], label=f'{country} - Cluster {cluster_label+1}', linestyle='--', marker='o')

    # Customize the plot
    plt.title(f"Comparison of Countries within Cluster {cluster_label+1}", fontsize=18)
    plt.xlabel("Year")
    plt.ylabel(indicator)

    # Place the legend to the right outside the plot
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return selected_countries



#Main Function
if __name__ == "__main__":
  data = read_file("API_1_DS2_en_csv_v2_6303247.csv")
  data = preprocess_agriculture_data(data)

  #filter relevant data
  data = filter_data(data)

#Describing Data
data.head()
data.describe()
data.info
data.columns

#Get Trend of different indicators on World Levele
#get_trend(data, indicator, ylabel, title)

get_trend(data, "Arable land (% of land area)", "Arable land (% of land area)", "World Arable Land (% of land area)")
get_trend(data, "Agricultural land (% of land area)", "Agricultural land (% of land area)", "World Agricultural Land (% of land area)")
get_trend(data, "Cereal yield (kg per hectare)", "Cereal yield (kg per hectare)", "World Cereal Yield (kg per hectare)")
get_trend(data, "Fertilizer consumption (kilograms per hectare of arable land)", "Fertilizer Consumption\n (kg per hectare of arable land)", "World Fertilizer Consumption")
get_trend(data, "Rural population (% of total population)", "Rural population\n (% of total population)", "World Rural Population")
get_trend(data, "Agricultural machinery, tractors per 100 sq. km of arable land", "Agricultural machinery, tractors \n(per 100 sq. km of arable land)", "World Agricultural Machinery, Tractors")

top_countries(data, 'Cereal yield (kg per hectare)')
top_countries(data, 'Crop production index (2014-2016 = 100)')

df = data.copy()
#get transpose of data
transposed_data = get_transpose(df)
#transposed_data.head()

#Extract specific country data
country_name = "Oman"
country_data = extract_country(transposed_data,country_name)
#country_data.head()

# Indicator_mapping is a dictionary mapping original indicator names to desired names
indicator_mapping = {
              'Cereal yield (kg per hectare)': 'Cereal yield',
              'Agricultural land (% of land area)': 'Agricultural land',
              'Crop production index (2014-2016 = 100)': 'Crop production index ',
              'Fertilizer consumption (kilograms per hectare of arable land)': 'Fertilizer consumption',
              'Rural population (% of total population)' : 'Rural population',
              'Agricultural machinery, tractors per 100 sq. km of arable land' : 'Agricultural machinery'
}
calculate_correlation(country_data, country_name, indicator_mapping)

# Plotting Trend of various indicators over time
overall_trends_analysis(data)

#horizonatal bar graph to plot top countries of specific indicator
plot_topcountries(data, 'Cereal yield (kg per hectare)')
plot_topcountries(data, 'Fertilizer consumption (kilograms per hectare of arable land)')

#clustering of countries on indicators
indicator_list = ['Cereal yield (kg per hectare)', 'Arable land (% of land area)']
df1, clusters1, cluster_center1, silhoutte1, df1_min, df1_max = perform_clustering(data, 3, indicator_list)

indicator_list = ['Cereal yield (kg per hectare)', 'Rural population (% of total population)']
df2, clusters2, cluster_center2, silhoutte2, df2_min, df2_max = perform_clustering(data, 5, indicator_list)

indicator_list = ['Crop production index (2014-2016 = 100)', 'Arable land (% of land area)']
df3, clusters3, cluster_center3, silhoutte3, df3_min, df3_max = perform_clustering(data, 3, indicator_list)

perform_curve_fitting(data, 'Kuwait', 'Cereal yield (kg per hectare)',
                      ['1988', '1992', '1996', '2000', '2004', '2008', '2012', '2016', '2020', '2022'])

#Compare countries from each cluster
#To find similarity and difference among them
selected_countries = compare_selected_countries(df1, clusters1, ['1990', '1995', '2000', '2004', '2008', '2012', '2016', '2020'], 'Cereal yield (kg per hectare)')

print (selected_countries)

# compare countries within Cluster 3 for the specific indicator for which cluster are constructed

cluster_label_to_compare = 2
indicator_to_compare1 = 'Cereal yield (kg per hectare)'
indicator_to_compare2 = 'Crop production index (2014-2016 = 100)'
years_to_compare = ['1980','1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020']

cluster_countries1 = compare_countries_within_cluster(df1, indicator_to_compare1, cluster_label_to_compare, years_to_compare)

cluster_countries2 = compare_countries_within_cluster(df3, indicator_to_compare2, cluster_label_to_compare, years_to_compare)

cluster_countries1

years_to_plot = ['2000', '2005', '2010', '2015', '2020']
kind = 'bar'
#comparing countries within same clusters for all indicators
for indicator in indicators:

  plot_filtered_data(data, cluster_countries1, years_to_plot, indicator, kind)

#comparing countries from each clusters for all indicators
kind = 'line'
for indicator in indicators:
  plot_filtered_data(data, selected_countries, years_to_compare, indicator, kind)