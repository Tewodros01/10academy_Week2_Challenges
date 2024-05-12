# src/utils.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data():
    # Load data from SQL database
    engine = create_engine("postgresql://postgres:1992202424@localhost/telecom_db")
    data = pd.read_sql("SELECT * FROM xdr_data", engine)
    return data

def task_1():
    st.subheader("Task 1 - User Overview Analysis")
    st.write("Performing user overview analysis...")

    # Load data
    data = load_data()

    # Correct any data formatting issues
    data.replace("\\N", pd.NA, inplace=True)

    # Name the columns
    column_names = [
      "Bearer Id", "Start", "Start ms", "End", "End ms", "Dur. (ms)", "IMSI",
      "MSISDN/Number", "IMEI", "Last Location Name", "Avg RTT DL (ms)",
      "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)",
      "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)",
      "DL TP < 50 Kbps (%)", "50 Kbps < DL TP < 250 Kbps (%)", "250 Kbps < DL TP < 1 Mbps (%)",
      "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)", "10 Kbps < UL TP < 50 Kbps (%)",
      "50 Kbps < UL TP < 300 Kbps (%)", "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)",
      "HTTP UL (Bytes)", "Activity Duration DL (ms)", "Activity Duration UL (ms)",
      "Dur. (ms).1", "Handset Manufacturer", "Handset Type", "Nb of sec with 125000B < Vol DL",
      "Nb of sec with 1250B < Vol UL < 6250B", "Nb of sec with 31250B < Vol DL < 125000B",
      "Nb of sec with 37500B < Vol UL", "Nb of sec with 6250B < Vol DL < 31250B",
      "Nb of sec with 6250B < Vol UL < 37500B", "Nb of sec with Vol DL < 6250B",
      "Nb of sec with Vol UL < 1250B", "Social Media DL (Bytes)", "Social Media UL (Bytes)",
      "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)", "Email UL (Bytes)",
      "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)",
      "Netflix UL (Bytes)", "Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)",
      "Other UL (Bytes)", "Total UL (Bytes)", "Total DL (Bytes)"
    ]

    data.columns = column_names

    # Check data types and missing values
    st.write("Data Information:")
    st.write(data.info())
    missing_values = data.isnull().sum()
    st.write("Missing Values:")
    st.write(missing_values)

    # Top 10 handsets
    top_10_handsets = data['Handset Type'].value_counts().head(10)
    st.write("Top 10 Handsets:")
    st.write(top_10_handsets)

    # Plotting the top 10 handsets
    st.write("Plot of Top 10 Handsets:")
    st.bar_chart(top_10_handsets)

    # Top 3 handset manufacturers
    top_3_manufacturers = data['Handset Manufacturer'].value_counts().head(3)
    st.write("Top 3 Manufacturers:")
    st.write(top_3_manufacturers)

    # pie chart
    st.write("Market Share of Top 3 Handset Manufacturers:")
    fig, ax = plt.subplots()
    ax.pie(top_3_manufacturers, labels=top_3_manufacturers.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    # Display image in Streamlit
    st.image(img, caption="Market Share of Top 3 Handset Manufacturers", use_column_width=True)

    # Group by manufacturer and get top 5 handsets per top 3 manufacturers
    top_5_per_manufacturer = {}
    for manufacturer in top_3_manufacturers.index:
        handsets_per_manufacturer = data[data['Handset Manufacturer'] == manufacturer]
        top_5_per_manufacturer[manufacturer] = handsets_per_manufacturer['Handset Type'].value_counts().head(5)

    st.write("Top 5 Handsets Per Top 3 Manufacturers:")
    for manufacturer, handsets in top_5_per_manufacturer.items():
        st.write(f"{manufacturer}:")
        st.write(handsets)

    # Bar plots for each of the top 3 manufacturers' top 5 handsets
    manufacturers = top_5_per_manufacturer.keys()

    for manufacturer in manufacturers:
        st.write(f"Plot for Top 5 Handsets for {manufacturer}:")
        st.bar_chart(top_5_per_manufacturer[manufacturer])

    # Additional data processing and analysis...
def task_2():
    st.subheader("Task 2 - User Engagement Analysis")
    st.write("Performing user engagement analysis...")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Load data
    data = load_data()

    # Aggregate engagement metrics per customer (MSISDN/Number)
    engagement_metrics = data.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Session frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum'  # Total upload data
    })

    # Rename columns for clarity
    engagement_metrics.rename(columns={
        'Bearer Id': 'Session Frequency',
        'Dur. (ms)': 'Total Session Duration',
        'Total DL (Bytes)': 'Total Download Data',
        'Total UL (Bytes)': 'Total Upload Data'
    }, inplace=True)

    # Add a total traffic column
    engagement_metrics['Total Traffic'] = (
        engagement_metrics['Total Download Data'] + engagement_metrics['Total Upload Data']
    )

    # Check top 10 customers per engagement metric
    st.write("Top 10 Customers by Session Frequency:")
    st.write(engagement_metrics['Session Frequency'].sort_values(ascending=False).head(10))

    st.write("Top 10 Customers by Total Session Duration:")
    st.write(engagement_metrics['Total Session Duration'].sort_values(ascending=False).head(10))

    st.write("Top 10 Customers by Total Traffic:")
    st.write(engagement_metrics['Total Traffic'].sort_values(ascending=False).head(10))

    # Normalize engagement metrics
    scaler = StandardScaler()
    normalized_engagement_metrics = scaler.fit_transform(engagement_metrics)

    # Perform k-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    engagement_clusters = kmeans.fit_predict(normalized_engagement_metrics)

    # Add cluster labels to the data
    engagement_metrics['Engagement Cluster'] = engagement_clusters

    # Compute statistics for each cluster
    cluster_stats = engagement_metrics.groupby('Engagement Cluster').agg({
        'Session Frequency': ['min', 'max', 'mean', 'sum'],
        'Total Session Duration': ['min', 'max', 'mean', 'sum'],
        'Total Traffic': ['min', 'max', 'mean', 'sum']
    })

    st.write("Cluster Statistics:")
    st.write(cluster_stats)

    # Plot the clusters
    st.write("Engagement Clusters:")
    sns.scatterplot(
        x=engagement_metrics['Session Frequency'],
        y=engagement_metrics['Total Traffic'],
        hue=engagement_metrics['Engagement Cluster'],
        palette='viridis'
    )
    plt.xlabel('Session Frequency')
    plt.ylabel('Total Traffic')
    plt.title('Engagement Clusters')
    st.pyplot()

    # Display statistics table
    st.write("Cluster Statistics:")
    st.write(cluster_stats)

    # Define the application data mapping
    app_data = {
        'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
        'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
        'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
        'YouTube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
        'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
        'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
        'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
    }

    application_traffic = {}
    for app, fields in app_data.items():
        application_traffic[app] = (
            data.groupby('MSISDN/Number')[fields[0]].sum() +
            data.groupby('MSISDN/Number')[fields[1]].sum()
        )

    # Find the top 10 most engaged users per application
    top_users_per_app = {}
    for app, traffic in application_traffic.items():
        top_users_per_app[app] = traffic.sort_values(ascending=False).head(10)

    st.write("Top 10 Most Engaged Users per Application:")
    for app, top_users in top_users_per_app.items():
        st.write(f"{app}:")
        st.write(top_users)

    # Plot the top 3 most used applications
    top_3_apps = sorted(application_traffic, key=lambda k: application_traffic[k].sum(), reverse=True)[:3]

    for app in top_3_apps:
        sns.histplot(application_traffic[app], kde=True)
        plt.title(f"Distribution of Total Traffic in {app}")
        plt.xlabel("Total Traffic (Bytes)")
        plt.show()

    # Using the elbow method to find the optimal number of clusters
    sum_of_squared_distances = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_engagement_metrics)
        sum_of_squared_distances.append(kmeans.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    st.pyplot()

def task_3():
    st.subheader("Task 3 - Experience Analytics")
    st.write("Performing experience analytics...")

    # Load data
    data = load_data()

    # Task 3 code here...

    # Task 3.1: Aggregate information per customer
    # Convert relevant columns to numeric data types
    numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Aggregate information per customer
    customer_info = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',  # Average TCP retransmission
        'Avg RTT DL (ms)': 'mean',  # Average RTT
        'Handset Type': 'first',  # Handset type (assuming it's constant for each customer)
        'Avg Bearer TP DL (kbps)': 'mean'  # Average throughput
    }).reset_index()

    # Task 3.2: Compute and list top, bottom, and most frequent values
    # Top values
    top_tcp_values = data['TCP DL Retrans. Vol (Bytes)'].value_counts().head(10)
    top_rtt_values = data['Avg RTT DL (ms)'].value_counts().head(10)
    top_throughput_values = data['Avg Bearer TP DL (kbps)'].value_counts().head(10)

    # Bottom values
    bottom_tcp_values = data['TCP DL Retrans. Vol (Bytes)'].value_counts().tail(10)
    bottom_rtt_values = data['Avg RTT DL (ms)'].value_counts().tail(10)
    bottom_throughput_values = data['Avg Bearer TP DL (kbps)'].value_counts().tail(10)

    # Most frequent values
    frequent_tcp_values = data['TCP DL Retrans. Vol (Bytes)'].mode()
    frequent_rtt_values = data['Avg RTT DL (ms)'].mode()
    frequent_throughput_values = data['Avg Bearer TP DL (kbps)'].mode()

    # Task 3.3: Compute and report distributions and interpretations
    # Distribution of average throughput per handset type
    throughput_per_handset = customer_info.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=customer_info, x='Avg Bearer TP DL (kbps)', hue='Handset Type', kde=True)
    plt.title('Distribution of Average Throughput per Handset Type')
    plt.xlabel('Average Throughput (kbps)')
    plt.ylabel('Density')
    plt.legend(title='Handset Type')
    plt.show()

    # Average TCP retransmission per handset type
    tcp_retransmission_per_handset = customer_info.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()

    # Convert all object dtype columns to numeric (if needed)
    customer_info = customer_info.apply(pd.to_numeric, errors='coerce')

    # Task 3.4: Perform k-means clustering
    # Prepare data for clustering
    X = customer_info[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Handset Type']]

    # Fill any missing values with 0
    X = X.fillna(0)

    # Perform k-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_info['Cluster'] = kmeans.fit_predict(X)

    # Provide a brief description of each cluster
    cluster_descriptions = customer_info.groupby('Cluster').mean()
    st.write("Cluster Descriptions:")
    st.write(cluster_descriptions)

def task_4():
    st.subheader("Task 4 - Satisfaction Analysis")
    st.write("Performing satisfaction analysis...")

    # Load data
    data = load_data()

    # Task 4 code here...

    # Convert relevant columns to numeric data types
    numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Aggregate information per customer
    customer_info = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',  # Average TCP retransmission
        'Avg RTT DL (ms)': 'mean',  # Average RTT
        'Handset Type': 'first',  # Handset type
        'Avg Bearer TP DL (kbps)': 'mean'  # Average throughput
    }).reset_index()

    # Prepare data for clustering
    X = customer_info[['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']].fillna(0)

    # Perform k-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_info['Cluster'] = kmeans.fit_predict(X)

    # Calculate the Euclidean distance between each user data point and the centroids of the clusters
    customer_info['Engagement_Score'] = np.linalg.norm(X - kmeans.cluster_centers_[0], axis=1)
    customer_info['Experience_Score'] = np.linalg.norm(X - kmeans.cluster_centers_[-1], axis=1)

    # Calculate the Satisfaction Score as the average of engagement and experience scores
    customer_info['Satisfaction_Score'] = (customer_info['Engagement_Score'] + customer_info['Experience_Score']) / 2

    # Report Top 10 Satisfied Customers
    top_10_satisfied_customers = customer_info.nsmallest(10, 'Satisfaction_Score')

    # Prepare features and target variable for regression model
    X_regression = customer_info[['Engagement_Score', 'Experience_Score']].fillna(0)
    y_regression = customer_info['Satisfaction_Score']

    # Split the data into training and testing sets for regression model
    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

    # Initialize and train the regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    # Predict the satisfaction score using regression model
    y_pred = regression_model.predict(X_test)

    # Run k-means (k=2) on the engagement & the experience score
    kmeans_scores = KMeans(n_clusters=2, random_state=42)
    customer_info['Cluster_Scores'] = kmeans_scores.fit_predict(X_regression)

    # Aggregate the average satisfaction & experience score per cluster
    cluster_aggregate_scores = customer_info.groupby('Cluster_Scores')[['Satisfaction_Score', 'Experience_Score']].mean()

    engine = create_engine("postgresql://postgres:1992202424@localhost/telecom_db")

    # Export the final table containing all user ID, engagement, experience & satisfaction scores to database
    customer_info.to_sql('satisfaction_scores', con=engine, if_exists='replace', index=False)

    # Display basic information about the satisfaction_scores
    satisfaction_scores_data = pd.read_sql("SELECT * FROM satisfaction_scores", engine)
    st.write("Basic Information about Satisfaction Scores:")
    st.write(satisfaction_scores_data.info())
    st.write("Head of Satisfaction Scores Data:")
    st.write(satisfaction_scores_data.head())
