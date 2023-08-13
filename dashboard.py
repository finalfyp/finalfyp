from ntpath import join
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import folium
import plotly.express as px
import numpy as np
from streamlit_folium import st_folium
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff


st.set_page_config(page_title=" Job Postings Dashboard | Real Time IT Skills Recommender and Predictor", layout='wide', initial_sidebar_state='expanded')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Job Postings Dashboard')
st.title('Job Postings Dashboard')

jobs_dataset = pd.read_csv('all_data1.csv')

#Date filteration Row
st.markdown('### Select Date to Filter the Data')
col1,col2=st.columns((2))
jobs_dataset["Date"]=pd.to_datetime(jobs_dataset["Date"])

startDate=pd.to_datetime(jobs_dataset["Date"]).min()
endDate=pd.to_datetime(jobs_dataset["Date"]).max()

with col1:
    date1=pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2=pd.to_datetime(st.date_input("End Date", endDate))

jobs_dataset=jobs_dataset[(jobs_dataset["Date"] >= date1) & (jobs_dataset["Date"] <= date2)].copy()

#----------------------------------------------------------------------------------------------------------

# Quick Stats
st.markdown('### Trending IT Job Postings: Quick Stats')
s_col1,s_col2=st.columns(2)
with s_col1:
    col1, col2 = st.columns(2)
    col1.metric("Total Job Postings Records", f"{len(jobs_dataset)}", "now")
with s_col2:
    no_of_jobs_col_1, no_of_jobs_col_2, no_of_jobs_col_3, no_of_jobs_col_4 = st.columns(4)
    no_of_jobs_col_1.metric("Senior", str(len(
        jobs_dataset[jobs_dataset.Position == 'Senior'])))
    no_of_jobs_col_2.metric(
        "Manager", str(len(jobs_dataset[jobs_dataset.Position == 'Manager'])))
    no_of_jobs_col_3.metric("Junior", str(len(
        jobs_dataset[jobs_dataset.Position == 'Junior'])))
    no_of_jobs_col_4.metric("Not Mentioned", str(len(
        jobs_dataset[jobs_dataset.Position == 'Others'])))
st.write("-----")

s1_col1,s1_col2=st.columns(2)
with s1_col1:
    gender_col_1, gender_col_2, gender_col_3,gender_col_4 = st.columns(4)
    gender_col_1.metric("Male", str(len(
        jobs_dataset[jobs_dataset.Gender == 'Male'])))
    gender_col_2.metric(
        "Female", str(len(jobs_dataset[jobs_dataset.Gender == 'Female'])))
    gender_col_3.metric("Both", str(len(
        jobs_dataset[jobs_dataset.Gender == 'Both'])))
with s1_col2:
    job1_col_4, job1_col_1, job1_col_2, job1_col_3 = st.columns(4)
    job1_col_1.metric("Part-time", str(len(
        jobs_dataset[jobs_dataset.Job_Type_I == 'Part-time'])))
    job1_col_2.metric(
        "Full-time", str(len(jobs_dataset[jobs_dataset.Job_Type_I == 'Full-time'])))
    job1_col_3.metric("Contract", str(len(
        jobs_dataset[jobs_dataset.Job_Type_I == 'Contract'])))
st.write("-----")


s2_col1,s2_col2=st.columns(2)
with s2_col1:
    job2_col_1, job2_col_2, job2_col_3, job2_col_4= st.columns(4)
    job2_col_1.metric("Remote", str(len(
        jobs_dataset[jobs_dataset.Job_Type_II == 'Remote'])))
    job2_col_2.metric(
        "Hybrid", str(len(jobs_dataset[jobs_dataset.Job_Type_II == 'Hybrid'])))
    job2_col_3.metric("On-Site", str(len(
        jobs_dataset[jobs_dataset.Job_Type_II == 'On-Site'])))

with s2_col2:
    shift_col_4, shift_col_1, shift_col_2, shift_col_3 = st.columns(4)
    shift_col_1.metric("Day Shift", str(len(
        jobs_dataset[jobs_dataset.Shift == 'Day Shift'])))
    shift_col_2.metric(
        "Night Shift", str(len(jobs_dataset[jobs_dataset.Shift == 'Night Shift'])))
    shift_col_3.metric("Both", str(len(
        jobs_dataset[jobs_dataset.Shift == 'Not Mentioned'])))
st.write("-----")
#----------------------------------------------------------------------------------------------------

#Sidebar for JobTitle
st.sidebar.header("Choose IT Field Filter:")
jobtitle=st.sidebar.multiselect("Pick IT Field (Main Filter)",jobs_dataset["JobTitle"].unique())
if not jobtitle:
    job_df=jobs_dataset.copy()
else:
    job_df=jobs_dataset[jobs_dataset["JobTitle"].isin(jobtitle)]


#--------------------------------------------------------------------------------------------

#Sidebar for Location
st.sidebar.header("Choose Location Filter:")
location=st.sidebar.multiselect("Pick Location",job_df["Location"].unique())
if not location:
    loc_df=job_df.copy()
else:
    loc_df=job_df[job_df["Location"].isin(location)]


if not jobtitle and not location:
    filtered_loc_df=jobs_dataset
elif not jobtitle:
    filtered_loc_df=jobs_dataset[jobs_dataset["Location"].isin(location)]
elif not location:
    filtered_loc_df=jobs_dataset[jobs_dataset["JobTitle"].isin(jobtitle)]
else:
    filtered_loc_df=jobs_dataset[jobs_dataset["JobTitle"].isin(jobtitle) & jobs_dataset["Location"].isin(location)] 

#--------------------------------------------------------------------------------------------


top10_col, job_loc_col = st.columns(2)


with top10_col:
    # Get the top 10 job titles
    st.markdown('### Top 10 Job Postings w.r.t. date')

    # Filter the dataset based on the selected date range
    filtered_dataset = jobs_dataset[(jobs_dataset["Date"] >= date1) & (jobs_dataset["Date"] <= date2)]

    # Get the top 10 job titles from the filtered dataset
    top_10_job_titles = filtered_dataset["JobTitle"].value_counts().head(10).index.tolist()

    # Display the top 10 job titles
    for job_title in top_10_job_titles:
        st.write(job_title)


with job_loc_col:
    # Get the top 10 job titles
    st.markdown('### Job Title Count by Location (Select Location)')

    # Filter the dataset based on selected job titles and locations
    if not location:
        filtered_location_df = jobs_dataset.copy()
    else:    
        filtered_location_df = jobs_dataset[jobs_dataset["Location"].isin(location)]

    # Calculate the count of job titles
    job_count_data = filtered_location_df.groupby('JobTitle').size().reset_index(name='Count')

    # Create the Sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=job_count_data['JobTitle'],
        parents=['' for _ in job_count_data['JobTitle']],
        values=job_count_data['Count'],
        branchvalues='total',
    ))

    # Configure the Sunburst chart layout
    fig.update_layout(
        height=500,
    )

    # Display the Sunburst chart on Streamlit dashboard
    st.plotly_chart(fig, use_container_width=True)

st.write("-----")
#-------------------------------------------------------------------------------------------------------------------

#IT trends
if st.checkbox("IT Field Job Posting Trends over Time"):
    st.markdown('### IT Field Job Posting Trends over Time (Select IT Field)')

    # Filter the data based on selected job titles
    if not jobtitle:
        line_df = jobs_dataset.copy()
    else:
        line_df = jobs_dataset[jobs_dataset["JobTitle"].isin(jobtitle)]

    # Create the line chart
    chart = alt.Chart(line_df).mark_line().encode(
        x='Date:T',
        y='count():Q',
        color='JobTitle:N',
        tooltip=['Date:T', 'JobTitle:N', 'count():Q']
    ).properties(
        width=800,
        height=400
    ).interactive()

    # Enable hover interaction
    chart = chart.interactive()

    # Display the line chart on Streamlit dashboard
    st.altair_chart(chart, use_container_width=True)

st.write("-----")

#-------------------------------------------------------------------------------------------------------------------
#Job Counts
if st.checkbox("Job Postings Counts w.r.t date"):
    st.markdown('### Job Postings Counts w.r.t date(Select IT Trends to get unique counts)')

    chart = alt.Chart(job_df).mark_bar().encode(
        y=alt.Y('count(JobTitle)'),  # 'count(JobTitle):Q',
        x=alt.Y('JobTitle',
                # scale=alt.Scale(range=range_),
                sort='-y')
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

st.write("-----")
#-------------------------------------------------------------------------------------------------------------------
#Sidebar for Position
st.sidebar.header("Choose Position Filter:")
position=st.sidebar.multiselect("Pick Position",job_df["Position"].unique())
if not position:
    position_df=job_df.copy()
else:
    position_df=job_df[job_df["Position"].isin(position)]


if st.checkbox("Position count across IT Job Postings & Location"):
    st.markdown('### Position count across IT Job Postings & Location (Select IT Field & Location)')

    # Calculate the count of subcategories of positions for each job title
    position_counts = filtered_loc_df.groupby(['JobTitle', 'Position'])['Position'].count().reset_index(name='Count')

    # Filter the position_counts based on the selected position
    if position:
        position_counts = position_counts[position_counts['Position'].isin(position)]

    # Create the stacked bar chart
    chart = alt.Chart(position_counts).mark_bar().encode(
        x='JobTitle:N',
        y='Count:Q',
        color='Position:N',
        tooltip=['JobTitle:N', 'Position:N', 'Count:Q']
    ).properties(
        width=800,
        height=400
    ).interactive()

    # Display the stacked bar chart on Streamlit dashboard
    st.altair_chart(chart, use_container_width=True)

st.write("-----")
# ------------------------------------------------------------------------------------------------------------------------------

if st.checkbox("Gender & Shift count across IT Job Postings & Location"):
    gender_col1, shift_col2 = st.columns(2)
    with gender_col1:   #for gender
        st.markdown('### Gender count across IT Job Postings & Location (Select IT Field & Location)')

        # Calculate the count of subcategories of gender
        gender_counts = filtered_loc_df.groupby('Gender').size().reset_index(name='Count')

        # Create the pie chart
        fig = px.pie(gender_counts, values='Count', names='Gender', hole=0.5)
        fig.update_traces(text=gender_counts['Gender'], textposition='outside')

        # Display the pie chart on Streamlit dashboard
        st.plotly_chart(fig, use_container_width=True)

    with shift_col2:  #for shift
        st.markdown('### Shift count across IT Job Postings & Location (Select IT Field & Location)')

        # Calculate the count of subcategories of shift
        shift_counts = filtered_loc_df.groupby('Shift').size().reset_index(name='Count')

        # Create the pie chart
        fig = px.pie(shift_counts, values='Count', names='Shift', hole=0.5)
        fig.update_traces(text=shift_counts['Shift'], textposition='outside')

        # Display the pie chart on Streamlit dashboard
        st.plotly_chart(fig, use_container_width=True)

st.write("-----")

#--------------------------------------------------------------------------------------------------------------------------------

# Filter the dataset based on selected job titles
if not jobtitle:
    filtered_df = jobs_dataset.copy()
else:
    filtered_df = jobs_dataset[jobs_dataset["JobTitle"].isin(jobtitle)]

if st.checkbox("Salary data across IT Job Postings"):
    st.markdown('### Salary data across IT Job Postings (Select IT Field)')
    # Define the salary ranges
    bins = np.arange(0, filtered_df["Salary"].max() + 25000, 25000)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]

    # Group the data into salary ranges and calculate the counts
    salary_counts = pd.cut(filtered_df["Salary"], bins=bins, labels=labels, right=False).value_counts().reset_index()
    salary_counts.columns = ["Salary Range", "Count"]

    # Sort the salary ranges in increasing order
    salary_counts = salary_counts.sort_values("Salary Range")

    # Create the grouped bar chart
    chart = alt.Chart(salary_counts).mark_bar().encode(
        x=alt.X("Salary Range:O", title="Salary Range", sort=alt.SortField(field="Salary Range")),
        y=alt.Y("Count:Q", title="Count"),
        tooltip=["Salary Range:O", "Count:Q"],
    ).properties(
        width=800,
        height=400,
        title="Salary Distribution by Range",
    )

    # Display the grouped bar chart on Streamlit dashboard
    st.altair_chart(chart, use_container_width=True)

st.write("-----")


#--------------------------------------------------------------------------------------------------------------------------------


#Sidebar for Company
st.sidebar.header("Choose Company Filter:")
company=st.sidebar.multiselect("Pick Company",jobs_dataset["Company"].unique())
if not company:
    company_df=jobs_dataset.copy()
else:
    company_df=jobs_dataset[jobs_dataset["Company"].isin(company)]

# Filter the dataset based on selected companies
if company:
    company_df = jobs_dataset[jobs_dataset["Company"].isin(company)]
else:
    company_df = jobs_dataset.copy()


if st.checkbox("Company count across IT Job Postings"):
    st.markdown('### Company count across IT Job Postings (Select IT Field & Company)')
    # Calculate the count of companies for each date
    company_date_counts = company_df.groupby(['Date', 'Company']).size().reset_index(name='Count')

    # Create the area chart
    chart = alt.Chart(company_date_counts).mark_area().encode(
        x='Date:T',
        y='Count:Q',
        color='Company:N',
        tooltip=['Company:N', 'Count:Q']
    ).properties(
        width=600,
        height=400
    )

    # Display the area chart on Streamlit dashboard
    st.altair_chart(chart, use_container_width=True)

st.write("-----")
#--------------------------------------------------------------------------------------------------------------------------------   

# Calculate counts based on selected job titles or overall counts
if jobtitle:
    job_counts = filtered_loc_df.groupby('Location')['JobTitle'].count().reset_index()
else:
    job_counts = filtered_loc_df.groupby('Location')['JobTitle'].count().reset_index()

if st.checkbox("IT Job Postings Across Locations"):
    st.markdown('### IT Job Postings Across Locations (Select IT Field)')
    # Create the bar chart
    plt.figure(figsize=(14, 6))
    barplot = sns.barplot(x='Location', y='JobTitle', data=job_counts)
    plt.xlabel('Location')
    plt.ylabel('IT Field Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add tooltips to display count number on hover
    ax = barplot.axes
    bars = ax.patches
    counts = job_counts['JobTitle'].tolist()

    # Add count number annotations to each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')

    # Display the bar chart with tooltips on Streamlit dashboard
    st.pyplot(plt)

st.write("-----")
#--------------------------------------------------------------------------------------------------------------------------------
if st.checkbox("Job Postings-wise Experience Level Count"):
    st.markdown('### Job Postings-wise Experience Level Count (Select IT Trend)')

    # Group the data by experience and count the occurrences
    experience_counts = job_df.groupby('Experience').size().reset_index(name='Count')

    # Sort the experience levels by count in descending order
    experience_counts = experience_counts.sort_values('Count', ascending=False)

    # Create the scatter plot
    chart = alt.Chart(experience_counts).mark_circle().encode(
        y=alt.Y('Experience:O', title='Experience Level', sort='-x'),
        x=alt.X('Count:Q', title='Count', scale=alt.Scale(zero=False)),
        size=alt.Size('Count:Q', title='Count', scale=alt.Scale(range=[50, 500])),
        color=alt.Color('Count:Q', legend=None, scale=alt.Scale(scheme='viridis')),
        tooltip=['Experience:O', 'Count:Q']
    ).properties(
        width=600,
        height=400,
    )

    # Display the scatter plot on Streamlit dashboard
    st.altair_chart(chart, use_container_width=True)

st.write("-----")
#--------------------------------------------------------------------------------------------------------------------------------
if st.checkbox("Job Postings-wise Job_Type_I & Job_Type_II"):
    jobI_col, jobII_col=st.columns(2)
    with jobI_col:
        st.markdown('### Job Postings-wise Job_Type_I (Select IT Trend)')

        # Group the data by Job_Type_I and count the occurrences
        Job_Type_I_counts = job_df.groupby('Job_Type_I').size().reset_index(name='Count')

        # Sort the Job_Type_I levels by count in descending order
        Job_Type_I_counts = Job_Type_I_counts.sort_values('Count', ascending=False)

        # Create the horizontal bar chart
        chart = alt.Chart(Job_Type_I_counts).mark_bar().encode(
            x=alt.X('Count:Q', title='Count'),
            y=alt.Y('Job_Type_I:N', title='Job_Type_I', sort='-x', axis=alt.Axis(labelLimit=200)),
            color=alt.Color('Job_Type_I:N', legend=None),
            tooltip=['Job_Type_I:N', 'Count:Q']
        ).properties(
            width=600,
            height=400
        )

        # Display the horizontal bar chart on Streamlit dashboard
        st.altair_chart(chart, use_container_width=True)

    with jobII_col:
        st.markdown('### Job Postings-wise Job_Type_II (Select IT Trend)')

        # Group the data by Job_Type_II and count the occurrences
        Job_Type_II_counts = job_df.groupby('Job_Type_II').size().reset_index(name='Count')

        # Sort the Job_Type_II levels by count in descending order
        Job_Type_II_counts = Job_Type_II_counts.sort_values('Count', ascending=False)

        # Create the horizontal bar chart
        chart = alt.Chart(Job_Type_II_counts).mark_bar().encode(
            x=alt.X('Count:Q', title='Count'),
            y=alt.Y('Job_Type_II:N', title='Job_Type_II', sort='-x', axis=alt.Axis(labelLimit=200)),
            color=alt.Color('Job_Type_II:N', legend=None),
            tooltip=['Job_Type_II:N', 'Count:Q']
        ).properties(
            width=600,
            height=400
        )

        # Display the horizontal bar chart on Streamlit dashboard
        st.altair_chart(chart, use_container_width=True)


st.write("-----")
#-----------------------------------------------------------------------------------------------------

st.sidebar.markdown('''
---
Group 25: Sana Parveen | Maryam Bashir | Sidra Fatima | Wasia Mukhtar
''')
