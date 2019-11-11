import pandas as pd
import streamlit as st
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from pywaffle import Waffle
import seaborn as sns
from lifelines.statistics import multivariate_logrank_test
from matplotlib.offsetbox import AnchoredText
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from figures import *



# Set outline
pages = ["1. Introduction to survival analysis",
         "2. Explore the dataset",
         "3. Estimate the overall survival function",
         "4. The proportional hazard assumption",
         "5. Visual inspection of proportional hazards",
         "6. Statistical testing of the proportional hazard assumption",
         "7. Identify factors affecting survival probability",
         "8. Where to go from here?"]

page = st.sidebar.selectbox('Navigate', options=pages)



# Test proportional hazard assumption
# Import data
df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

# Do not encode Churn to 0/1 here, as it would interfere with data exploration tab

# For plotting
## Set colour dictionary for consistent colour coding of KM curves
colours = {'Yes':'g', 'No':'r',
           'Female':'b', 'Male':'y',
           'Month-to-month':'#007f0e', 'Two year':'#c4507c', 'One year':'#feba9e',
           'DSL':'#ad53cd', 'Fiber optic':'#33ccff',
           'Electronic check':'#33cc33', 'Mailed check':'#ff8000', 'Bank transfer (automatic)':'#9933ff', 'Credit card (automatic)':'#ff66b3',
           "$0-29.4":'#ff3333', "$29.4-56":'#55ff00', "$56-68.8":'#1a8cff', "$68.8-107":'#ffff4d', "$107-118.75":'#8f5f7f',
           '40-65m':'#d04813', '5-20m':'#97fa83', '20-40m':'#38ccde', '0-5m':'#6d8d97', '65-72m':'#a053b4'}

## Plot KM curve for each categorical variable
def categorical_km_curves(feature, t='Tenure', event='Churn', df=df, ax=None):
    for cat in sorted(df[feature].unique(), reverse=True):
        idx = df[feature] == cat
        kmf = KaplanMeierFitter()
        kmf.fit(df[idx][t], event_observed=df[idx][event], label=cat)
        kmf.plot(ax=ax, label=cat, ci_show=True, c=colours[cat])


# Each page
if page == pages[0]:
    st.sidebar.markdown('''
    
    ---
    
    In many applications of data analysis, we are often interested in how long it takes before an event occurs, 
    such as patient death in medicine, customer churn in business or equipment failure in engineering. 
    **Survival analysis** is a well-established method to characterize the probability of time-dependent events occuring,
     and, most importantly, the contribution of various factors in modifying this probability.
     
     In this series, we will put together a starter kit of essential concepts and tools 
     for performing survival analysis using Python. As survival analysis commonly used in analyzing customer churn patterns, 
     we will use the [Telco customer churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) available from IBM. 
    ''')

    st.image('./study.png', caption='Image credit: Icons 8', use_column_width=True)

    st.markdown('''
        ###
        
        ## What is it?
        Survival analysis is based on two inversely related functions that model time-to-event data:
        - **Survival function**: the probability that the event of interest does not happen at a given time *t*
        - **Hazard function**: the instantaneous probability of the event of interest occurring at a given time *t*, given that it has not happened yet
        
        ## Why use it?
        Survival analysis helps to answer questions like:
        - What is the probability that the event of interest happens in three years from now?
        - What is the impact of certain factors on the probability of the event happening?
        - Are there differences in the probability of the event between different groups of subects?
    
        Also, survival analysis is needed as regular regression methods cannot handle time-to-event data, which tend to have a heavily skewed distribution and be right censored, where the event of interest is not observed for a subject under study, potentially due to the event having not yet happened by the end of the study period, subject dropout, etc.
        ''')

    st.image('./censor.png', use_column_width=True, caption="Image credit: https://www.karlin.mff.cuni.cz/~pesta/NMFM404/survival.html")

    st.markdown('''
        ## When to use it?
        
        ### Areas of application
        - **Cancer studies**: patients survival time analyses
        - **Business**: customer churn analysis
        - **Sociology**: event-history analysis
        - **Engineering**: failure-time analysis
        
        ### Requirements on the input data
        Valid interpretation of survival analysis results rests on the assumption that the censoring of observations is random, i.e. has no relationship to the event of interest or biased by any factor.
    
        Some analyses also rest upon the proprotional hazards assumption, which assumes that the effects of variables on the probability of the event of interest occurring remain constant with respect to time.
        
        ## Who should be involved?
        Should examine the reasons for censoring and censoring patterns with whoever was involved in data collection. Valid interpretation of the analysis results rest on the assumption that censoring is random and unrelated to the event of interest.
        
        ## How to use it?
        Three most common methods used in survival analysis are:
        - **Kaplan-Meier plots** to visualize survival curves
        - **Log-rank test** to compare the survival curves of two or more groups
        - **Cox proportional hazards regression** to describe the effect of variables on survival
        
        ''')

if page == pages[1]:
    st.sidebar.markdown('''
    
    ---
    
    Before diving into survival analysis, it is important to get familiar with the dataset. 
    
    Select any two (can be the same) variables in the dropdown menus below
    to create a exploratory visualization.
    ''')

    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    cat_list = sorted([' ', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'Churn', 'Tenure'])

    var1 = st.sidebar.selectbox("Select variable 1", cat_list)

    var2 = st.sidebar.selectbox("Select variable 2", cat_list)



    plt.rcParams.update({'axes.labelpad': 15, 'axes.labelsize': 18, 'xtick.labelsize':16, 'ytick.labelsize': 14,
                         'legend.title_fontsize': 20, 'legend.loc':'best', 'legend.fontsize':'large',
                         'figure.figsize':(10, 8)})

    if var1 == ' ' and var2 == ' ':
        '''
        Click on any of the categorical variable names to expand the
         sunburt chart and see distribution of the levels in more detail, where the size of each leaf is proportional 
         to the number of customers in that level.
        '''

        with st.spinner('Working on it...'):

            fig = sunburst_fig()

            st.plotly_chart(fig, width=700, height=700)
    elif var1 != ' ' and (var2 == ' ' or var2 == var1) and df[var1].dtype == 'object':
        '''
        There are 7,043 customers in the dataset. Each symbol represents ~100 customers.
        '''

        with st.spinner('Working on it...'):
            data = df[var1].value_counts().to_dict()

            fig = plt.figure(
                FigureClass=Waffle,
                rows=5,
                columns=14,
                values=data,
                legend={'loc': 'center', 'bbox_to_anchor': (0.5, 1.2), "fontsize":20, 'ncol':2},
                icons='user',
                font_size=38,
                icon_legend=True,
                figsize=(12, 8)
            )

            # plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif var1 != ' ' and var2 == ' ' and df[var1].dtype != 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, nbins=n_bins, width=1200)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              )
                              )

            st.plotly_chart(fig)
    elif var1 != var2 and df[var1].dtype == 'object' and df[var2].dtype == 'object':
        with st.spinner('Working on it...'):
            fig = sns.countplot(x=var1, hue=var2, data=df, palette="Set3")

            plt.ylabel('Count')

            if var1 == 'PaymentMethod':
                plt.xticks(rotation=30, ha="right")
            else:
                pass

            plt.grid(False)

            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype == 'object':
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=var2, opacity=0.4, barmode = 'overlay', nbins=n_bins, width=1000)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              legend_orientation="h",
                              legend=dict(x=0, y=1.1),
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              ))

            st.plotly_chart(fig)
    elif df[var1].dtype == 'object' and df[var2].dtype != 'object':
        with st.spinner('Working on it...'):
            # sns.set(style='ticks', font_scale=2.2,
            #         rc={'figure.figsize':(18, 16), 'axes.labelpad':30})

            fig = sns.barplot(x=var1, y=var2, data=df, palette="Set3")

            if var1 == 'PaymentMethod':
                plt.xticks(rotation=20, ha="right")
            else:
                pass


            plt.tight_layout()

            st.pyplot()

            plt.clf()
    elif df[var1].dtype != 'object' and df[var2].dtype != 'object' and var1 == var2:
        n_bins = st.slider("Number of bins",
                           min_value=10, max_value=50, value=10, step=2)

        with st.spinner('Working on it...'):
            fig = px.histogram(df, x=var1, color=None, nbins=n_bins, width=1000)

            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              legend_orientation="h",
                              legend=dict(x=0, y=1.1),
                              yaxis=go.layout.YAxis(
                                  title=go.layout.yaxis.Title(
                                      text="Count"
                                  )
                              ))

            st.plotly_chart(fig)
    elif var1 != var2 and df[var1].dtype != 'object' and df[var2].dtype != 'object':
        with st.spinner('Working on it...'):
            sns.set(style='ticks', font_scale=1.1, rc={'figure.figsize':(12, 6)})

            sns.jointplot(df[var1], df[var2], kind="hex", color="#4CB391")

            st.pyplot()

            plt.clf()

    else:
        pass






if page == pages[2]:
    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way? 
    
    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    **Need more info**?
    - [`lifelines` documentation - Estimating univariate survival models using Kaplan-Meier](https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html)
    - [More background on the Kaplan-Meier method](http://www.sthda.com/english/wiki/survival-analysis-basics#kaplan-meier-survival-estimate)  
       
    ''')

    #hide_code = st.sidebar.checkbox("Hide source code")

    # Import data
    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

    st.markdown('''
    At its core, survival analysis aims to estimate how much time it would take before a particular 
    event happens. Data required for survival analysis is collected over a fixed-length observation period 
    for a population of subjects, whether they be patients, customers or machinery, by noting 1) **whether** 
    the event of interest occurs for each object, and if so, 2) **when** does it occur relative to the start 
    of the observation period.

    These two parameters allow estimation of the **survival function**, which is one of the two key functions
     in survival analysis (the other is hazard function and will be covered in the next post). The survival 
     function computes, at each time *t*, the probability that the event of interest *does not* happen. 
    ''')


    st.header('3.1 Import and preprocess data')

    st.markdown('''
    We will first import the Telco customer churn dataset, which has been cleaned up, and re-encode the outcome column 
    "Churn" in binary form (0/1) as required by the `lifelines` package.
    
    ```python
    ## Import data
    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    ## Re-encode "Churn" as 0/1
    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)
    ```
    ''')


    if st.checkbox("Preview data"):
        st.dataframe(df.head(10))

        st.header('3.2 Plot the Kaplan-Meier survival curve')

        st.markdown('''
        In reality, the actual survival function of a population cannot be observed. Instead, it is estimated by
           the **Kaplan-Meier (KM) estimator**, which essentially calculates the proportion of at-risk
            subjects that has not yet "succumbed" (to death, malfunction, etc) out of all at-risk subjects present at
             each time *t*. Being a non-parametric method, the Kaplan-Meier estimator does not make assumptions about
         the distribution of survival times and any specific relationships between covariates and 
         time to the event of interest.
         
        The `lifelines` API is very similar to that of `scikit-learn`, where the `KaplanMeierFitter` object is 
        first instantiated and then fitted to the data.
    
        ```python
        ## Instantiate kmf object
        kmf = KaplanMeierFitter()

        ## Fit kmf object to data
        kmf.fit(df['Tenure'], event_observed = df['Churn'])

        ## Plot KM curve
        ax = kmf.plot(xlim=(0, 75), ylim=(0, 1))
        ax.set_title("Overall survival probability")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Survival probability")
        ```
         
         ''')



        if st.checkbox("Plot!"):
            ## Instantiate kmf object
            kmf = KaplanMeierFitter()

            ## Fit kmf object to data
            kmf.fit(df['Tenure'], event_observed = df['Churn'])

            ## Plot KM curve
            ax = kmf.plot(xlim=(0, 75), ylim=(0, 1))
            ax.set_title("Overall survival probability")
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Survival probability")

            st.pyplot()

            plt.clf()

            st.markdown('''
            The y-axis represents the probability that a
             customer is still subscribed to the company's services at a given time (x-axis) since they first signed on.
             
              We see that the probability of a given customer leaving decreases (the curve flattens) with time, 
              consistent with what we saw previously showing that the probability of churn decreases as the customer 
              tenure increases. The coloured band superimposed on the KM curve is the 95% confidence interval.
            ''')


if page == pages[3]:
    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way? 
    
    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    **Need more info?**
    - [Succinct introduction to Cox proportional hazard model and underlying assumption](https://journals.lww.com/anesthesia-analgesia/Fulltext/2018/09000/Survival_Analysis_and_Interpretation_of.32.aspx#JCL8)
    - [`lifelines` documentation - Testing the proportional hazard assumptions ](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model)
    - [Detailed walkthrough (in R) of the methods and concepts for evaluating assumptions underlying the Cox regression model](http://www.sthda.com/english/wiki/cox-model-assumptions)
    ''')


    st.markdown('''
    One of the key insights that survival analysis provides is the effect of 
    various variables on the time-dependent probability of an event of interest occurring. 
    Some of the most commonly used techniques in survival analysis rest upon the critical **proportional hazards assumption**, 
    which dictates that **the effect of each variable on probability of the event of
     interest occurring remains the same with respect to time**. For variables that violate 
     this assumption, which is very likely, the results of Cox regression model regarding *cannot* be considered valid.
     Nevertheless, determining which factors have time-varying effects can be quite useful in itself 
     in terms of gaining insights into the data.
    
    Imagine that the schematics below plot the number of patients that have died (y-axis) against the
     length of time since diagnosis (x-axis), with male and female patients indicated by green and red,
      respectively. 
    ''')

    st.image('./proportional_hazard.png', use_column_width=True,
             caption="Image credit: https://altis.com.au/a-crash-course-in-survival-analysis-customer-churn-part-iii/")

    st.markdown('''
    On the left, we see what the plot would look like if the gender variable meets the
       proportional hazards assumption, as being male or female has the same effect on patient mortality 
       across all time points examined. Recall that the hazard is the instantaneous probability of the 
       event of interest occurring at a given time *t*, given that it has not happened yet. 
       So in other words, the probability of dying in female and male patients are proportional to each other.
        On the right, we see what it would look like if the assumption is not met, where being
         female has a protective effect at early stages of the disease, but becomes detrimental later on.
         
    As a **rule of thumb**, if the the survival (or hazard) curves of two groups stratified by a variable intersect 
    (like the plot on the right), then most likely the proportional hazards assumption is *violated* for that variable.

    ''')




if page == pages[4]:
    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way? 
    
    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    **Need more info?**
    - [Spotting proportional hazard assumption violations from Kaplan-Meier curves](https://www.theanalysisfactor.com/assumptions-cox-regression/)
   ''')

    #hide_code = st.sidebar.checkbox("Hide source code")

    st.markdown('''
    One of the ways we can get some preliminary ideas as to which factors violate the proportional
     hazard assumption is by plotting and examining Kaplan-Meier survival curves that are stratified by each variable.
     
    ''')

    st.header('5.1 Supervised discretization of continuous variable')

    st.markdown('''
     As Kaplan-Meier estimation of the survival function cannot characterize the probability of event occurrence 
     in relation to a continuous variable, we need to bin the continuous variable `MonthlyCharges` into discrete levels, 
     so that we can examine the probability of customer churn at different "tiers" of customer spending. 
     The best tool that I found for this is the R package `arulesCBA`, which uses a supervised approach to
      identify bin breaks that are most informative with respect to a class label, which is `Churn` in this dataset.
       We will ignore the `Tenure` variable, as it is already part of the survival function.
     
     
     ''')

    disc_df = pd.read_csv('./disc_churn.csv')


    st.markdown(
    '''
    ```r
    ## Import library
    library(plyr)
    library(dplyr)
    library(arulesCBA)
    
    ## Import data
    df <- read.csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")
    
    ## Encode "Churn" as 0/1
    df <- df %>%
            mutate(Churn = ifelse(Churn == "No",0,1)) %>%
            mutate(Churn = as.factor(Churn))
    
    ## Discretize "MonthlyCharges" with respect to "Churn"/"No Churn" label and assign to new column in dataframe
    df$Binned_MonthlyCharges <- discretizeDF.supervised(Churn ~ ., 
                                                        df[, c('MonthlyCharges', 'Churn')], method='mdlp')$MonthlyCharges
    
    ## Rename the levels based on knowledge of min/max monthly charges
    df$Binned_MonthlyCharges = revalue(df$Binned_MonthlyCharges,
                                        c("[-Inf,29.4)"="$0-29.4", 
                                          "[29.4,56)"="$29.4-56",
                                          "[56,68.8)"="$56-68.8",
                                          "[68.8,107)"="$68.8-107",
                                          "[107, Inf]" = "$107-118.75"))
    ```
    ''')

    if st.checkbox("Preview discretized data"):
        with st.spinner('Working on it...'):

            st.dataframe(disc_df.head(10))

            st.markdown('''
            Scrolling all the way to the right, we see that there is a new column that contains binned `MonthlyCharges` values.
            ''')

            disc_df['Churn'] = np.where(disc_df['Churn'] == 1, "Yes", "No")

            x = pd.crosstab(disc_df['Binned_MonthlyCharges'], disc_df['Churn'])

            fig = x.loc[['$0-29.4', "$29.4-56", "$56-68.8", "$68.8-107", "$107-118.75"]].plot.barh(stacked=True,
                                                                                                   figsize=(13, 9),
                                                                                                   fontsize=18)

            fig.set_xlabel("Count", fontsize=20)
            fig.set_ylabel("Monthly fee")
            plt.legend(fontsize=20)

            st.markdown('''
            Since these bins are identified as being most "informative" with respect to the target variable `Churn`, let's see the proportion of 
            churned/not churned customers at each monthly fee range:
            ''')

            st.pyplot()

            plt.clf()

            st.markdown('''
            We see that both plots identified two groups of customers that are more likely to churn, 
            one group paying $26-56/month and another paying $68-106/month. As we had mentioned previously, 
            this could be of interest for the company as these may reflect uncompetitive pricing that should be adjusted.
             Most interestingly, most customers are in the $68-106/month tier, which poses a potentially significant 
             problem as these customers are also much more likely to churn than the rest.
             
             Getting back on track, now we can plot stratified Kaplan-Meier survival curves for each variable.
            ''')

        st.header("5.2 Plot stratified Kaplan-Meier curves")

        st.markdown('''
        As mentioned in the previous section, crossing of the curves for a factor is an indication that the 
        proportional hazard assumption is violated. So let's take a look.
        
        The code below is adapted from work by [Zach Angell](https://medium.com/@zachary.james.angell/applying-survival-analysis-to-customer-churn-40b5a809b05a).
        ''')

        st.markdown(
            '''
            ```python
            ## Set up subplot grid
            fig, axes = plt.subplots(nrows = 9, ncols = 2,
                                     sharex = True, sharey = True,
                                     figsize=(10, 35))
                                     
            ## Set colour dictionary for consistent colour coding of KM curves
            colours = {'Yes':'g', 'No':'r',
                       'Female':'b', 'Male':'y',
                       'Month-to-month':'#007f0e', 'Two year':'#c4507c', 'One year':'#feba9e',
                       'DSL':'#ad53cd', 'Fiber optic':'#33ccff',
                       'Electronic check':'#33cc33', 'Mailed check':'#ff8000', 'Bank transfer (automatic)':'#9933ff', 'Credit card (automatic)':'#ff66b3',
                       "$0-29.4":'#ff3333', "$29.4-56":'#55ff00', "$56-68.8":'#1a8cff', "$68.8-107":'#ffff4d', "$107-118.75":'#8f5f7f',
                       '40-65m':'#d04813', '5-20m':'#97fa83', '20-40m':'#38ccde', '0-5m':'#6d8d97', '65-72m':'#a053b4'}
    
            ## Plot KM curve for each categorical variable
            def categorical_km_curves(feature, t='Tenure', event='Churn', df=df, ax=None):
                for cat in sorted(df[feature].unique(), reverse=True):
                    idx = df[feature] == cat
                    kmf = KaplanMeierFitter()
                    kmf.fit(df[idx][t], event_observed=df[idx][event], label=cat)
                    kmf.plot(ax=ax, label=cat, ci_show=True, c=colours[cat])
    
            col_list = df.drop(['Churn', 'Tenure', 'MonthlyCharges', 'TotalCharges'], axis=1).columns
            
            for cat, ax in zip(col_list, axes.flatten()):
                categorical_km_curves(feature=cat, t='Tenure', event='Churn', df = df, ax=ax)
                ax.legend(loc='lower left', prop=dict(size=14))
                ax.set_title(cat, fontsize=18)
                ax.set_xlabel('Tenure (months)')
                ax.set_ylabel('Survival probability')
    
            fig.subplots_adjust(top=0.97)
            ```
            ''')

        if st.checkbox("Plot everything!"):
            with st.spinner('Working on it... This one is bit of a doozy...'):

                disc_df['Churn'] = np.where(disc_df['Churn'] == 'Yes', 1, 0)

                ## Set up subplot grid
                fig, axes = plt.subplots(nrows = 9, ncols = 2,
                                         sharex = True, sharey = True,
                                         figsize=(10, 35)
                                         )

                col_list = disc_df.drop(['Churn', 'Tenure', 'MonthlyCharges', 'TotalCharges'], axis=1).columns

                plt.rcParams.update({'axes.labelpad': 10})

                for cat, ax in zip(col_list, axes.flatten()):
                    categorical_km_curves(feature=cat, t='Tenure', event='Churn', df = disc_df, ax=ax)
                    ax.legend(loc='lower left', prop=dict(size=14))
                    ax.set_title(cat, fontsize=18)
                    #p = multivariate_logrank_test(df['Tenure'], df[cat], df['Churn'])
                    #ax.add_artist(AnchoredText(p.p_value, loc='upper right', frameon=False))
                    ax.set_xlabel('Tenure (months)', fontsize=16)
                    ax.set_ylabel('Survival probability', fontsize=16)
                    ax.tick_params(labelsize=12)

                fig.delaxes(axes[8][1])

                fig.subplots_adjust(top=0.97)

                fig.tight_layout()

                st.pyplot()

                plt.clf()

                st.markdown('''
                For each variable, the curve(s) that decline faster to 0% survival probability 
                represent population subsets that are more likely to stop buying the company's services.
                
                We see that the stratified Kaplan-Meier curves cross for `StreamingTV`, `StreamingMovies`, `MultipleLines` and `MonthlyCharges`,
                suggesting that these factors most likely do not meet the proportional hazard assumption. Nevertheless, this is 
                useful information, as it shows that purchasing these services and/or having high monthly fees
                 might contribute to the propensity of long-term customers to leave the company as time goes on.
                ''')




if page == pages[5]:

    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way? 
    
    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    **Need more info?**
    - [`lifelines` documentation - Testing the proportional hazard assumptions ](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model)
    - [Detailed walkthrough (in R) of the methods and concepts for evaluating assumptions underlying the Cox regression model](http://www.sthda.com/english/wiki/cox-model-assumptions)
    
    ''')


    #hide_code = st.sidebar.checkbox("Hide source code")

    # Import data
    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

    st.markdown('''
    A well-established approach to test the proportional hazards assumption is based on **scaled Schoenfeld residuals**, 
    which is independent of time if the assumption holds. Therefore, for any given covariate, a significant 
    relationship between the Schoenfeld residuals and time indicates its effect on hazard is time-dependent.
    
    First, we will use label encoding to quickly convert categorical variables to numerical encoding, 
    as required by the `lifelines` package. Then, like what we did to plot Kaplan-Meier curves, the `CoxPHFitter` object is 
    first instantiated and fitted to the data in order to test the proportional hazard assumption.
    ''')

    st.markdown('''
    ```Python
    ## Label encode categorical features
    cat_col = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    le = LabelEncoder()
    
    df[cat_col] = df[cat_col].apply(le.fit_transform)
    
    ## Instantiate object
    cph = CoxPHFitter()
    
    ## Fit to data
    cph.fit(df, 'Tenure', 'Churn', show_progress=False)
    
    ## Test assumption
    results = proportional_hazard_test(cph, df, time_transform='all')        
    ''')

    if st.checkbox("Test the proportional hazard assumption"):
        with st.spinner('Working on it...'):
            ## Label encode categorical features
            cat_col = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

            le = LabelEncoder()

            df[cat_col] = df[cat_col].apply(le.fit_transform)

            ## Instantiate object
            cph = CoxPHFitter()

            ## Fit to data
            cph.fit(df, 'Tenure', 'Churn', show_progress=False)

            ## Test assumption
            results = proportional_hazard_test(cph, df, time_transform='all')

            y = results.summary

            y = y.drop('test_statistic', axis=1)

            x = y.unstack()

            x.columns = x.columns.droplevel()

            x['row_mean'] = x.mean(axis=1)

            x = x.sort_values(by='row_mean')

            st.markdown('''
            When testing the proportional hazard assumption, the `lifelines` package offers several transform methods for the "time" parameter, 
            all of which we will use here for comparison. Calling the `proportional_hazard_test()` function on the fitted `CoxPHFitter` object produces 
            a dataframe that contains the p-value for each variable in the proportional hazard assumption test for various transformations of 
            time, where p < 0.05 indicates that the assumption does not hold true for the variable.
            
            For ease of viewing, let's make a heat map of these results:
    
            ''')

            with st.spinner('Hang on...'):
                sns.set(rc={'figure.figsize':(15, 10)})

                fig2 = sns.heatmap(x.drop('row_mean', axis=1), annot=True, annot_kws={"size": 20}, cbar=False).tick_params(labelsize=22)

                plt.tight_layout()

                st.pyplot()

                plt.clf()

                st.markdown('''
                Consistent with what we saw previously with the stratified Kaplan-Meier curves, the proportional hazard assumption 
                does not hold true for the variables`StreamingTV`, `StreamingMovies` and `MonthlyCharges`. Statistical testing also
                identified several more factors that fail to satisfy this assumption, such as `Contract`, `InternetService` and `PhoneService`. 
                
                To be on the safe side, in the next section, we will carry out survival regression on *only* the variables that satisfy the proportional 
                hazard assumption for any transformation of the "time" parameter.
                
                For the rest, there are several ways to deal with factors that have time-varying effects, including stratification, adding time-varying covariates, etc.
                This is an advanced topic that will not be covered here. For more information, please see the `lifelines` package [documentation](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html).
                ''')



if page == pages[6]:
    st.sidebar.markdown('''
    
    ---
    
    Feeling overwhelmed by all the new info coming your way? 
    
    No fear! Follow the checkboxes to run one code chunk at a time and progressively reveal new content!
    
    ---
    
    **Need more info?**
    - [`lifelines` documentation - Survival regression](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html)
    - [Evaluating (in R) and interpreting hazard ratios from survival regression](http://www.sthda.com/english/wiki/cox-proportional-hazards-model)
    ''')

    #hide_code = st.sidebar.checkbox("Hide source code")

    # Import data
    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

    ## Label encode categorical features
    cat_col = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

    le = LabelEncoder()

    df[cat_col] = df[cat_col].apply(le.fit_transform)

    ## Instantiate object
    cph = CoxPHFitter()

    ## Fit to data
    cph.fit(df, 'Tenure', 'Churn', show_progress=False)

    ## Test assumption
    results = proportional_hazard_test(cph, df, time_transform='all')

    y = results.summary

    y = y.drop('test_statistic', axis=1)

    x = y.unstack()

    x.columns = x.columns.droplevel()

    x['row_mean'] = x.mean(axis=1)

    x = x.sort_values(by='row_mean')

    t = x[~(x < 0.05).any(1)]

    col_list = list(t.index.values)



    st.markdown('''
    Here we will use the Cox regression model to more closely examine the effects of variables on the survival probability. 
    In the previous section, we identified seven variables that satisfy the proportional hazard assumption:
    ''')

    st.write(col_list)

    st.markdown('''
    Once again, it is very important to remember that the results of survival regression is **not** valid for 
    variables that fail the proportional hazard test. So let's run survival regression on these seven factors that are shown to 
    have time-invariant effects on customer churn:
    ''')

    st.markdown('''
    ```Python
    cph = CoxPHFitter()
    
    col_list.extend(['Churn', 'Tenure'])

    cph.fit(df[col_list], duration_col='Tenure', event_col='Churn')

    cph.plot()
        
    ''')

    if st.checkbox("Fit Cox model"):
        with st.spinner('Construction in progress...'):
            cph = CoxPHFitter()

            col_list.extend(['Churn', 'Tenure'])

            cph.fit(df[col_list], duration_col='Tenure', event_col='Churn')

            plt.style.use('seaborn-ticks')

            fig = cph.plot()

            plt.tick_params(axis='both', which='major', labelsize=20)

            plt.xlabel("Log Hazard Ratio", fontsize=22)

            plt.tight_layout()

            st.pyplot(height=400)

            plt.clf()


        st.markdown('''
        The **hazard ratio** (HR) of each variable provides a measure of how it influences the probability of the event of interest, in 
        this case customer churn, happening at a given point in time. 
        
        - HR < 1 --> *log(HR) < 0* --> the variable **reduces** the probability
        - HR = 1 --> *log(HR) = 0* --> the variable has **no** effect
        - HR > 1 --> *log(HR) > 0* --> the variable **increases** the probability
                
        Let's start with the most obvious bits: 
        - Customers who have dependents (`Dependents`) and have purchased `OnlineBackup`, `TechSupport` and `OnlineSecurity` 
        services are *less* likely to churn than their counterparts who do not
        - Customers that have `PaperlessBilling` are *more* likely to churn from the company
        - Male and female customers appear to have the same likelihood of leaving
        
        What is harder to interpret is how `PaymentMethod` is also associated with higher probability of customer churn, 
        as it has four possible levels: `Mailed check` [sic], `Electronic check` [sic], `Credit card (automatic)` 
        and `Bank transfer (automatic)`. 
        
        So, for more information, let's look at the stratified Kaplan-Meier curves for these variables:
        ''')

        if st.checkbox("Plot Kaplan-Meier curves"):
            with st.spinner("Coming right up..."):
                df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

                df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

                ## Set up subplot grid
                fig, axes = plt.subplots(nrows = 4, ncols = 2,
                                         sharex = True, sharey = True,
                                         figsize=(10, 15)
                                         )

                for cat, ax in zip(col_list[:-2], axes.flatten()):
                    categorical_km_curves(feature=cat, t='Tenure', event='Churn', df = df, ax=ax)
                    ax.legend(loc='lower left', prop=dict(size=14))
                    ax.set_title(cat, fontsize=18)
                    #p = multivariate_logrank_test(df['Tenure'], df[cat], df['Churn'])
                    #ax.add_artist(AnchoredText(p.p_value, loc='upper right', frameon=False))
                    ax.set_xlabel('Tenure (months)', fontsize=16)
                    ax.set_ylabel('Survival probability', fontsize=16)
                    ax.tick_params(direction='out', labelsize=12)

                plt.style.use('seaborn-ticks')

                fig.subplots_adjust(top=0.97)

                fig.delaxes(axes[3][1])

                fig.tight_layout()

                st.pyplot()

                plt.clf()

                st.markdown('''
                First off, we see that these stratified Kaplan-Meier curves confirm what the Cox regression model showed us above.
                Recall that for each variable, the curve(s) that decline faster to 0% survival probability represent population subsets 
                that are more likely to stop buying the company's services. We can see that customers (green curves) that 
                    have `Dependents` and purchased `OnlineBackup`, `OnlineSecurity` and `TechSupport` have more slowly declining curves than those 
                    who do not (red curves). The inverse is true for customers who have `PaperlessBilling`. The overlapping curves for male and female 
                    customers indicate that they have similar rates of churn as time goes on.
                
                Interestingly, it is customers who pay by `Electronic check` [sic] who are much more likely to churn than 
                those who pay by `Mailed check` [sic], `Credit card` or `Bank transfer`. Let's look at what makes these customers
                 "special" as compared to those paying by other means:  
                
                ''')

                if st.checkbox("Compare customers"):
                    df = pd.read_csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_renamed.csv")

                    x = df[df['PaymentMethod'] == 'Electronic check']

                    y = df[df['PaymentMethod'] != 'Electronic check']

                    def get_values(df, label):
                        o =[]

                        for i in df.columns:
                            if df[i].dtype == object:
                                o.append(df[i].value_counts().to_dict())

                        result = {}

                        for k in o:
                            result.update((k))

                        j = pd.DataFrame()

                        j['Category'] = [key for key, value in result.items() if not 'No' in key]

                        j[label] = [value/df.shape[0]*100 for key, value in result.items() if not 'No' in key]

                        return j

                    elec_res = get_values(x, "Electronic_cheq")

                    non_res = get_values(y, "Non_cheq")

                    final = pd.merge(elec_res, non_res, on='Category', how='inner')

                    # Reorder it following the values of the first value:
                    ordered_df = final.sort_values(by='Electronic_cheq')
                    my_range=range(1,len(final.index)+1)

                    plt.hlines(y=my_range, xmin=ordered_df['Electronic_cheq'], xmax=ordered_df['Non_cheq'],
                               color='grey', alpha=0.4)
                    plt.scatter(ordered_df['Electronic_cheq'], my_range, color='red', alpha=1, label='Customers using electronic cheque')
                    plt.scatter(ordered_df['Non_cheq'], my_range, color='green', alpha=1, label='Customers using other methods')
                    plt.legend(loc='lower right', prop={'size': 16})

                    # Add title and axis names
                    plt.yticks(my_range, ordered_df['Category'])
                    plt.xlabel('% customers in the group', fontsize=18)

                    plt.style.use('default')

                    plt.tick_params(axis='both', which='major', labelsize=14)

                    plt.tight_layout()

                    st.pyplot(height=300)

                    plt.clf()

                    st.markdown('''
                    Most strikingly, a **higher** percentage of customers who pay by electronic cheque `Month-to-month` contracts, 
                    purchase `Fiber optic` internet service, and have `PaperlessBilling` than that of customers who pay by other methods. These characteristics have been shown in this and [previous 
                    analyses](http://rpubs.com/nchelaru/famd) to be associated with higher likelihood of churn. Interestingly too, a **smaller** proportion of customers
                    paying by electronic cheque have purchased `OnlineSecurity` and have `Dependents` as compared to customers who do so also but pay by other means. 
                    [Previous analyses](http://rpubs.com/nchelaru/famd) have shown that these traits are associated with more "loyal" customers. 
                    
                    Taken together, all of these findings help us to create portraits of customers who need to be targeted for retention campaigns.
                    ''')


if page == pages[7]:

    st.sidebar.markdown('''
    
    ---
    
    You can find the source code for this app [@nchelaru](https://github.com/nchelaru/survival_analysis_streamlit).
     
    Curious to see what else I have cooked up? Head over to my portfolio, 
    [The Perennial Beginner](http://nancychelaru.rbind.io/portfolio/). 
    
    
    
    ''')

    st.balloons()

    '''
    ## Congratulations for making it to the finish line! 
    
    ### Now you have a starter-kit of key concepts and tools for performing survival analysis in Python.
    
    ###
    '''

    st.image('./survival_tools.png', caption='Image credit: Icons 8', use_column_width=True)

    '''
    ###
    
    If you are interested in trying everything out in R, head over to the excellent guide at [STHDA](http://www.sthda.com/english/wiki/survival-analysis) 
    on performing survival analysis using the fantastic `survival` package. 
    
    Finally, we need to approach a data science problem from multiple angles if we want to gain robust and nuanced insights 
    from the data. Check out what I gleaned from this customer churn dataset using [association rule mining](https://nancy-chelaru-centea.shinyapps.io/association_rule_mining/) and 
    [factor analysis of mixed data](https://rpubs.com/nchelaru/famd) methods, to see how those findings agree and differ from 
    those shown here.
    
    Hope you have enjoyed your stay! :)
    '''





