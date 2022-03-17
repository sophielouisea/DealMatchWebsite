from cmath import nan
from shutil import copyfileobj
import streamlit as st
import requests
import pandas as pd
import numpy as np

# ---- App configurations
st.set_page_config(title="DealMatch - Recommendation Engine")


# ---- Welcome text
st.markdown(f"""
         # DealMatch - Recommendation Engine
         Welcome to the recommender app {st.image('data:image/png;base64,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', width=25)}. \n
         Let's match investors and target companies for a higher chance of a successful deal!
         """)

# ---- Create an expandable "Help" section for instructions
with st.expander("Help!"):
    st.markdown("""

             To generate a recommendation, open the sidebar, fill in the target information and click on "Display recommendations".

             If you get an error, double-check that you filled in the data accurately.

             The required input data types are:\n

             - Deal ID: number
             - Deal name: words and/or number
             - Target company ID: number
             - Target name: words and/or number
             - Target description: words and/or number
             - Target revenue: number
             - Target EBITDA: number
             - Target EBIT: number
             - Keywords: words, separated by a comma

             """)


# ---- Sidebar with form to get new target information
st.sidebar.image(
    'https://c.smartrecruiters.com/sr-company-logo-prod-dc5/6168ff832f8fb46fc18533dc/huge?r=s3-eu-central-1&_1634271911128')


st.sidebar.markdown("""
                    ***Enter the target information below to generate \
                        recommendations.***
                    """)

deal_id = st.sidebar.text_input('Deal ID')
deal_name = st.sidebar.text_input('Deal name')
deal_type_name = st.sidebar.selectbox(
    'Deal type', ('DISTRESSED', 'MAJORITY', 'MINORITY', 'OTHER', 'VC'))
target_company_id = st.sidebar.text_input('Target company ID')
target_name = st.sidebar.text_input('Target name')
target_description = st.sidebar.text_input('Target description')
target_revenue = st.sidebar.text_input('Target revenue')
target_ebitda = st.sidebar.text_input('Target EBITDA')
target_ebit = st.sidebar.text_input('Target EBIT')
country_name = st.sidebar.selectbox('Target country', ('Austria',
                                                       'Belgium',
                                                       'Bulgaria',
                                                       'Czechia',
                                                       'Egypt',
                                                       'Germany',
                                                       'Italy',
                                                       'Netherlands',
                                                       'Poland',
                                                       'Portugal',
                                                       'Romania',
                                                       'Slovakia',
                                                       'Spain',
                                                       'Switzerland',
                                                       'United States of America'))
region_name = st.sidebar.selectbox('Target region', ('Baden-Württemberg',
                                   'Bavaria',
                                   'Berlin',
                                   'Brandenburg',
                                   'Bremen',
                                   'Hamburg',
                                   'Hesse',
                                   'Lower Saxony',
                                   'Mecklenburg-Vorpommern',
                                   'North Rhine-Westphalia',
                                   'Rhineland-Palatinate',
                                   'Saarland',
                                   'Saxony',
                                   'Saxony-Anhalt',
                                   'Schleswig-Holstein',
                                   'Thuringia'))
sector_name = st.sidebar.selectbox('Target sector', ('Agriculture',
                                   'Automotive',
                                   'Biotechnology & Life Sciences',
                                   'Chemicals',
                                   'Computer Hardware & Equipment',
                                   'Construction',
                                   'Consumer Goods & Apparel',
                                   'Defense',
                                   'Electronics',
                                   'Energy',
                                   'Financial Services',
                                   'Food & Beverages',
                                   'Food & Staples Retailing',
                                   'Health Care Equipment & Services',
                                   'IT services',
                                   'Industrial automation',
                                   'Industrial products and services',
                                   'Insurance',
                                   'Internet/ecommerce',
                                   'Leisure & consumer services',
                                   'Manufacturing (other)',
                                   'Media',
                                   'Mining',
                                   'Pharmaceuticals',
                                   'Professional Services (B2B)',
                                   'Real Estate',
                                   'Retailing',
                                   'Semiconductors & Semiconductor Equipment',
                                   'Software & Services',
                                   'Telecommunication Hardware',
                                   'Telecommunication Services',
                                   'Transportation',
                                   'Utilities'))
strs = st.sidebar.text_input('Keywords')
display = st.sidebar.button('Display recommendations')


# ---- Display recommendations list when clicking on display button
if display:
    st.markdown("""
            **Your results**
            """)

    api_url = f'https://dealmatch-rec3-jlx73eg7oq-ew.a.run.app/recommend?deal_id={deal_id}&deal_name={deal_name}&deal_type_name={deal_type_name}&target_company_id={target_company_id}&target_name={target_name}&target_description={target_description}&target_revenue={target_revenue}&target_ebitda={target_ebitda}&target_ebit={target_ebit}&country_name={country_name}&region_name={region_name}&sector_name={sector_name}&strs={strs}'
    response = requests.get(api_url).json()
    print(response)
    response_df = pd.DataFrame(
        {'name': list(response['name'].values()),
         'match_probability': list(response['match_probability'].values()),
         'description': list(response['description'].values()),
         'distance_target<=>investor': list(response['distance_target<=>investor'].values()),
         'Rationale': list(response['Rationale'].values())})
    response_df = pd.DataFrame(response_df, index=list(range(0, response_df.shape[0])))
    response_df.index = np.arange(1, len(response_df)+1)
    st.dataframe(response_df)

    def convert_results(df):
        return df.to_csv().encode('utf-8')

    csv_to_download = convert_results(response_df)

    st.download_button(
        label="Download results",
        data=csv_to_download,
        file_name='deal_match_results.csv'
    )
