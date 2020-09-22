import base64
from io import StringIO
from os import getenv
from typing import List

import pandas as pd
import requests
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder

DISPLAY_ROWS = int(getenv('DISPLAY_ROWS', '5')) + 1
INPUT_DATA_MIN_ROWS = int(getenv('INPUT_DATA_MIN_ROWS', '1000'))


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    data = StringIO(str(data, 'utf-8'))

    return pd.read_csv(data)


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    _result = StringIO()

    df.to_csv(_result, mode='w', encoding='UTF_8', index=False)

    result = _result.getvalue().encode('utf-8')

    return result


def process(server_url: str, input_data: StringIO, categorical_columns: List[str], ordinal_columns: List[str],
            rows: int) -> str:
    m = MultipartEncoder(
        fields={'rows': str(rows), 'ordinal_columns': ','.join(ordinal_columns),
                'categorical_columns': ','.join(categorical_columns),
                'file': ('filename', input_data, 'text/csv')}
    )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r.content.decode('utf-8'), r.elapsed.total_seconds()


def get_data_download_link(data):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = data.to_csv(index=False, header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'##### <a href="data:file/csv;base64,{b64}">Download .csv</a>'

    return href


url = getenv('API_URL', 'http://localhost:8000')
endpoint = '/synthesize'

# UI -------------------------------------------------------------------------------------------------------------------

st.beta_set_page_config(page_title='Synthsonic', page_icon=':hedgehog:', layout='wide')
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Synthsonic - synthetic data service')
st.write('''Obtain synthetic data based on real data.''')  # description and instructions

input_data = st.sidebar.file_uploader('Insert CSV File (with header)')  # image upload widget

if input_data is None:
    st.info(f"Insert not empty CSV with at least {INPUT_DATA_MIN_ROWS:,} rows.")
else:
    real_df = _bytes_to_df(input_data.getvalue().encode('utf-8'))
    input_df_rows_count = real_df.shape[0]
    if input_df_rows_count < INPUT_DATA_MIN_ROWS:
        st.error(
            f'Found only {input_df_rows_count} rows. At least {INPUT_DATA_MIN_ROWS} rows in input file is required.')

    else:
        st.markdown('### Raw data')
        st.markdown(f'###### **{input_df_rows_count:,}** rows found')
        st.markdown('---')
        st.write(real_df.head(DISPLAY_ROWS))

        return_rows = st.sidebar.slider('Rows to generate',
                                        min_value=0,
                                        max_value=int(real_df.shape[0]) * 2,
                                        value=int(real_df.shape[0]))

        categorical = st.sidebar.multiselect('Select categorical columns', real_df.columns.tolist())
        ordinal = st.sidebar.multiselect('Select ordinal columns', real_df.columns.tolist())

        categorical = set(categorical)
        ordinal = set(ordinal)

        if st.sidebar.button('Get synthetic data') and input_data:
            with st.spinner('Generating synthetic data :alembic:'):
                synthetic_data_bytes, time_taken = process(url + endpoint, input_data, categorical, ordinal,
                                                           return_rows)
                data = StringIO(synthetic_data_bytes)

                df = pd.read_csv(data)

            st.markdown('### Synthetic data')
            st.markdown(f'###### **{return_rows:,}** rows generated in **{round(time_taken, 2):,}** seconds')
            st.markdown('---')

            st.write(df.head(DISPLAY_ROWS))
            st.markdown(get_data_download_link(df), unsafe_allow_html=True)

            st.markdown('### Comparison')
            st.markdown('---')
