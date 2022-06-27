# Prepare .py 


# These are the Imports that are used 

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from plotly.subplots import make_subplots
import pandas as pd
import requests
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# set figure size
plt.rcParams["figure.figsize"] = (8,6)

# specify decimal places to display
pd.set_option('display.float_format', lambda x: '%.2f' % x)


sns.set()

# set figure size
plt.rcParams["figure.figsize"] = (8, 6)

# specify decimal places to display
pd.set_option("display.float_format", lambda x: "%.2f" % x)

import acquire
from acquire import get_stores,get_germany_data


# -------------------------------------------------------------------------------------------



def clean_store(df):
    """
    clean_store will take one argument df, a pandas dataframe and will:
    do initial clean up by dropping unnecessary columns and re-arranging the leftover columns,  # This is a discription on what this does !!!! 
    convert date column to datetime format, ,
    set the index to be the datetime variable,
    add a 'month' and 'day of week' column,
    add sales_total column, which is a derived from sale_amount (total items) and item_price,
    plot the distribution of sale_amount and item_price, and
    
    return: a single pandas dataframe with the above operations performed
    """

    # drop unnecessary columns
    df = df.drop(
        columns=[  "store", "item"]
    )  # Dropping not needed col

    # reorder the columns
    df = df[
        [
            "store_id",
            "item_id",
            "sale_id",
            "sale_date",
            "sale_amount",
            "item_upc14",  # Reorg
            "item_upc12",
            "item_brand",
            "item_name",
            "item_price",
            "store_address",
            "store_zipcode",
            "store_city",
            "store_state",
        ]
    ]

    # convert the 'sale_date' col to datetime format
    df.sale_date = pd.to_datetime(
        df.sale_date, format="%a, %d %b %Y %H:%M:%S %Z"
    )  # Formating

    # set datetime variable as index
    df = df.set_index("sale_date").sort_index()

    # add a 'month' and 'day of week' column
    df["month"] = df.index.month
    df["day_of_week"] = df.index.day_name()

    # add sales_total column
    df["sales_total"] = df.sale_amount * df.item_price

    # plot the distribution of sale_amount and item_price
    df[["sale_amount", "item_price"]].hist()
    plt.show()

    return df


def clean_germany(df):
    '''
    clean_germany will take one argument df, a pandas dataframe and will:
    convert date column to datetime format, ,
    set the index to be the datetime variable,
    add a 'month' and 'year' column,
    fill in missing values with '0',
    plot the distribution of each of the variables, and
    
    return: a single pandas dataframe with the above operations performed
    '''
    
    #convert the 'sale_date' col to datetime format
    df.Date = pd.to_datetime(df.Date)
    
    #set datetime variable as index
    df = df.set_index('Date')
    
    #add a 'month' and 'year' column
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    #fill in missing values
    df = df.fillna(0)
    
    #plot the distribution of each of the variables
    df.hist()
    plt.show()
    
    return df