import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from env import get_db_url
from sklearn.model_selection import train_test_split

import sklearn.preprocessing

# Joseph Goerner
#Connection function to access Codeup Database and retrieve zillow dataset from mysql


#------------------------------------------------------------------------------------ACQUIRE--------------------------------------------------

#This function reads in the zillow data from the Codeup 
    #Database connection made from get_connection
    #and returns a pandas DataFrame with all columns in tow, needed to start this proccess


def acquire_stores():
    '''
THIS FUNCTION TAKES IN THREE URLS AND A BASE URL TO BE USED IN GETTING MULTIPLE PAGES OF 
INFORMATION, AND CREATES A DF FOR EACH, THEN WRITING EACH DF TO CSV FILES.    
    '''

    base_url = 'https://python.zgulde.net'

    #url1 | ITEMS DATA 
    url1 = 'https://python.zgulde.net/api/v1/items'
    response = requests.get(url1)
    data = response.json()
    df = pd.DataFrame(data['payload']['items'])

    while data['payload']['next_page'] != None:
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        df = pd.concat([df, pd.DataFrame(data['payload']['items'])])#.reset_index(drop = True)

    df_items = df.copy
    print('Items data acquired')

    #url2 | STORES DATA
    url2 = 'https://python.zgulde.net/api/v1/stores'
    response = requests.get(url2)
    data = response.json()
    df = pd.DataFrame(data['payload']['stores'])

    df_stores = df.copy()
    print('Stores data acquired')

    #url3 | SALES DATA
    #url3 = 'https://python.zgulde.net/api/v1/sales'
    #response = requests.get(url3)
    #data = response.json()
    #df = pd.DataFrame(data['payload']['sales'])

    while data['payload']['next_page'] != None:
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        df = pd.concat([df, pd.DataFrame(data['payload']['sales'])])#.reset_index(drop = True)

    df_sales = df.copy()
    print(' Sales data acquired')

    #saving dfs to csv files
    df_items.to_csv('items.csv')
    df_stores.to_csv('stores.csv')
    df_sales.to_csv('sales.csv')

    return df_items, df_stores, #df_sales





def get_stores():
    '''
THIS FUNCTION CHECKS TO SEE IF CSV FILES EXISTS FOR THE STORES DATA AND, IF SO, 
WRITES THE CSV FILES TO DFS. IF NOT THE FUNCTION WILL REUN THE PREVIOUS acquire_stores()
FUNCTION.
    '''

    #checking to see if csv files exist
    if (os.path.isfile('items.csv') == False) or (os.path.isfile('stores.csv') == False) or (os.path.isfile('sales.csv') == False):
        print('Data is not cached. Acquiring new data . . .')
        df_items, df_stores, df_sales = acquire_stores()
        #if no local csv exists, running above function to write to df and cache

    else:
        print('Data is cached. Reading from csv files.')
        
        df_items = pd.read_csv('items.csv')
        print('Items data acquired')

        df_stores = pd.read_csv('stores.csv')
        print('Items data acquired')

        df_sales = pd.read_csv('sales.csv')
        print('Sales data acquired')

    df_combined = pd.merge(df_items,
                            df_sales,
                            how = 'right',
                            right_on = 'item',
                            left_on = 'item_id')
    df_combined = pd.merge(df_combined,
                            df_stores,
                            how = 'left',
                            left_on = 'store',
                            right_on = 'store_id')
    print(f'{"*" * len("Acquisition Complete")}\nAcquisition Complete')

    #removing index cols
    #df_combined.drop(columns = ['Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0'], inplace = True)

    #caching combined df
    df_combined.to_csv('combined.csv')

    return df_combined


def acquire_zillow():
    '''
    This function reads in the zillow data from the Codeup 
    Database connection made from get_connection
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
        SELECT prop.*,
                   pred.logerror,
                   pred.transactiondate,
                   air.airconditioningdesc,
                   arch.architecturalstyledesc,
                   build.buildingclassdesc,
                   heat.heatingorsystemdesc,
                   landuse.propertylandusedesc,
                   story.storydesc,
                   construct.typeconstructiondesc
            FROM   properties_2017 prop
            INNER JOIN (SELECT parcelid,
                               logerror,
                               Max(transactiondate) transactiondate
                        FROM   predictions_2017
                        GROUP  BY parcelid, logerror) pred
                     USING (parcelid)
            LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
            LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
            LEFT JOIN storytype story USING (storytypeid)
            LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
            WHERE prop.latitude IS NOT NULL
                  AND prop.longitude IS NOT NULL
                  AND transactiondate like '2017%%'
                '''

    return pd.read_sql(sql_query, get_db_url('zillow'))    


   

# This function reads in zillow data from Codeup database and 
    #writes data to a csv file if cached == False. If cached == True 
    #reads in zillow df from a csv file, returns df. 


def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and 
    writes data to a csv file if cached == False. If cached == True 
    reads in zillow df from a csv file, returns df. 
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # reads the infomation from the data frame and database
        df = acquire_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # I have  found that the notebook once the cvs is imported will keep reverting to same file 
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df

    
def count_values(df):
    '''
    This function reads in a dataframe and outputs the column name along 
    with the value counts for each column
    '''
    for column in df.columns:
        if column != 'parcelid'and column != 'id':
            print('Column:', column)
            print(df[column].value_counts())
            print('\n')

def null_city(df):
    '''
    This function takes in a dataframe and outputs all the 
    nulls for each column
    '''
    for column in df.columns:
        print('Column:', column)
        print('Null count:', df[column].isnull().sum())
        print('\n')        

def zillow_dist():
    '''
    This function takes in a dataframe and outputs histograms
    of bedrooms, finished area, logerror, and taxvaluedollarcount
    '''
    
    plt.figure(figsize = (12,8))
    plt.subplot(221)
    plt.hist(df.bedroomcnt)
    plt.title('Bedrooms')



    plt.subplot(222)
    plt.hist(df.calculatedfinishedsquarefeet)
    plt.title('finished area')



    plt.subplot(223)
    plt.hist(df.logerror)
    plt.title('logerror')



    plt.subplot(224)
    plt.hist(df.taxvaluedollarcnt)
    plt.title('taxvaluedollarcnt')

    plt.tight_layout()

def nulls_by_col(df):
    '''
    function that takes in a dataframe of observations and attributes 
    and returns a dataframe where each row is an atttribute name, the 
    first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows': num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing


def cols_missing(df):
    '''
    A function that takes in a dataframe and returns a dataframe with 3 columns: 
    the number of columns missing, 
    percent of columns missing, 
    and number of rows with n columns missing
    '''
    df2 = pd.DataFrame(df.isnull().sum(axis =1), columns = ['num_cols_missing']).reset_index()\
    .groupby('num_cols_missing').count().reset_index().\
    rename(columns = {'index': 'num_rows' })
    df2['pct_cols_missing'] = df2.num_cols_missing/df.shape[1]
    return df2

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    '''
    A function that will drop rows or columns based on the percent of values that are missing: 
    handle_missing_values(df, prop_required_column, prop_required_row)
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    '''
    A function that will drop columns you want removed from dataframe
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    '''
    A function that will handle erroneous data, handle missing values,
    remove columns, add columns, replace nulls, fill nulls, and drop nulls
    for zillow dataset
    '''
    df = pd.read_csv('zillow.csv')
    
    # Restrict df to only properties that meet single unit use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc'])


    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df


# This is our scaler that we will use on the project ---------------------------------------------------------- MIN_MAX_SCALER 

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

# ---------------------------------------- Missing Value Tables ---------------------------------------------------- #

def missing_zero_values_table(df):
    '''This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values
        and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table
