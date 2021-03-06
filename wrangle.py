import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from env import get_db_url

'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''



def zillow17():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with:
    - all fields related to the properties that are available
    - using all the tables in the database
    - Only include properties with a transaction in 2017
    - include only the last transaction for each property
    - zestimate error
    - date of transaction
    - Only include properties that include a latitude and longitude value
    '''
    
    query = """
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
    """
    
    return pd.read_sql(query, get_db_url('zillow'))




'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''
    
def drop_based_on_pct(df, pc, pr):
    """
    drop_based_on_pct takes in: 
    - dataframe, 
    - threshold percent of non-null values for columns(# between 0-1), 
    - threshold percent of non-null values for rows(# between 0-1)
    Returns: a dataframe with the columns and rows dropped as indicated.
    """
    tpc = 1-pc
    tpr = 1-pr
    df.dropna(axis = 1, thresh = tpc * len(df.index), inplace = True)
    df.dropna(axis = 0, thresh = tpr * len(df.columns), inplace = True)
    return df
    
    


    
def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound




    
def wrangle_zillow():
    """
    wrangle_zillow will:
    - read in df acquired from SQL query
    - filter data to single unit homes with min 1B/1B over 500 sqft
    - drop columns with 40%+ & rows 30%+ null
    - add a column for county names
    - drop unnecessary columns
    - fills in unitcnt/lotsizesquarefeet/buildingqualitytypeid nulls
    - drops remaining nulls
    - removes extreme outliers for home_value & sqft
    - adds a home_age & log_quartiles column
    - converts certain float columns to int
    - renames certain columns
    """
    
    df = pd.read_csv('zillow.csv')
    df = df.set_index("parcelid")
    
    # Restrict df to only properties that meet single-use criteria
    single_use = [260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Filter those properties without at least 1 bath & bed and 500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>500)]
    
    # Handle missing values i.e. drop columns and rows based on a threshold
    df = drop_based_on_pct(df, .6, .7)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange',
                                   'Ventura'))
    
    # Drop unnecessary/redundant columns
    df = df.drop(['id', 'calculatedbathnbr', 'finishedsquarefeet12', 
                  'fullbathcnt', 'heatingorsystemtypeid', 
                  'propertycountylandusecode', 'propertylandusetypeid', 
                  'propertyzoningdesc', 'censustractandblock', 'propertylandusedesc', 
                  'heatingorsystemdesc', 'assessmentyear', 'regionidcounty' ],axis=1)
    
    # Replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)
    
    # Replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7265, inplace = True)
    df.buildingqualitytypeid.fillna(7.0, inplace = True)
    
    # Drop any remaining nulls
    df = df.dropna()
    
    # Columns that need to be adjusted for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 12500]
    
    # create column for age of home
    df['home_age'] = 2021 - df.yearbuilt
    
    # List of cols to convert to 'int'
    cols = ['fips', 'buildingqualitytypeid', 'bedroomcnt', 
            'roomcnt', 'home_age', 'yearbuilt', 
            'regionidzip', 'unitcnt', 'home_age']
    # loop through cols list in conversion
    for col in cols:
        df[col] = df[col].astype('int')
        
    # Rename columns
    df.rename(columns={"bathroomcnt": "bathrooms", 
                   "bedroomcnt": "bedrooms",
                   "buildingqualitytypeid": "property_quality", 
                   "calculatedfinishedsquarefeet": "sqft",
                   "lotsizesquarefeet": "lot_sqft",
                   "regionidzip": "zip_code",
                   "landtaxvaluedollarcnt": "land_value",
                   "structuretaxvaluedollarcnt": "structure_value",
                   "taxvaluedollarcnt": "home_value"
                  }, inplace=True)
    

    # create a categorical version of target by splitting into quartiles
    df['logerror_quartiles'] = pd.qcut(df.logerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    
    
    return df

def clean_zillow(df):
    """
    clean_zillow will:
    - read in df acquired from SQL query
    - filter data to single unit homes with min 1B/1B over 500 sqft
    - drop columns with 40%+ & rows 30%+ null
    - add a column for county names
    - drop unnecessary columns
    - fills in unitcnt/lotsizesquarefeet/buildingqualitytypeid nulls
    - drops remaining nulls
    - removes extreme outliers for home_value & sqft
    - adds a home_age & log_quartiles column
    - converts certain float columns to int
    - renames certain columns
    """
    
    #df = pd.read_csv('zillow.csv')
    df = df.set_index("parcelid")
    
    # Restrict df to only properties that meet single-use criteria
    single_use = [260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Filter those properties without at least 1 bath & bed and 500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>500)]
    
    # Handle missing values i.e. drop columns and rows based on a threshold
    df = drop_based_on_pct(df, .6, .7)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange',
                                   'Ventura'))
    
    # Drop unnecessary/redundant columns
    df = df.drop(['id', 'calculatedbathnbr', 'finishedsquarefeet12', 
                  'fullbathcnt', 'heatingorsystemtypeid', 
                  'propertycountylandusecode', 'propertylandusetypeid', 
                  'propertyzoningdesc', 'censustractandblock', 'propertylandusedesc', 
                  'heatingorsystemdesc', 'assessmentyear', 'regionidcounty' ],axis=1)
    
    # Replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)
    
    # Replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7265, inplace = True)
    df.buildingqualitytypeid.fillna(7.0, inplace = True)
    
    # Drop any remaining nulls
    df = df.dropna()
    
    # Columns that need to be adjusted for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 12500]
    
    # create column for age of home
    df['home_age'] = 2021 - df.yearbuilt
    
    # List of cols to convert to 'int'
    cols = ['fips', 'buildingqualitytypeid', 'bedroomcnt', 
            'roomcnt', 'home_age', 'yearbuilt', 
            'regionidzip', 'unitcnt', 'home_age']
    # loop through cols list in conversion
    for col in cols:
        df[col] = df[col].astype('int')
        
    # Rename columns
    df.rename(columns={"bathroomcnt": "bathrooms", 
                   "bedroomcnt": "bedrooms",
                   "buildingqualitytypeid": "property_quality", 
                   "calculatedfinishedsquarefeet": "sqft",
                   "lotsizesquarefeet": "lot_sqft",
                   "regionidzip": "zip_code",
                   "landtaxvaluedollarcnt": "land_value",
                   "structuretaxvaluedollarcnt": "structure_value",
                   "taxvaluedollarcnt": "home_value"
                  }, inplace=True)
    

    # create a categorical version of target by splitting into quartiles
    df['logerror_quartiles'] = pd.qcut(df.logerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    
    
    return df

def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array((df.dtypes == "object") | (df.dtypes == "category"))

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_X_cols(train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in train.columns.values if col not in object_cols]
    
    return numeric_cols

def split_zillow(df):
    """
    split_zillow will take one argument(df) and 
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test & df
    """  
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210)
    return df, train, validate, test

def train_validate_test_split(df, target, seed):
    '''
    spilts our data  into train, validate, test
    '''
    # Train, Validate, and test
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed)
    # Split with X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, y_train, X_validate, y_validate, X_test, y_test 
