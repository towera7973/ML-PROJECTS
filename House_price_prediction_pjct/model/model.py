import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


Dt1=pd.read_csv("House_price_prediction_pjct/model/bengaluru_house_prices.csv")
Data_1=pd.DataFrame(Dt1)
print(Data_1.head())
print(Data_1.columns)
print(Data_1.shape)
print(Data_1.area_type.unique())

#data cleaning,dropping columns not wanted 
Data_1.drop(['area_type', 'availability', 'society', 'balcony'] , axis='columns', inplace=True)
print(Data_1.shape)
print(Data_1.head())
#data cleaning 
print(Data_1.isnull().sum())
#drop all the rows with null values 
Data_2=Data_1.dropna()
print(Data_2.isnull().sum())

#now we have to intergerise the size column to number and BHK ,BEDROON HOUSE KITCHEN, BATHROOM
Data_3=Data_2.copy()
Data_3['Bedrooms']= Data_3['size'].apply( lambda x:  int(x.split(' ')[0]))
print(Data_3.head())
#we can now drop size column
Data_4=Data_3.drop(['size'] , axis='columns')
print(Data_4.head()) 
#check uniquie= values in todta sqft column
print(Data_4.total_sqft.unique())
#from the output of the above line we can see that some values are not in the form of number, we can convert them to number or float 
#before converting let first filter out which numbers are not float 
def float_check(y):
    try:
        float(y)
    except:
        return False
    return True
#this is an independent function to check if a value can be converted to float or not
#application of the above function to the total_sqft column
print("--------------------------------------------- ---------------------------------")
print('Printing the rows where total_sqft is not a float value')
print("--------------------------------------------- ---------------------------------")
print(Data_4[~Data_4['total_sqft'].apply(float_check)].head(10))
#now we need to convert the total_sqft column to float by finding average of the range , we can do this by applying a function to the column
def convert_range_to_float(y):
    token=y.split('_')
    if len(token)== 2:
        return (float(token[0]) + float(token[1]))/2
    try:
        return float(y)
    except:
        return None  
Data_4['total_sqft']= Data_4['total_sqft'].apply(convert_range_to_float)
Data_4_cleaned = Data_4.dropna(subset=['total_sqft'])
Data_4_cleaned = Data_4_cleaned.reset_index(drop=True)
print('---------------------------------------------------------------------------------')
print('Printing the rows where total_sqft is not a float value after conversion')
print(Data_4_cleaned.head())
print(Data_4_cleaned.loc[30])

#Feature Engineering 
print('add new feature price per square feet') 
Data_5= Data_4_cleaned.copy()
Data_5['price_per_sqft'] = Data_5['price']*100000/ Data_5['total_sqft']
print(Data_5.head())

#noew lets sort out location ... Lets see what in Location and print statistics of location
print('---------------------------------------------------------------------------------')
print(Data_5.location.unique())
#removing leading and trailing whitespace characters from a string.
Data_5.location = Data_5.location.apply(lambda x: x.strip())
#statistics of how many houses are there in each location
print(Data_5['location'].value_counts())
#Demention reduction by removing all locations with less than 10 houses in them 
print('---------------------------------------------------------------------------------')
print('Printing the locations with less than 10 houses')
Location_stats= Data_5["location"].value_counts(ascending=False)
Location_small= Location_stats[Location_stats<=10]
print('these are the locations ', Location_small)
#now we want to put all these in locatiopn called others coz they are minors 
Data_5.location =Data_5['location'].apply(lambda x : 'Others' if x in Location_small else x)
print('---------------------------------------------------------------------------------')
print(sorted(Data_5.location.unique().tolist()))
print("from this list we can see that minor locations like High Ground , Whitefield, etc. are now replaced with Others")
# Outliyer removal ..all houses less than 400 sqft per bedroom should be removed bellow
print(Data_5.shape)
Data_6 = Data_5[~(Data_5.total_sqft / Data_5.Bedrooms<400)]
print('---------------------------------------------------------------------------------')
print(Data_6)
print(Data_6.shape)

#dropping prices that are outside normal range above or bellow mean in a group of each location 
def outlier_price_removal(Data_n):
    Empty_df =pd.DataFrame()
    for key , subgroup in Data_n.groupby("location"):
       meanx=np.mean(subgroup.price_per_sqft)
       stdx=np.std(subgroup.price_per_sqft)
       right_price_group=subgroup[(subgroup.price_per_sqft >= (meanx-stdx)) & (subgroup.price_per_sqft <= (meanx+stdx))]
       Empty_df= pd.concat([Empty_df ,right_price_group], ignore_index=True)
    return Empty_df
Data_7= outlier_price_removal(Data_6)
print('--------------------------------New Data-------------------------------------------------')
print(Data_7.shape)

#Now we need to remove outliers of locations that has unrealistic prices per square feet based on number of room
#house with 3 bedrooms should not have price per square feet more than the one with 2 or 1 bedroom
#we need to put these in an outlier box

def remove_bedroom_outlier(data_x):
    exclude_indices = np.array([])
    for location_name, location_data in Data_7.groupby("location"):
       loc_Bedroom_stats = {}  #Dictionary to hold statistics for  number of bedrooms in a unique location
       for bedroom_num ,bedroom_data in location_data.groupby("Bedrooms"):
        loc_Bedroom_stats[bedroom_num]={
             'mean': np.mean(bedroom_data.price_per_sqft),
             'std': np.std(bedroom_data.price_per_sqft),
             'count': bedroom_data.shape[0]
               }
        #once all the stats for a specific location has been retrived we can now compare the prices per square feet of each bedroom with the next one
        ##using the mean and std of the next bedroom    
       for bedroom_num , bedroom_data in location_data.groupby('Bedrooms'):
           #get stats for previous number of bedrooms 
           if bedroom_num > 1:
               prev_bedroom_stats =loc_Bedroom_stats.get(bedroom_num - 1, None)
               #now compare the current bedroom stats with the previous one and put the outllier  in an array
               if prev_bedroom_stats and prev_bedroom_stats['count'] > 5:
                   #if the current bedroom price per square feet is more than the previous one then we put it in an outlier bo
                exclude_indices = np.append(exclude_indices, bedroom_data[bedroom_data.price_per_sqft < prev_bedroom_stats['mean']].index.values)
        #now we can drop the index number in the array because we dont want the index number to be in the final dataframe
    return Data_7.drop(exclude_indices, axis='index').reset_index(drop=True)
Data_8= remove_bedroom_outlier(Data_7)
print('--------------------------------New Data after removing bedroom outliers---------------------------------')
print(Data_8.shape)
#if number of bathrooms is greater than bedrooms plus 1 then that is an outlier
Data_9= Data_8[Data_8.bath < Data_8.Bedrooms + 1]
print('--------------------------------New Data after removing bathroom outliers---------------------------------')
print(Data_9.shape)
#now we need to drop out data on columns that we dont need anymore
Data_10 =Data_9.drop(['price_per_sqft', 'price'],  axis='columns')
print('--------------------------------New Data after dropping price_per_sqft and price columns---------------------------------')
print(Data_10)

#Now all data cleaning and preparation is done.For training we need to convert all data to numerical values
#therefore we need to convert the location column to numerical values using  One Hot Encoding For Location

Dummies = pd.get_dummies(Data_10.location)
print(Dummies.head(10))
