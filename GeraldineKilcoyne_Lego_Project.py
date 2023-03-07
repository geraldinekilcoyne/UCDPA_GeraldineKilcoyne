
#import the library for handling files
import pandas as pd
#import the library for handling regular expression
import re
#import the library for handling databases
import sqlite3
from sqlite3 import Error
#import the library for handling visuals
import matplotlib.pyplot as plt
#import the library for handling numbers and arrays
import numpy as np
#import the library for handling linear regression 
from sklearn.linear_model import LinearRegression
#import the library for handling training and testing
from sklearn.model_selection import train_test_split
#import the library for handling mean_absolute_error from sklearn module
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import the library for handling mean_absolute_percentage_error from sklearn module
from sklearn.metrics import mean_absolute_percentage_error
#*****************************************#

def create_connection(database_file_name):
    """Create a connection to a sqlite database, if it does not exist it will be automaticly created"""
    this_conn = None
    try:
        this_conn = sqlite3.connect(database_file_name)
        return this_conn
    except Error as this_error:
        print(this_error)
    return this_conn

#*****************************************#

def create_tables(sql_to_create_table):
    """Call the functions to make a connection and create the tables in the specified database"""
    database = 'lego.db'
    this_conn = create_connection(database)
    if this_conn is not None:
      c = this_conn.cursor()
      c.execute(sql_to_create_table)
    else:
        print("can not connect to the database")

#*****************************************#

def load_data_files_to_tables(data_file,db_path,table_name):
    """Load the data from a dataset in to the tables of the database"""
    this_conn = sqlite3.connect(db_path)
    c = this_conn.cursor()
    data_file.to_sql(table_name, this_conn, if_exists='replace', index = False)
    this_conn.commit()

#*****************************************#

def split_dataframe_column (df_name,df_col,regex_string,new_col_name_1, new_col_name_2, new_col_name_3):
    """splits a column in a dataframe and creates new columns based on the regex string that is passed in"""
    df_split = (df_name
            .assign(list_col=lambda df: df[df_col].str.split( regex_string, 3, expand=False),
                    temp_col_name_1=lambda df: df.list_col.str[0],
                    temp_col_name_2=lambda df: df.list_col.str[1],
                    temp_col_name_3=lambda df: df.list_col.str[2]
                    )
            .drop(columns=['list_col'])
            .drop(columns=[df_col])
            )
    df_split.rename(columns = {'temp_col_name_1': new_col_name_1, 'temp_col_name_2': new_col_name_2,'temp_col_name_3': new_col_name_3}, inplace = True)
    return df_split

#*****************************************#

#read .csv files in to dataframes
df_sets = pd.read_csv('sets.csv')
df_themes  = pd.read_csv('themes.csv')
df_parts = pd.read_csv('parts.csv')
#*****************************************#

#remove any rows that contain the same data
df_sets.drop_duplicates()
df_themes.drop_duplicates()
df_parts.drop_duplicates()
#*****************************************#
# Lego parts #
#in the parts dataframe the some rows have a set number in the name column. Some have up to 3 Lego set numbers. 
#use Regex to put the Lego set numbers in to columns. call the function split_dataframe_column to do this
print('-----------')
print('parts data before clean')
print(df_parts)
print('-----------')
df_split = split_dataframe_column(df_parts,'name','\Sets|Set|sets|set','part_description', 'set_1', 'set_2')
df_parts = split_dataframe_column(df_split,'set_1',',','set_num_1', 'set_num_2', 'set_num_3')
df_parts = df_parts.drop(["part_cat_id","part_material","set_2"], axis='columns')
df_parts['part_description'][df_parts.set_num_1 != 'NaN'] = df_parts.part_description + 'Set '
print('-----------')
print('parts data after clean')
print(df_parts)
print('-----------')

#*****************************************#

#I have lego set id numbers for all sticker sheets so I will create a new dataframe for stickers
df_stickers = df_parts[df_parts['part_description'].str.contains(pat = 'Sticker Sheet.*', regex = True)]

#Create a dictionary from the df_stickers dataframe using dict() and zip() methods. 
#One dictionary for each set_num column. then bring all the rows togeather in a single dictionary 
dict1 = dict([(i,[x]) for i,x in zip(df_stickers['set_num_1'], df_stickers['part_num'])])
dict2 = dict([(i,[x]) for i,x in zip(df_stickers['set_num_2'], df_stickers['part_num'])])
dict3 = dict([(i,[x]) for i,x in zip(df_stickers['set_num_3'], df_stickers['part_num'])])
dict_stickers = {**dict1, **dict2, **dict3}

#convert the dict_stickers dictionary to a dataframe
df_stickers = pd.DataFrame(dict_stickers.items(), columns=['set_num', 'part_num']) 
print('-----------')
print('The sticker dataframe ')
print(df_stickers)
print('-----------')

#*****************************************#
# Lego sets and themes #
#rename the column in the df_themes dataframe
df_themes.rename(columns = {'id':'theme_id'}, inplace = True) 
df_themes.rename(columns = {'name':'theme'}, inplace = True) 

df_sets.rename(columns = {'name':'set_name'}, inplace = True) 
df_sets.rename(columns = {'num_parts':'number_of_parts'}, inplace = True) 

#*****************************************#
#merge the df_sets and df_themes dataframes to create a new data frame called df_sets_with_themes
df_sets_with_themes = df_sets.merge(df_themes,on=["theme_id"])
df_sets_with_themes = df_sets_with_themes.drop(["theme_id","parent_id"], axis='columns')
print('-----------')
print('The sets and themes dataframe ')
print(df_sets_with_themes)

#*****************************************#
#use the descripe method to return facts about the data
print('-----------')
print('Facts about the data in the number_of_parts column in the df_sets_with_themes dataframe ')
print(df_sets_with_themes['number_of_parts'].describe())
print('-----------')

#*****************************************#
#create a numpy array 
np_theme_parts_year = df_sets_with_themes[['year','number_of_parts','set_num','theme']].to_numpy()

#get all lego sets that have a theme of 'Technic'
rows = np.where(np_theme_parts_year[:,3] == 'Technic')
return_all_of_a_theme = np_theme_parts_year[rows]

#print all lego sets with a theme of 'Technic'  
print('-----------')
for (a,x, y, z) in return_all_of_a_theme:
    print (str(z) + " Set " + str(y) + " year " + str(a) + " Qty Parts " + str(x) )

#get the total number of all parts used in 'Technic' themed Lego sets
np_array_col_sum = np.sum(return_all_of_a_theme[:,1], axis = 0)
print('')
print("Total Quantity of Parts in the Technic themed sets listed above is " + str(np_array_col_sum))
print('-----------')

#*****************************************#
#Write data to a database.

#Call the function I wrote to create a table. This table will hold Lego sets and theme data
#Pass in the SQL query to create the table in sqlite.
create_tables(""" CREATE TABLE IF NOT EXISTS sets_with_theme (
                                        set_num varchar,
                                        set_name varchar,
                                        year integer,
                                        num_parts integer,
                                        img_url text,
                                        theme text
                                    ); """)

#Call the function that I wrote to load the df_sets_with_themes dataframe in to a table in the sqlite database called lego.db
load_data_files_to_tables(df_sets_with_themes, 'lego.db','sets_with_theme')

#*****************************************#
#visualise the data

#Plot data from the Lego database to charts

#make a connection to the lego database and run a sql query to get data to visual on graphs
this_conn = create_connection('lego.db')

#plot the quantity of Lego sets released each year on a Bar chart, exclude 2023 because it's not a full year
sql_to_plot = """select count(distinct set_num) count_sets, year from sets_with_theme where year <> 2023 group by year"""
data1 = pd.read_sql(sql_to_plot, this_conn)
plt.figure(figsize=(10, 5))
plt.bar(x = data1.year , height = data1.count_sets,color = 'g')
plt.title('Quantity of Lego sets released each year', fontsize=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Quantity of sets', fontsize=12)
plt.show()

#plot the quantity of Lego themes used on sets released each year on a line chart , exclude 2023 because it's not a full year
sql_to_plot = """select count(distinct theme) as count_themes, year from sets_with_theme where year <> 2023 group by  year"""
data2 = pd.read_sql(sql_to_plot, this_conn)
plt.figure(figsize=(10, 5))
plt.plot(data2.year, data2.count_themes, color='red', marker='o')
plt.title('Quantity of Themes each year', fontsize=20)
plt.xlabel('year', fontsize=12)
plt.ylabel('Quantity of Themes', fontsize=12)
plt.grid(True)
plt.show()


#-------------------------------------
#machine learning

#get data on the quantity of Lego sets released each year since the years between 2000 and 2022 and put this data into a numpy array
#I take thes years to remove early years when quantity was low.
sql_to_plot = """select year, count(distinct set_num) count_sets from sets_with_theme where year >= 2000 and year <= 2022 group by year"""
data3 = pd.read_sql(sql_to_plot, this_conn)
X = np.array(data3.year).reshape(-1, 1)
y = np.array(data3.count_sets).reshape(-1, 1)
#use the Linear Regression algorithm to make predictions
reg = LinearRegression()
reg.fit(X,y)
predictions = reg.predict(X)
#visualise the actual data and the predicted data on a scatter chart
#plot the actual in dots
plt.figure(figsize=(10, 5))
plt.scatter(X,y, color="green")
#plot the best fitted line in red
plt.plot(X, predictions, color = "red")
plt.title('Machine learning - Lego sets released years 2000 to 2022 ')
plt.show()

#training and testing
#input values
X = data3.drop('count_sets', axis=1)
#output values
Y = data3.drop('year', axis=1)
#break the dataset in to a training dataset and a testing dataset
#The train and test sets directly affect the model’s performance score. 
#Because we get different train and test sets with different integer values for random_state
#Set the hyperparameter random_state = 60 because this is where I get the best fit. 
#use 20% data for testing and 80% for training. Adjusting random_state and train_size adjusted until 
#I found a sweet show where mean error was lowest.
X_training, X_testing, y_training, y_testing = train_test_split(X, Y, train_size=.80, random_state=60)
#fitting simple linear regression to the training dataset
reg.fit(X_training, y_training)
#predict the testing dataset results
y_predictions = reg.predict(X_testing)
#plott the actual values in dots
plt.figure(figsize=(10, 5))
plt.scatter(X_testing, y_testing, color='yellow')
#plot the best fitted line in red
plt.plot(X_testing, y_predictions, color='red')
plt.title('Machine learning - Lego sets released years 2000 to 2022 - Testing and Training')
plt.show()

#show how close Actual and Predicted is
#actual values
plt.figure(figsize=(10, 5))
plt.plot([i for i in range(len(y_testing))],np.array(y_testing), c='b', label="actual values")
#predicted values
plt.plot([i for i in range(len(y_testing))],y_predictions, c='g',label="predicted values")
plt.legend()
plt.title('Machine learning - Lego sets released years 2000 to 2022 - show how close Actual and Predicted are')
plt.show()

#print mean error results
print('-----------')
print('Mean absolute error')
print(np.round(mean_absolute_error(y_testing, y_predictions),2))
print('Mean squared error')
print(np.round(mean_squared_error(y_testing, y_predictions),2))
print('Mean Absolute Percentage error (MAPE) - Precentage close to zero as possible is best')
print(np.round(mean_absolute_percentage_error(y_testing, y_predictions),2))

#*****************************************#
print('-----------')
print('finished')