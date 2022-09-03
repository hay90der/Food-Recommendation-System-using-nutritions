import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from IPython.display import Markdown, display
import seaborn as sns
sns.set()
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import cdist
import os
from PIL import Image

#Pillow==9.1.0
background = Image.open('logo2.png')
st.image(background,width=300)
"""
# Food Recommendation System using food nutrition
"""

def get_data():
    path = "clustering_and_anomalies_hayder.csv"
    return pd.read_csv(path)

data = get_data()
#data
# remove duplicatite name , data = data.drop_duplicates(subset=['product_name'], keep='last', inplace=False)
# rename duplicatcs name , df1.index = df1.index + df1.groupby(level=0).cumcount().astype(str).replace('0','')


data['product_name'] = data['product_name'] + data.groupby(['product_name']).cumcount().astype(str).replace('0','')

pn = data['product_name'].drop_duplicates()
pn_choice = st.sidebar.selectbox("Select your food:", pn)
fat = data["fat_100g"].loc[data["product_name"] == pn_choice]
carbs = data["carbohydrates_100g"].loc[data["product_name"] == pn_choice]
sugars = data["sugars_100g"].loc[data["product_name"] == pn_choice]
proteins = data["proteins_100g"].loc[data["product_name"] == pn_choice]
salts = data["salt_100g"].loc[data["product_name"] == pn_choice]
energy = data["energy_100g"].loc[data["product_name"] == pn_choice]

#fat_choice = st.sidebar.write(f"Value of d", fat)
#carbs_choice = st.sidebar.text("amount of carbs" + carbs)
#sugars_choice = st.sidebar.multiselect("amount of sugars", sugars)
#proteins_choice = st.sidebar.multiselect("amount of proteins", proteins)
#salts_choice = st.sidebar.multiselect("amount of salts", salts)
#energy_choice = st.sidebar.multiselect("amount of energy", energy)
#st.sidebar.text_area("amount of fat & carbs:", fat_choice + fat_choice +fat_choice )


# initialize data of lists.
newdata = {
        'name  ':['fat','carbs','sugars','proteins','salts','energy'],
        'amount':[np.float64(fat),np.float64(carbs),np.float64(sugars),np.float64(proteins),np.float64(salts),np.float64(energy)]
}

dff = pd.DataFrame(newdata)
st.sidebar.write('Food Nutrition Facts')
st.sidebar.table(dff)

defn = 5
myn = st.text_input('Choose your cluster size, input numbers only',defn)
st.markdown('---')

n = int(myn)
"""
##### According to your food nutrition we recommend these foods
* using Euclidean Distance
"""

########################################
# start from here i add my project code#
########################################



original = pd.read_csv("clustering_and_anomalies_hayder.csv")
results = pd.read_csv("clustering_and_anomalies_hayder.csv")


cal = data.pivot_table(columns='product_name',
                          values=['fat_100g','carbohydrates_100g','sugars_100g',
                                  'proteins_100g','salt_100g','energy_100g',
                                  'reconstructed_energy','g_sum','cluster','certainty','anomaly'
                                 ]
                         )



def your_food_cal(recipe_name, corr = 0.999, recipe_number = n):
    # create a dataframe
    dataframe = pd.DataFrame(data, columns = ['product_name','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','energy_100g','reconstructed_energy','g_sum','cluster','certainty','anomaly'])
    #the full dataset
    display(dataframe.head(2))
    # selecting rows based on condition
    rslt_df = dataframe[dataframe['product_name'] == recipe_name]
    #user choise
    display(rslt_df)
    #similarity between 2 dataframes
    #cos_sim = dot(dataframe, rslt_df) / (norm(dataframe) * norm(rslt_df))
    #print(cos_sim)
    data_x=dataframe.drop(columns=['product_name', 'anomaly'])
    data_y=rslt_df.drop(columns=['product_name', 'anomaly'])
    cosine_sim = cosine_similarity(data_x,data_y)
    cosine_sim_df = pd.DataFrame(cosine_sim,columns=['cosine_similarity'])
    cosine_sim_df = cosine_sim_df.sort_values('cosine_similarity', ascending=False)
    idx = cosine_sim_df.index
    display(cosine_sim_df.head(n))
    ary = cdist(data_x, data_y, metric='euclidean')
    listt = pd.DataFrame(ary,columns=['euclidean_dist'])
    listt = listt.sort_values(by='euclidean_dist', ascending=True)
    #print(listt.head(6))
    #return listt.head(6)

newdata = your_food_cal('Banana Chips Sweetened (Whole)')
#######################################
### start using eculidean algorithm ###
#######################################

dataframe1 = pd.DataFrame(data, columns = ['product_name','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','energy_100g','reconstructed_energy','g_sum','cluster','certainty','anomaly'])
rslt_df = dataframe1[dataframe1['product_name'] == pn_choice]


def Euclidean_Dist(dataframe1, rslt_df, cols=['fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','energy_100g']):
    data_x=dataframe1 #.drop(columns=['product_name','anomaly','reconstructed_energy','g_sum','certainty','cluster'])
    data_y=rslt_df #.drop(columns=['product_name','anomaly','reconstructed_energy','g_sum','certainty','cluster'])
    
    data_y["product_name"] =   original.loc[dataframe1.index.values, "product_name"]
    data_y["cluster"] =   original.loc[dataframe1.index.values, "cluster"]
    
    
    results = np.linalg.norm(data_x[cols].values - data_y[cols].values,axis=1)
    
    eculideanresultsarr = []
    eculideanresultsarr = results
    
    rdf = pd.DataFrame(results, columns = ['EculideanDist'])
    rdf["product_name"] =   original.loc[dataframe1.index.values, "product_name"]
    rdf["fat_100g"] =   original.loc[dataframe1.index.values, "fat_100g"]
    rdf["carbohydrates_100g"] =   original.loc[dataframe1.index.values, "carbohydrates_100g"]
    rdf["sugars_100g"] =   original.loc[dataframe1.index.values, "sugars_100g"]
    rdf["proteins_100g"] =   original.loc[dataframe1.index.values, "proteins_100g"]
    rdf["g_sum"] =   original.loc[dataframe1.index.values, "g_sum"]
    rdf["cluster"] =   original.loc[dataframe1.index.values, "cluster"]
    
    rdf = rdf.sort_values(by='EculideanDist', ascending=True)
    
    #skip first row - drop first row
    rdf = rdf.iloc[1: , :]
    display(rdf.head(n))
    
    print(rdf.head(n))
    return rdf.head(n)
    #return np.linalg.norm(dataframe1[cols].values - rslt_df[cols].values,axis=1)


EucDist = Euclidean_Dist(dataframe1, rslt_df)
st.write(EucDist)
#####################################
### end using eculidean algorithm ###
#####################################




#######################################
### start using manhattan algorithm ###
#######################################


dataframe1 = pd.DataFrame(results, columns = ['product_name','fat_100g','carbohydrates_100g','sugars_100g','proteins_100g','salt_100g','energy_100g','reconstructed_energy','g_sum','cluster','certainty','anomaly'])
rslt_df = dataframe1[dataframe1['product_name'] == pn_choice]
data_x=dataframe1.drop(columns=['product_name','anomaly','reconstructed_energy','g_sum','certainty','cluster'])
data_y=rslt_df.drop(columns=['product_name','anomaly','reconstructed_energy','g_sum','certainty','cluster'])

st.dataframe(data_x)
st.markdown('---')
st.dataframe(data_y)
mhtn = manhattan_distances(data_x, data_y, sum_over_features=True)
mhtndf = pd.DataFrame(mhtn, columns = ['ManhattanDist'])

mhtndf["product_name"] =   original.loc[dataframe1.index.values, "product_name"]
mhtndf["fat_100g"] =   original.loc[dataframe1.index.values, "fat_100g"]
mhtndf["carbohydrates_100g"] =   original.loc[dataframe1.index.values, "carbohydrates_100g"]
mhtndf["sugars_100g"] =   original.loc[dataframe1.index.values, "sugars_100g"]
mhtndf["proteins_100g"] =   original.loc[dataframe1.index.values, "proteins_100g"]
mhtndf["g_sum"] =   original.loc[dataframe1.index.values, "g_sum"]
mhtndf["cluster"] =   original.loc[dataframe1.index.values, "cluster"]


mhtndf = mhtndf.sort_values(by='ManhattanDist', ascending=True)
#skip first row - drop first row
mhtndf = mhtndf.iloc[1: , :]
mn = mhtndf.head(n)


#####################################
### End using manhattan algorithm ###
#####################################

###################################
#end to here i add my project code#
###################################

"""
##### According to your food nutrition we recommend these foods
* using Manhattan Distance
"""
st.write(mn)
st.markdown('---')
image = Image.open('hm.jpg')

st.image(image)
"""
##### Developed By: Hayder Hussien Al-Chalabi
##### Supervised By: Assist. Prof. Dr. Mahdi N. Jassim
##### Informatics Inistitute for Postgraduate Studies - IIPS
"""

GitHub = '[GitHub](https://github.com/hay90der/)'
facebook = '[Facebook](https://facebook.com/jojoslover)'
twitter = '[Twitter](https://twitter.com/JOJOSLoVeR)'

st.markdown(''+ GitHub + ' - ' + facebook + ' - ' + twitter + '', unsafe_allow_html=True)




