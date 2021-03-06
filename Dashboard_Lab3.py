import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import timeit
import sys
import time
import streamlit as st
import streamlit.components.v1 as components
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        with open('log.txt', 'w+') as f:
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds',file=f)
        return result
    return timeit_wrapper


@st.cache(suppress_st_warning=True)
def importData(wt): 
    val= pd.read_csv(wt)
    return val


def PlotDataset(rt):
    st.write("Taille du dataset :")
    st.write(rt.shape)
    st.write("Nombres de variables :")
    st.write(rt.columns)
    st.write("Premieres lignes de dataset")
    st.write(rt.head(5))


@timeit
@st.cache(suppress_st_warning=True)
def FirstTrans(df):
    df["Date/Time"]= pd.to_datetime(df["Date/Time"])
    df['dom'] = df['Date/Time'].dt.day
    df['weekday']= df['Date/Time'].dt.weekday
    df['hour']= df['Date/Time'].dt.hour
    st.write("Avec l'optimisation du dataset")
    st.write(df.head(5))
    return df




@timeit
def hist(df):
    plt.hist(df["dom"],bins = 30, rwidth=0.8, range=(0.5,30.5))
    plt.title('Frequency by DoM - Uber - April 2014, x-axis Date of the month and y-axis Frequency')

@st.cache(suppress_st_warning=True)
def count_rows(rows): 
    return len(rows)

@st.cache(suppress_st_warning=True)
def FiltrateH(fg):
    values = fg['Date/Time'].drop_duplicates()
    return values

def PlotHour(fg,values):
    date_to_filter = st.selectbox('Choisir date', values)
    st.subheader(f'Données de chaque trip à {date_to_filter}')
    st.write(fg[fg['Date/Time'] == date_to_filter])

#@st.set_option('deprecation.showPyplotGlobalUse', False)
def HistG(df):
    plt.figure(figsize = (30, 15))
    plt.hist(df["hour"],bins=24,range=(0.5,24))
    plt.title('Fréquence de durée en  heure - Uber - Avril 2014, axe x= heures et axe y= fréquence')
    st.pyplot()
    plt.hist(df["weekday"],bins=7,range = (-.5,6.5), rwidth=0.8)
    plt.title('Fréquence de jours de la semaine - Uber - Avril 2014, axe x= jours de la semaine et axe y= fréquence')


def Heatmap(br):
    gdf = br.groupby(['hour','weekday']).apply(count_rows).unstack()
    sns.heatmap(gdf, linewidths = .5)
    st.write("Heatmap entre l'heure et les jours de la semaine")

def plotweek(rt):
    plt.figure(figsize = (30, 15))
    plt.hist(rt.weekday, bins = 7, rwidth = 0.8, range = (-.5, 6.5))
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Frequence')
    plt.title('Frequence par Heure - Uber - April 2014')
    plt.xticks(np.arange(7), 'Mon Tue Wed Thu Fri Sat Sun'.split())
    

def Scatter(rt,sd,dr):
    plt.figure(figsize =(20,20), dpi=80)
    plt.scatter(sd,dr)
    plt.title('Relation entre la Longitude et la Latiude - Uber - Avril 2014, axe x= Latitude et axe y= Longitude')
    

def ScatterGrid(sd,dr):
    plt.plot(sd, dr, '.', ms = 2, alpha = .5)
    plt.grid()
    st.pyplot()

#@st.cache(suppress_st_warning=True)
def Datatransf(rt):
    rt["tpep_pickup_datetime"]= pd.to_datetime(rt["tpep_pickup_datetime"])
    rt["tpep_dropoff_datetime"]= pd.to_datetime(rt["tpep_dropoff_datetime"])
    rt['pickup_day']=rt['tpep_pickup_datetime'].dt.day_name()
    rt['dropoff_day']=rt['tpep_dropoff_datetime'].dt.day_name()
    rt['pickup_day_wd']=rt['tpep_pickup_datetime'].dt.weekday
    rt['dropoff_day_wd']=rt['tpep_dropoff_datetime'].dt.weekday
    rt['pickup_day_hour']=rt['tpep_pickup_datetime'].dt.hour
    rt['dropoff_day_hour']=rt['tpep_dropoff_datetime'].dt.hour
    st.write ("Dataset post transformation :")
    rt.sort_values(by=['pickup_day_hour'])
    st.write(rt.head(5))
    return rt

def Histviz(rt):
    plt.hist(rt["trip_distance"],bins = 30, rwidth=0.8, range=(0.5,21.5))  
    plt.title('Frequency by Trip distance - Ny Trip - January 2015, x-axis trip distance and y-axis Frequency')
    st.pyplot()


@st.cache(suppress_st_warning=True)
def Vendorsort(rt):
    rt.sort_values(by=['trip_distance'])
    return rt
    
def Vendorplot(rt):
    sns.countplot(rt['VendorID'])
    st.pyplot()
    st.write("On peut voir que le vendeur 2 vend un peu plus de billets mais les écarts restent relatifs")




def nbpassager(rt):
    st.write(rt.passenger_count.value_counts())
    sns.countplot(x='passenger_count',data=rt)
    st.pyplot()



def colorizeScatter(rt):   
    plt.hist(rt.dropoff_longitude, bins = 100, range = (-74.1, -73.9), color = 'b', alpha = 0.5, label = 'Longitude')
    plt.legend(loc = 'best')
    plt.twiny()
    plt.hist(rt.dropoff_latitude, bins = 100, range = (40.5, 41), color = 'r', label = 'Latitude')
    plt.legend(loc = 'upper left')
    st.pyplot()


def NyLongLat(rt) : 
    plt.figure(figsize = (20, 20))
    plt.plot(rt.pickup_longitude, rt.pickup_latitude, '.', ms = 2, alpha = .5)
    plt.xlim(-74.05, -73.75)
    plt.ylim(40.6, 40.98)
    plt.grid()
    st.pyplot()


@st.cache(suppress_st_warning=True)
def groupbyhours(rt):
    by_hour_dropoff = rt.groupby('dropoff_day_hour').apply(count_rows)
    by_hour_pickup = rt.groupby('pickup_day_hour').apply(count_rows)
    return by_hour_pickup, by_hour_dropoff

def plotbyhours(x,y):
    plt.figure(figsize = (15, 10))
    plt.plot(x, color = 'b', label = 'pickup')
    plt.plot(y, color = 'r', label = 'dropoff')
    plt.legend()
    st.pyplot()

def pickdrop(rt):
    plt.figure(figsize = (20, 20))
    plt.plot(rt.pickup_longitude, rt.pickup_latitude, '.', ms = 2, alpha = .5, color = 'r', label = 'pickup')
    plt.plot(rt.dropoff_longitude, rt.dropoff_latitude, '.', ms = 2, alpha = .5, color = 'g', label = 'dropoff')
    plt.xlim(-74.1, -73.7)
    plt.ylim(40.6, 41.1)
    plt.legend()
    plt.grid()
    st.pyplot()

def sidebarr():
    add_selectbox = st.sidebar.selectbox(
        "Que pensez-vous de ce Dashboard ?",
        ("Bien", "Moyen", "Nul"))
    add_button = st.sidebar.button("Envoyer")
    if (add_selectbox== "Bien" or "Moyen"):
        st.sidebar.write("Merci :)")
    elif(add_button==true & add_selectbox== "Nul"):
        st.sidebar.write("Méchant ! :(")
sidebarr() 
st.balloons()




# Début du Lab3 Data VIZ
st.title('Ce dashboard reprends les résultats des études Data du Lab1 et du Lab2')
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
# Update the progress bar with each iteration.
    latest_iteration.text('Chargement des modèles')
    bar.progress(i + 1)
    time.sleep(0.1)
'...Tout est chargé!'


#Importation du premier Dataset
st.write('Ce dataset va étudier les trajets en Uber et à New York, pour en savoir veuillez utiliser le navigateur ci dessous')
components.iframe("https://www.google.com/")
st.write('Pour en savoir plus sur Stremlit :')
components.iframe("https://docs.streamlit.io/en/latest")

df = importData("uber-raw-data-apr14.csv")

PlotDataset(df)

df= FirstTrans(df)


st.write("Affichage des fréquences d'apparition des variables")

#@st.cache(suppress_st_warning=True)

hist(df)
st.pyplot()






#Filtrer par heure
st.write("Choisissez une heure")


values =FiltrateH(df)
PlotHour(df,values)



HistG(df)
st.pyplot()

Heatmap(df)
st.pyplot()


plotweek(df)
st.pyplot()




Scatter(df,df.Lat,df.Lon)
st.pyplot()

ScatterGrid(df.Lat,df.Lon)

st.write(" Les voyages en diagonale semblent très plebiscités")

st.title ("Utilisation du Data NY TRIP 2015")

ds= importData("ny-trips-data.csv")

PlotDataset(ds)

#Transformation du Dataset

ds= Datatransf(ds)


Histviz(ds)
plt.hist(ds.trip_distance,bins = 30, rwidth=0.8, range=(0.5,21.5))  
plt.title('Frequency by Trip distance - Ny Trip - January 2015, x-axis trip distance and y-axis Frequency')
st.pyplot()


ds = Vendorsort(ds)
Vendorplot(ds)


nbpassager(ds)
st.write("Les voyages solo sont privilégiés")

ScatterGrid(ds.pickup_latitude,ds.pickup_longitude)


colorizeScatter(ds)


NyLongLat(ds)


val1, val2 =groupbyhours(ds)

plotbyhours(val1,val2)



pickdrop(ds)
