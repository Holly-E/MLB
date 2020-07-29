# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:27:56 2019

@author: Erick
"""

import pandas as pd
import json
import requests
import math
import numpy as np

df = pd.read_csv('C:/Baseballcloud/MLB Data/stadium_address_lat_long/stadium_info.csv')


#%%

# https://elevation-api.io/
# example: https://elevation-api.io/api/elevation?points=(39.90974,-106.17188),(62.52417,10.02487)&key=YOUR-API-KEY-HERE


def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


elev_key = 'li0Y6Y9Mh-64Dt5v-4ed5tdO9hV96d'
elevations = []

for ind,row in df.iterrows():
        # *** NOTE REMOVE &resolution=30-interpolated when testing ***
        lat = row['lat']
        lon = row['lng']
        loc = (lat,lon)
        elev_url = 'https://elevation-api.io/api/elevation?points={}&key={}&resolution=30-interpolated'.format(loc, elev_key)
        elev_url = elev_url.replace(" ", "")
        print(elev_url)

        # Make elevation API call
        response = requests.get(elev_url)
        print(response.status_code)
        # load response as JSON object
        resp = json.loads(response.text)
        for stad in resp['elevations']:
            elevations.append(stad['elevation'])

        
df['Elevation'] = elevations

#%%
# Convert Meters to Feet
df['Elevation Feet'] = df['Elevation'] * 3.28084
#df.to_csv('C:/Baseballcloud/MLB Data/stadium_address_lat_long/stadium_info_w_elevation.csv', index=False)
#%%
#df = pd.read_csv('C:/Baseballcloud/MLB Data/stadium_address_lat_long/stadium_info_w_elevation.csv')

#%%
years = ['2018b', '2018a', '2017', '2016', '2015'] #2019 done
for year in years: 
    locs = []
    for ind, row in df.iterrows():
        loc = str(row['lat']) + ',' + str(row['lng'])
        locs.append(loc)
        
    api_dict = {key: year[0:4] + '-06-01' for key in locs}
    #print(api_dict)

        
    FREE_API_ENDPOINT = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    # ex. http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=5e56ee7581614b4599735151190911&q=London&format=xml&date=2018-05-05&tp=1
    key2 = '693d9b7ade804a18ad254549202902' # bellevue
    tp = 1 # 24 hour interval 
    
    
    responses = [] # Holds the response from past temp api
    
    for loc, current_date in api_dict.items():
        elev_url = '{}?key={}&q={}&format=json&date={}&tp={}'.format(FREE_API_ENDPOINT, key2, loc, current_date, tp)
        print(elev_url)
        # Make elevation API call
        response = requests.get(elev_url)
        print(response.status_code)
        # load response as JSON object
        resp = json.loads(response.text)
        responses.append(resp)
        if response.status_code != 200:
            print(loc, current_date)
            break
    
    
    """
    # For some reason df1 responses are strings and df2 are dictionaries. Convert df1 resp to dict
    get_dicts = []
    for ind, row in df1.iterrows():
        resp = row['response']
        dict_resp = eval(resp) # transform string to dict
        get_dicts.append(dict_resp)
    
    df1['response'] = get_dicts
    df = df1.append(df2)
    """
    
    
    df['response'] = responses
    
    def get_density(temp_c, humid, barometric, elev_meters):
        beta = 0.0001217
        SVP = 4.5841*math.exp((18.687-temp_c/234.5)*temp_c/(257.14+temp_c))
        pressure = barometric * 0.75006 # mm Hg
        D2 = 1.2929*(273/(temp_c+273)*(pressure*math.exp(-beta*elev_meters)-0.3783*humid*SVP/100)/760)
        D1 = D2*0.06261
        air_density =0.07182*D1*(5.125/5.125)*(9.125/9.125)**2
        return air_density
    
    print(year)
    temp = []
    humidity = []
    press = []
    air = []
    for ind, row in df.iterrows():
        x = 12
        temp_c = float(row['response']['data']['weather'][0]['hourly'][x]['tempC']) # !!removed [0] after weather, x = Time_hour
        humid = float(row['response']['data']['weather'][0]['hourly'][x]['humidity']) # Humidity percent
        barometric = float(row['response']['data']['weather'][0]['hourly'][x]['pressure']) #Atmospheric pressure in millibars (mb)
        elev_meters = float(row['Elevation'])
        density = get_density(temp_c, humid, barometric, elev_meters)
        temp.append(temp_c)
        humidity.append(humid)
        press.append(barometric)
        air.append(density)
        
    
    df['Temperature C'] = temp
    df['Humidity Percent'] = humidity
    df['Barometric Pressure (mb)'] = press
    df['Air Density (K)'] = air
    
    games = pd.DataFrame()
    #path_list = ["../trackman data/Trackman By Year/2015_trackman.csv", "../trackman data/Trackman By Year/2016_trackman.csv", "../trackman data/Trackman By Year/2017_trackman.csv", "../trackman data/Trackman By Year/2018_trackman.csv", "../trackman data/Trackman By Year/2019_trackman.csv"]
    path_list = ["/2019.csv"]
    #path_list = ["/2019.csv"]
    path = "C:/Baseballcloud/MLB Data"
    for fname in path_list:
       current_csv = pd.read_csv(path + fname, header=0)
       games = games.append(current_csv)
    
    new_df = games.merge(df, how='left', left_on='home_team', right_on='Trackman')
    new_df.drop('Trackman', axis=1, inplace = True)
    
    
    efficiency = []
    phi_row = []
    theta_row = []
    true_spin = []
    for ind, row in new_df.iterrows():
        yR = 60.5-row['release_extension']
        tR = (-row['vy0']-math.sqrt(row['vy0']**2-2* row['ay']*(50-yR)))/row['ay']
        vxR = row['vx0']+row['ax']*tR
        vyR = row['vy0']+row['ay']*tR
        vzR = row['vz0']+row['az']*tR
        dv0 = row['release_speed']-math.sqrt(vxR**2 + vyR**2 + vzR**2)/1.467
        
        tf = (-vyR-math.sqrt(vyR**2-2*row['ay']*(yR-17/12)))/row['ay']
        calculated_x_mvt = (row['plate_x']-row['release_pos_x']-(vxR/vyR)*(17/12-yR))
        calculated_z_mvt = (row['plate_z']-row['release_pos_z']-(vzR/vyR)*(17/12-yR))+0.5*32.174*tf**2
        
        vxbar = (2*vxR+row['ax']*tf)/2
        vybar = (2*vyR+row['ay']*tf)/2
        vzbar = (2*vzR+row['az']*tf)/2
        vbar = math.sqrt(vxbar**2+vybar**2+vzbar**2)
        
        adrag = -(row['ax']*vxbar+row['ay']*vybar+(row['az']+32.174)*vzbar)/vbar
        cd = adrag/(row['Air Density (K)']*vbar**2)
        
        amagx = row['ax'] + adrag * vxbar/vbar
        amagy = row['ay'] + adrag * vybar/vbar
        amagz = row['az'] + adrag * vzbar/vbar + 32.174
        amag = math.sqrt(amagx**2 + amagy**2 + amagz**2)
        
        mx = 0.5*amagx*tf**2*12
        mz = 0.5*amagz*tf**2*12
        cl = amag/(row['Air Density (K)']*vbar**2)
        s = 0.4*cl/(1-2.32*cl)
        
        spinT = 78.92*s*vbar
        spinTX = spinT * (vybar * amagz - vzbar * amagy)/(amag * vbar)
        spinTY = spinT * (vzbar * amagx - vxbar * amagz)/(amag * vbar)
        spinTZ = spinT * (vxbar * amagy - vybar * amagx)/(amag * vbar)
        spin_check = math.sqrt(spinTX**2 + spinTY**2 + spinTZ**2)-spinT
        
        spin_eff = spinT/row['release_spin_rate']
    
        if amagz > 0:
            phi = (np.arctan2(amagz, amagx)*180/math.pi) + 90
        else:
            phi = 360+np.arctan2(amagz, amagx)*180/math.pi + 90
        phi_row.append(phi)
    
        if spin_eff <= 1 and spin_eff > 0:
            theta = np.arccos(spin_eff)*180/math.pi
        else:
            theta = np.nan
            spin_eff = 1 # change spin efficiency to 1 if calculated efficiency is greater than one
        
        theta_row.append(theta)
        efficiency.append(spin_eff)
        true = spin_eff * row['release_spin_rate']
        true_spin.append(true)
    
    new_df['Spin Efficiency'] = efficiency
    new_df['Phi'] = phi_row
    new_df['Theta'] = theta_row
    new_df['True Spin (rpm)'] = true_spin
    
    new_df.drop('response', axis = 1, inplace = True)
    new_df.to_csv(year+'_3D_spin.csv', index=False)
