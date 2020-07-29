# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:50:09 2019

@author: Erick
"""
import pandas as pd
import numpy as np

#%%
# Minimum NO. of pitches
min_val_region = 50
min_val_zone = 20
# Side padding in feet (taken from outside when calculating regions)
padding = 3/12

#%%

# Calculate variable for given dataframe
def calculate(variable, df):
    """
    Returns the sample size and percentage of the chosen variable for input df
    """
    if variable == 'GBP':
        # Ground Ball %
        sample_size = df['Angle'].count()
        ground_ball = df['GB'].sum() #df[df['GB']==True].shape[0]
        #print("ground balls {} sample size {}".format(ground_ball, sample_size))
        if sample_size == 0:
            return 0, 0
        else:
            gb_percent = round((ground_ball / sample_size) * 100, 1)
            #print(gb_percent, ground_ball, sample_size)
            return gb_percent, sample_size
    elif variable == 'SCP':
        # Soft Contact %
        sample_size = df['ExitSpeed'].count()
        soft_contact = df['SC'].sum()
        sc_percent = round((soft_contact / sample_size) * 100, 1)
        if sample_size == 0:
            return 0, 0
        else:
            return sc_percent, sample_size
    elif variable == 'Whiff':
        sample_size = df['Swung'].count() 
        whiff_count = df['Whiff'].sum() # calculate how many are whiffs
        whiff_percent = round((whiff_count / sample_size) * 100, 1)
        if sample_size == 0:
            return 0, 0
        else:
            return whiff_percent, sample_size
    else:
        pass


#%%

# create a zone class. Takes all columns from the zones13.csv as parameters.
# class checks if in zone
class Zone:
    def __init__(self, name, yb, yt, xl, xr):
        self.name = name
        self.y_bottom = yb
        self.y_top = yt
        self.x_left = xl
        self.x_right = xr
    
    def check_zone(self, vert, horz):
        check = 0
        if float(self.y_bottom) <= float(vert) and float(vert) < float(self.y_top):
          check += 1
        if float(self.x_left) <= float(horz) and float(horz) < float(self.x_right):
          check += 1
        if check == 2:
          return True
        else:
          return False
          
 

#%%

# Create Pitch Class
class Pitch:
    def __init__(self, name, pitch_df, total_pitches):
        self.name = name
        self.batter = ['L', 'R']
        self.df = pitch_df
        self.gb_sc_df = pitch_df[(pitch_df['PitchCall'] == 'hit_into_play')]
        self.total_pitches = total_pitches
        #print(self.gb_sc_df.shape)
    
    def get_region_insight(self, side, var, df, region_dict):
        """
        returns region dict with vert and side: ['sample', 'sample desc', 'string', '%']
        """
        region_dict['multiple'] = False
        
        if side == 'vert':
            top = df[df['Vert Region'] == 'Top']
            top_percent, top_sample = calculate(var, top)
            bottom = df[df['Vert Region'] == 'Bottom']
            bottom_percent, bottom_sample = calculate(var, bottom)
            full_list.append([self.name, 'top', var, top_percent, top_sample])
            full_list.append([self.name, 'bottom', var, bottom_percent, bottom_sample])
            
            if top_percent == 0 and bottom_percent == 0:
                region_dict['sample'] = top_sample + bottom_sample
                region_dict['percent'] = bottom_percent
                region_dict['string'] = 'No high or lows resulted in any {}.'.format( var)
                region_dict['sample desc'] = 'None top or bottom'
            elif top_percent > bottom_percent:
                # Perform this variable better in top of zone
                region_dict['sample'] = top_sample
                region_dict['sample desc'] = 'Top of zone'
                region_dict['percent'] = top_percent
                region_dict['string'] = "Your highest {} is in the top of the strikezone at {}%.".format(var, top_percent)
                #print(True)
                #print(top_sample)
                #print(region_dict)
            elif top_percent == bottom_percent:
                region_dict['sample'] = top_sample + bottom_sample
                region_dict['percent'] = top_percent
                region_dict['string'] = "Top and bottom are equal for {} at {}%.".format(var, top_percent)
                region_dict['sample desc'] = "Top and bottom"
                region_dict['multiple'] = True
            elif top_percent < bottom_percent:
                # Perform this variable better in bottom of zone
                region_dict['sample'] = bottom_sample
                region_dict['percent'] = bottom_percent
                region_dict['string'] = "Your highest {} is in the bottom of the strikezone at {}%.".format(var, bottom_percent)
                region_dict['sample desc'] = 'Bottom of zone'
                
        elif side == 'side':
            inside = df[df['Side Region'] == 'Inside']
            inside_percent, inside_sample = calculate(var, inside)
            outside = df[df['Side Region'] == 'Outside']
            outside_percent, outside_sample = calculate(var, outside)
            full_list.append([self.name, 'inside', var, inside_percent, inside_sample])
            full_list.append([self.name, 'outside', var, outside_percent, outside_sample])
        #print('top percent ' + str(top_percent))
        #print('bottom percent ' + str(bottom_percent))
            
            if inside_percent == 0 and outside_percent == 0:
                region_dict['sample'] = inside_sample + outside_sample
                region_dict['percent'] = inside_percent
                region_dict['string'] = 'No inside or outsides resulted in any {}.'.format(var)
                region_dict['sample desc'] = 'None inside or outside'
            elif inside_percent > outside_percent:
                # Perform this variable better inside the zone
                region_dict['sample'] = inside_sample 
                region_dict['percent'] = inside_percent
                region_dict['string'] = "Your highest {} is in the inside of the strikezone at {}%.".format(var,inside_percent)
                region_dict['sample desc'] = 'Inside'
            elif inside_percent == outside_percent:
                region_dict['sample'] = inside_sample + outside_sample
                region_dict['percent'] = inside_percent
                region_dict['string'] = "Inside and outside are equal for {} at {}t%.".format(var, inside_percent)
                region_dict['sample desc'] = 'Inside & outside'
                region_dict['multiple'] = True
            elif inside_percent < outside_percent:
                # Perform this variable better outside the zone
                region_dict['sample'] = outside_sample 
                region_dict['percent'] = outside_percent
                region_dict['string'] = "Your highest {} is in the outside of the strikezone at {}%.".format(var, outside_percent)
                region_dict['sample desc'] = 'Outside'
            
        return region_dict
            
            
    def get_zone_insight(self, var, df, zone_dict):
        """
        returns zone dict with best zones: ['sample', 'sample desc', 'string', '%']
        """
        zones = list(df['Zone Number'].unique())
        
        max_dict = {
            'percent': 0,
            'sample': 0,
            'zone': [],
            'multiple': False
            }
        
        # loop through each zone, calculating whiff, gb % and sc %
        for zone in zones:
            current_zone = df[df['Zone Number'] == zone]
            zone_percent, zone_sample = calculate(var, current_zone)
            full_list.append([self.name, zone, var, zone_percent, zone_sample])
            # update max_dict values
            if zone_percent > max_dict['percent']:
                # Performs variable better in this zone
                max_dict['sample'] = zone_sample
                max_dict['zone'] = [zone]
                max_dict['percent'] = zone_percent
             
            elif zone_percent == max_dict['percent']:
                max_dict['zone'].append(zone)
                max_dict['sample'] += zone_sample
                
        if max_dict['percent'] == 0:
            max_dict['sample desc'] = 'None found'
            max_dict['string'] = 'No {} resulted from any {}'.format(var, self.name)
        else:
            max_dict['sample desc']= 'zone {}'.format(max_dict['zone'])
            max_dict['string'] = "Your highest {} is in zone {} at {}%.".format(var, max_dict['zone'], max_dict['percent'])
            if len(max_dict['zone']) > 1:
                max_dict['multiple'] = True
                
        for key, val in max_dict.items():
            # update zone dictionary with max zone info for current variable
            if key != 'zone':
                zone_dict[key] = val
                
        return zone_dict
                 


#%%
# Level Data 
    def level_1(self):
        """ 
        pitches >= min required, but sample size of max zone for variable < zone_min
        provides region data for all batters (no split)
        """
        global output
        l1_dict = {
                'Cluster': self.name,
                'split': 'No'
                }
        
        # update output with region insights for each variable        
        for var in variables:
            if var == 'GBP' or var == 'SCP':
                # use gb_sc dataframe
                l1_dict['var'] = var
                l1_dict = self.get_region_insight("vert", var, self.gb_sc_df, l1_dict)
                l1_dict['level'] = 'Top / Bottom'
                output = output.append(l1_dict, ignore_index=True)
                l1_dict = self.get_region_insight("side", var, self.gb_sc_df, l1_dict)
                l1_dict['level'] = 'Inside / Outside'
                output = output.append(l1_dict, ignore_index=True)
            elif var == 'Whiff':
                # use whiff dataframe 
                l1_dict['var'] = var
                l1_dict = self.get_region_insight("vert", var, self.df, l1_dict)
                l1_dict['level'] = 'Top / Bottom'
                output = output.append(l1_dict, ignore_index=True)
                l1_dict = self.get_region_insight("side", var, self.df, l1_dict)
                l1_dict['level'] = 'Inside / Outside'
                output = output.append(l1_dict, ignore_index=True)
                
    def level_2(self):
        """
        pitches < min required for left/right split, but sample size of max zone for variable >= zone_min
        provides zone data for all batters (no split)
        """
        global output
        l2_dict = {
                'Cluster': self.name,
                'level': 'Best Zone',
                'split': 'No'
                }
        # update output with zone insights for each variable
        for var in variables:
            if var == 'GBP' or var == 'SCP':
                # use gb_sc dataframe
                l2_dict['var'] = var
                l2_dict = self.get_zone_insight(var, self.gb_sc_df, l2_dict)
                output = output.append(l2_dict, ignore_index=True)
            elif var == 'Whiff':
                # use whiff dataframe 
                l2_dict['var'] = var
                l2_dict = self.get_zone_insight(var, self.df, l2_dict)
                output = output.append(l2_dict, ignore_index=True)
    
    def level_3(self):
        """
        pitches >= min required for left/right split, but sample size of max zone for variable < zone_min
        provides region data for left/right split
        """
        global output
        l3_dict = {
                'Cluster': self.name,
                }
        
        # split into left and right data & update output with region insights
        for hand in self.batter:
            for var in variables:
                if var == 'GBP' or var == 'SCP':
                    # use gb_sc dataframe, and filter for hand
                    hand_df = self.gb_sc_df[self.gb_sc_df['BatterSide'] == hand]
                elif var == 'Whiff':
                    # use whiff dataframe, and filter for hand
                    hand_df = self.df[self.df['BatterSide'] == hand]

                l3_dict['var'] = var
                l3_dict['split'] = hand
                full_list.append([hand])
                l3_dict = self.get_region_insight("vert", var, hand_df, l3_dict)
                l3_dict['level'] = 'Top / Bottom LR Split'
                l3_dict['string'] = "Against " + hand + " handed batters: " + l3_dict['string']
                output = output.append(l3_dict, ignore_index=True)
                full_list.append([hand])
                l3_dict = self.get_region_insight("side", var, hand_df, l3_dict)
                l3_dict['level'] = 'Inside / Outside LR Split'
                l3_dict['string'] = "Against " + hand + " handed batters: " + l3_dict['string']
                output = output.append(l3_dict, ignore_index=True)
                
    def level_4(self):
        """
        full depth (for now)
        provides zone data for left/right split
        """
        global output
        l4_dict = {
                'Cluster': self.name,
                'level': 'Best Zone LR Split'
                }
        
        # split into left and right data & update output with zone insights
        for hand in self.batter:
            for var in variables:
                if var == 'GBP' or var == 'SCP':
                    # use gb_sc dataframe, and filter for hand
                    hand_df = self.gb_sc_df[self.gb_sc_df['BatterSide'] == hand]
                elif var == 'Whiff':
                    # use whiff dataframe, and filter for hand
                    hand_df = self.df[self.df['BatterSide'] == hand]

                l4_dict['var'] = var
                l4_dict['split'] = hand
                full_list.append([hand])
                l4_dict = self.get_zone_insight(var, hand_df, l4_dict)
                l4_dict['string'] = "Against " + hand + " handed batters: " + l4_dict['string']
                output = output.append(l4_dict, ignore_index=True)
          
#%%
"""
READ DATA
"""
# Launch angle break for ground balls
angle = 10
# Soft contact rate
sc_mph = 75
variables = ['GBP', 'SCP', 'Whiff']              
pitches = ['Cutter', 'ChangeUp', 'Splitter', 'Curveball', 'Slider'] #['Fastball', 'Sinker', 'Slider']
#hand = ['Right', 'Left']
# Read relevant fields as df
fields = ['Pitcher', 'PitcherId', 'BatterSide', 'TaggedPitchType', 'PitchCall',
       'HitType', 'PlateLocHeight', 'PlateLocSide', 'ExitSpeed', 'Angle']


#for i in hand:
df = pd.read_csv('C:/Baseballcloud/MLB Data/Clusters/data_w_labels_all_L.csv')

#df = pd.read_hdf('C:/Baseballcloud/MLB Data/Clusters/data_w_labels_all_R.h5')

#%%
cols = list(df.columns)
#print(cols)
print(df.events.value_counts())
print(df.description.value_counts())
#%%
change_dict = {
        'launch_angle' : 'Angle',
        'plate_z': 'PlateLocHeight',
        'plate_x': 'PlateLocSide',
        'stand' : 'BatterSide',
        'launch_speed' : 'ExitSpeed',
        'events': 'PlayResult',
        'description': 'PitchCall',  
        }

df.rename(columns=change_dict, inplace=True)
cols = list(df.columns)
print(cols)
#%%
# Grab strikezone data 
zones = pd.read_csv('C:/Baseballcloud/MLB Data/Clusters/Zone dimensions13.csv')

play_result = ['single', 'double', 'triple', 'home_run', 'field_error', 'field_out','grounded_into_double_play', 'force_out', 'double_play', 'triple_play']
call = ['swinging_strike', 'foul' ]

df['GB'] = np.where(df['Angle'] < angle, 1, 0)
df['SC'] = np.where(df['ExitSpeed'] <= sc_mph, 1, 0)
df['Swung'] = np.where(df['PlayResult'].isin(play_result) | df['PitchCall'].isin(call), 1, 0)
df['Whiff'] = np.where(df['PitchCall'] == 'swinging_strike', 1, 0)

# Remove pitch data with NaN in Zone Indicator Columns
df = df[pd.notnull(df['PlateLocHeight'])]
df = df[pd.notnull(df['PlateLocSide'])]
        
# Replace infinity string type. Bottom & left replaced with negative infinity
zones['y-top'] = zones['y-top'].replace('infinity', float('inf'))
zones['y-bottom'] = zones['y-bottom'].replace('infinity', float('-inf'))
zones['x-left'] = zones['x-left'].replace('infinity', float('-inf'))
zones['x-right'] = zones['x-right'].replace('infinity', float('inf'))

# create a class instance for each zone and save in zone list
zone_list = []
for index, row in zones.iterrows():
    zone_list.append(Zone(row['Zone'], row['y-bottom'], row['y-top'], row['x-left'], row['x-right']))

# check zone of each row in df and add zone number
number = []
vert_region = []
side_region = []
 
for index, row in df.iterrows():
    vert = row['PlateLocHeight']
    horz = row['PlateLocSide']
    num = np.nan
    count = 0
    for item in zone_list:
        current = item.check_zone(vert, horz)
        if current == True:
            number.append(item.name)
            count += 1
            # Get vertical side info
            if (3.05 <= vert) and (vert <= (4.16 + padding)) and ((-.71 - padding) <= horz) and ((horz <= .71 + padding)):
                vert_region.append('Top')
            elif (.83 - padding <= vert) and (vert <= (1.94)) and ((-.71 - padding) <= horz) and ((horz <= .71 + padding)):
                vert_region.append('Bottom')
            else:
                vert_region.append(np.nan)
            
            # Get inside/outside info, have to look at batter side
            if row['BatterSide'] == 'L':
                if (.83 <= vert) and (vert <= (4.16 + padding)) and (.24 <= horz) and ((horz <= .71 + padding)):
                    side_region.append('Inside')
                elif (.83 - padding <= vert) and (vert <= (4.16 + padding)) and ((-.71 - padding) <= horz) and (horz <= -.24):
                    side_region.append('Outside')
                else:
                    side_region.append(np.nan)
            elif row['BatterSide'] == 'R':
                if (.83 - padding <= vert) and (vert <= (4.16 + padding)) and ((-.71 - padding) <= horz) and (horz <= -.24):
                    side_region.append('Inside')
                elif (.83 <= vert) and (vert <= (4.16 + padding)) and (.24 <= horz) and ((horz <= .71 + padding)):
                    side_region.append('Outside')
                else:
                    side_region.append(np.nan)
            else:
                side_region.append(np.nan)
    if count != 1:
        print("Zone assignment error: index {} count {}".format(index, count))
        
df['Zone Number'] = number
df['Vert Region'] = vert_region
df['Side Region'] = side_region
   
# Combine outer zone 'a/b' using 'replace'
initial = ['10a', '10b', '11a', '11b', '12a', '12b', '13a', '13b']
fix = [10, 10, 11, 11, 12, 12, 13, 13]
df['Zone Number'] = df['Zone Number'].replace(initial, fix)
df['Zone Number'] = pd.to_numeric(df['Zone Number'])


#%%
"""
#RUN THIS IF YOU WANT TO CHECK AVGS

avgs = pd.DataFrame()
for var in variables:
    if var == 'SCP':
        percent, sample_size = calculate(var, df_gb_sc)
        avg_dict = {}
        avg_dict['var'] = var
        avg_dict['sample'] = sample_size
        avg_dict['string'] = "2015_19 MLB {} {}% with sample size {}.".format(var, percent, sample_size)
        avg_dict['majors'] = "Average MLB SCP = 18.1%"
        avgs.append(avg_dict, ignore_index=True)
    elif var == "GBP":
        percent, sample_size = calculate(var, df_gb_sc)
        avg_dict = {}
        avg_dict['var'] = var
        avg_dict['sample'] = sample_size
        avg_dict['string'] = "2015-2019 MLB {} {}% with sample size {}.".format(var, percent, sample_size)
        avg_dict['majors'] = "Average MLB GPB = 44.8%"
        avgs.append(avg_dict, ignore_index=True)
    elif var == "Whiff":
        percent, sample_size = calculate(var, df)
        avg_dict = {}
        avg_dict['var'] = var
        avg_dict['sample'] = sample_size
        avg_dict['string'] = "2015_19 MLB {} {}% with sample size {}.".format(var, percent, sample_size)
        avg_dict['majors'] = "Average MLB Whiff = 9.5%"
        avgs.append(avg_dict, ignore_index=True)
"""     

#%%

#Create instance for player level data and update output dataframe with insights
count = 0

# Get pitchers full data all zones / sides
#full_data = pd.DataFrame(columns = ['Location', 'Pitch', 'Var', 'Percent', 'Sample'])   
full_list = [] 
output_cols = ['Cluster', 'level', 'split', 'sample', 'percent', 'var', 'sample desc', 'string', 'multiple']
output = pd.DataFrame(columns = output_cols)
# For each cluster:
#clusters = df['Cluster'].unique()
for cluster in range(30):
    current_cluster = df[df['Cluster'] == cluster]
    count += 1
    print(count)

    # Grab pitch type data as a dictionary, key = pitch name, val = count
    #pitches = dict(current_cluster['AutoPitchType'].value_counts())
    key = 'all pitches'
    total_pitches = len(df.index)
    name = cluster
    #for key, val in pitches.items():
        #if key != 'Undefined':
    #pitch_df = pd.DataFrame()
    instance = Pitch(name, current_cluster, total_pitches) # removed key, val 

    # Get the insights at each of the following levels
    instance.level_1()
    instance.level_2()
    instance.level_3()
    instance.level_4()

#%%
change_dict = {
        'pfx_x' : 'HorzBreak',
        'pfx_z': 'InducedVertBreak',
        'release_speed': 'RelSpeed',  
        }

df.rename(columns=change_dict, inplace=True)

#%%
df.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_pitches_w_zone.csv', index = False)

#dfl = pd.read_hdf('C:/Baseballcloud/MLB Data/Clusters/L_pitches_w_zone.h5')
#dfl.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_pitches_w_zone.csv', index = False)

#%%
output.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_insights.csv', index=False)
#dfl_output = pd.read_hdf('C:/Baseballcloud/MLB Data/Clusters/L_insights.h5')
#dfl_output.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_insights.csv', index=False)
full_list_df = pd.DataFrame.from_records(full_list)
#dfl_full_list = pd.read_hdf('C:/Baseballcloud/MLB Data/Clusters/L_insights_full_list.h5')
#dfl_full_list.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_insights_full_list.csv', index = False)
full_list_df.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_insights_full_list.csv', index = False)
