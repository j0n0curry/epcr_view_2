#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import division
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import requests
import glob
import os
import numpy as np
from pandas import datetime
import warnings
warnings.filterwarnings('ignore')
from typing import Callable
pd.options.display.float_format = '{:,.0f}'.format
from pathlib import Path



st.set_page_config(layout="wide")

st.button("Run")

#Written Josh Mehl - Jonathan Curry 2021

class WellDataManager:
    """Parent class for importing data from any instrument"""
    def __init__(self,
                 files,
                 run_name_split_loc=1,
                 group_name=""):
        super().__init__()
        # project attributes
        self.file_names = files
        self.group_name = group_name
        self.run_name = ""
        self.split_char_loc = run_name_split_loc
        self.run_df = pd.DataFrame()
        self.group_df = pd.DataFrame()

        # csv read attributes
        self.tables = 1
        self.index_column = [0]
        self.header_row = [1]
        self.row_count = [8]
        self.column_names = ['Row_ID', 'Col_ID', 'Value']

    def concatenate_dataframes(self):
        for each_file in self.file_names:
            self.get_run_name(each_file)
            self.build_dataframes(each_file)
            self.group_df = pd.concat([self.group_df, self.run_df], ignore_index=True)
        # print(self.group_df)
        return self.group_df

    def build_dataframes(self, each_file):
        self.read_file(each_file)
        self.coerce_numeric_values()
        self.run_df['Group_ID'] = self.group_name
        self.run_df['File_root'] = each_file
        self.run_df['Run_ID'] = self.run_name
        # print(self.run_df)

    def coerce_numeric_values(self):
        # may be used used to force values to numeric.  Not applicable to all instruments
        pass

    def read_file(self, file_name):
        """Reads Initial Data from CSV file"""
        df = pd.read_csv(file_name, header=self.header_row, nrows=self.row_count, index_col=self.index_column)
        df = df.stack()
        self.run_df = df.reset_index()
        self.run_df.columns = self.column_names

    def get_run_name(self, file_name):
        """Splits string to get run name from file name."""
        self.run_name = file_name[:self.split_char_loc]
        self.run_file_in = os.path.basename(csv)
        

class ArtelVMSManager(WellDataManager):
    """Class that handles Well Data Data"""
    def __init__(self,
                 files,
                 run_name_split_loc=1,
                 group_name=""):
        super().__init__(files,
                         run_name_split_loc,
                         group_name)

        # csv read attributes
        self.tables = 1
        self.index_column = [0]
        self.header_row = [18]
        self.row_count = [8]
        self.column_names = ['Row_ID', 'Col_ID', 'Volume']

    def coerce_numeric_values(self):
        """Coerce the 'volume' data to numeric.  Otherwise mixture of strings and numeric values"""
        num_col = self.column_names[2]
        self.run_df[num_col] = pd.to_numeric(self.run_df[num_col], errors='coerce')


class ArayaManager(WellDataManager):
    """Class that handles Well Data Data"""
    def __init__(self,
                 files,
                 run_name_split_loc=6,
                 group_name="",
                 dyes=None,
                 separate_column=True):
        super().__init__(files,
                         run_name_split_loc,
                         group_name)

        if dyes is None:
            dyes = ['FAM', 'VIC', 'ROX']

        # Ayara-specific items
        self.separate_column_per_dye = separate_column
        self.channel_df = pd.DataFrame()
        self.dyes = dyes
        self.channels = ['CH1', 'CH2', 'CH3']

        # csv read attributes
        self.tables = 3
        self.index_column = ["<>", "<>", "<>"]
        self.header_row = [5, 23, 41]
        self.row_count = [16, 16, 16]

        if self.separate_column_per_dye:
            # TODO: generalize for other dye names
            self.column_names = ['Row_ID', 'Col_ID', 'FAM_RFU', 'VIC_RFU', 'ROX_RFU']
        else:
            self.column_names = ['Row_ID', 'Col_ID', 'RFU', 'Channel', 'Dye']

    def read_each_channel(self, file_name, ch):
        """Reads Individual Channel Data from CSV file"""
        df = pd.read_csv(file_name,
                         header=self.header_row[ch],
                         nrows=self.row_count[ch],
                         na_values="<>")

        # Need to shift to get rid of annoying '<>'.  Otherwise won't parse correctly.
        df = df.shift(periods=1, axis='columns')
        #df.drop('<>', axis=1, inplace=True)

        # Stack df for various dyes and add additional columns
        df = df.stack()
        self.channel_df = df.reset_index()

        # For separate columns for each dye, rename RFU columns.  pd.concat() method does the rest!
        if self.separate_column_per_dye:
            self.channel_df.columns = self.column_names[0:3]
            self.channel_df.rename(columns={'FAM_RFU': f'{self.dyes[ch]}_RFU'},
                                   inplace=True)

        # case to stack all dyes into common RFU and Dye channels.
        else:
            self.channel_df['Channel'] = self.channels[ch]
            self.channel_df['Dye'] = self.dyes[ch]
            self.channel_df.columns = self.column_names


    def read_file(self, file_name):
        """Reads Each Channel Data from CSV file"""

        # loops through the 3 channel tables in the csv output files.
        self.run_df = pd.DataFrame()
        for ch in range(self.tables):
            self.read_each_channel(file_name, ch)

            # case to have separate columns for each dye
            if self.separate_column_per_dye:
                self.channel_df = self.channel_df[self.channel_df.columns.difference(self.run_df.columns)]
                self.run_df = pd.concat([self.run_df, self.channel_df], axis=1)

            # case to stack all dyes into common RFU and Dye channels.
            else:
                self.run_df = pd.concat([self.run_df, self.channel_df], ignore_index=True)

        # Force columns to correct order.  Fixed bug with concat of separate dye columns.
        self.run_df = self.run_df[self.column_names]

    def get_run_name(self, file_name):
        """Splits string to get run name from file name."""
        self.run_name = file_name[-(self.split_char_loc+4):-4]
        
        




#Class instantiation - class contains self variables which passed within. 




controls = {'P19': 'A1500', 'O19' : 'A1500', 'O20': 'A1500',
           'P21': 'NEG', 'O21': 'NEG', 'O22': 'NEG',
           'O23':'S06', 'P23':'S06', 'O24': 'S06'}



quad_384 = {'A1':'QUAD 1','A2':'QUAD 2','A3':'QUAD 1','A4':'QUAD 2','A5':'QUAD 1','A6':'QUAD 2','A7':'QUAD 1','A8':'QUAD 2','A9':'QUAD 1','A10':'QUAD 2','A11':'QUAD 1','A12':'QUAD 2','A13':'QUAD 1','A14':'QUAD 2','A15':'QUAD 1','A16':'QUAD 2','A17':'QUAD 1','A18':'QUAD 2','A19':'QUAD 1','A20':'QUAD 2','A21':'QUAD 1','A22':'QUAD 2','A23':'QUAD 1','A24':'QUAD 2','B1':'QUAD 3','B2':'QUAD 4','B3':'QUAD 3','B4':'QUAD 4','B5':'QUAD 3','B6':'QUAD 4','B7':'QUAD 3','B8':'QUAD 4','B9':'QUAD 3','B10':'QUAD 4','B11':'QUAD 3','B12':'QUAD 4','B13':'QUAD 3','B14':'QUAD 4','B15':'QUAD 3','B16':'QUAD 4','B17':'QUAD 3','B18':'QUAD 4','B19':'QUAD 3','B20':'QUAD 4','B21':'QUAD 3','B22':'QUAD 4','B23':'QUAD 3','B24':'QUAD 4','C1':'QUAD 1','C2':'QUAD 2','C3':'QUAD 1','C4':'QUAD 2','C5':'QUAD 1','C6':'QUAD 2','C7':'QUAD 1','C8':'QUAD 2','C9':'QUAD 1','C10':'QUAD 2','C11':'QUAD 1','C12':'QUAD 2','C13':'QUAD 1','C14':'QUAD 2','C15':'QUAD 1','C16':'QUAD 2','C17':'QUAD 1','C18':'QUAD 2','C19':'QUAD 1','C20':'QUAD 2','C21':'QUAD 1','C22':'QUAD 2','C23':'QUAD 1','C24':'QUAD 2','D1':'QUAD 3','D2':'QUAD 4','D3':'QUAD 3','D4':'QUAD 4','D5':'QUAD 3','D6':'QUAD 4','D7':'QUAD 3','D8':'QUAD 4','D9':'QUAD 3','D10':'QUAD 4','D11':'QUAD 3','D12':'QUAD 4','D13':'QUAD 3','D14':'QUAD 4','D15':'QUAD 3','D16':'QUAD 4','D17':'QUAD 3','D18':'QUAD 4','D19':'QUAD 3','D20':'QUAD 4','D21':'QUAD 3','D22':'QUAD 4','D23':'QUAD 3','D24':'QUAD 4','E1':'QUAD 1','E2':'QUAD 2','E3':'QUAD 1','E4':'QUAD 2','E5':'QUAD 1','E6':'QUAD 2','E7':'QUAD 1','E8':'QUAD 2','E9':'QUAD 1','E10':'QUAD 2','E11':'QUAD 1','E12':'QUAD 2','E13':'QUAD 1','E14':'QUAD 2','E15':'QUAD 1','E16':'QUAD 2','E17':'QUAD 1','E18':'QUAD 2','E19':'QUAD 1','E20':'QUAD 2','E21':'QUAD 1','E22':'QUAD 2','E23':'QUAD 1','E24':'QUAD 2','F1':'QUAD 3','F2':'QUAD 4','F3':'QUAD 3','F4':'QUAD 4','F5':'QUAD 3','F6':'QUAD 4','F7':'QUAD 3','F8':'QUAD 4','F9':'QUAD 3','F10':'QUAD 4','F11':'QUAD 3','F12':'QUAD 4','F13':'QUAD 3','F14':'QUAD 4','F15':'QUAD 3','F16':'QUAD 4','F17':'QUAD 3','F18':'QUAD 4','F19':'QUAD 3','F20':'QUAD 4','F21':'QUAD 3','F22':'QUAD 4','F23':'QUAD 3','F24':'QUAD 4','G1':'QUAD 1','G2':'QUAD 2','G3':'QUAD 1','G4':'QUAD 2','G5':'QUAD 1','G6':'QUAD 2','G7':'QUAD 1','G8':'QUAD 2','G9':'QUAD 1','G10':'QUAD 2','G11':'QUAD 1','G12':'QUAD 2','G13':'QUAD 1','G14':'QUAD 2','G15':'QUAD 1','G16':'QUAD 2','G17':'QUAD 1','G18':'QUAD 2','G19':'QUAD 1','G20':'QUAD 2','G21':'QUAD 1','G22':'QUAD 2','G23':'QUAD 1','G24':'QUAD 2','H1':'QUAD 3','H2':'QUAD 4','H3':'QUAD 3','H4':'QUAD 4','H5':'QUAD 3','H6':'QUAD 4','H7':'QUAD 3','H8':'QUAD 4','H9':'QUAD 3','H10':'QUAD 4','H11':'QUAD 3','H12':'QUAD 4','H13':'QUAD 3','H14':'QUAD 4','H15':'QUAD 3','H16':'QUAD 4','H17':'QUAD 3','H18':'QUAD 4','H19':'QUAD 3','H20':'QUAD 4','H21':'QUAD 3','H22':'QUAD 4','H23':'QUAD 3','H24':'QUAD 4','I1':'QUAD 1','I2':'QUAD 2','I3':'QUAD 1','I4':'QUAD 2','I5':'QUAD 1','I6':'QUAD 2','I7':'QUAD 1','I8':'QUAD 2','I9':'QUAD 1','I10':'QUAD 2','I11':'QUAD 1','I12':'QUAD 2','I13':'QUAD 1','I14':'QUAD 2','I15':'QUAD 1','I16':'QUAD 2','I17':'QUAD 1','I18':'QUAD 2','I19':'QUAD 1','I20':'QUAD 2','I21':'QUAD 1','I22':'QUAD 2','I23':'QUAD 1','I24':'QUAD 2','J1':'QUAD 3','J2':'QUAD 4','J3':'QUAD 3','J4':'QUAD 4','J5':'QUAD 3','J6':'QUAD 4','J7':'QUAD 3','J8':'QUAD 4','J9':'QUAD 3','J10':'QUAD 4','J11':'QUAD 3','J12':'QUAD 4','J13':'QUAD 3','J14':'QUAD 4','J15':'QUAD 3','J16':'QUAD 4','J17':'QUAD 3','J18':'QUAD 4','J19':'QUAD 3','J20':'QUAD 4','J21':'QUAD 3','J22':'QUAD 4','J23':'QUAD 3','J24':'QUAD 4','K1':'QUAD 1','K2':'QUAD 2','K3':'QUAD 1','K4':'QUAD 2','K5':'QUAD 1','K6':'QUAD 2','K7':'QUAD 1','K8':'QUAD 2','K9':'QUAD 1','K10':'QUAD 2','K11':'QUAD 1','K12':'QUAD 2','K13':'QUAD 1','K14':'QUAD 2','K15':'QUAD 1','K16':'QUAD 2','K17':'QUAD 1','K18':'QUAD 2','K19':'QUAD 1','K20':'QUAD 2','K21':'QUAD 1','K22':'QUAD 2','K23':'QUAD 1','K24':'QUAD 2','L1':'QUAD 3','L2':'QUAD 4','L3':'QUAD 3','L4':'QUAD 4','L5':'QUAD 3','L6':'QUAD 4','L7':'QUAD 3','L8':'QUAD 4','L9':'QUAD 3','L10':'QUAD 4','L11':'QUAD 3','L12':'QUAD 4','L13':'QUAD 3','L14':'QUAD 4','L15':'QUAD 3','L16':'QUAD 4','L17':'QUAD 3','L18':'QUAD 4','L19':'QUAD 3','L20':'QUAD 4','L21':'QUAD 3','L22':'QUAD 4','L23':'QUAD 3','L24':'QUAD 4','M1':'QUAD 1','M2':'QUAD 2','M3':'QUAD 1','M4':'QUAD 2','M5':'QUAD 1','M6':'QUAD 2','M7':'QUAD 1','M7':'QUAD 1','M8':'QUAD 2','M9':'QUAD 1','M10':'QUAD 2','M11':'QUAD 1','M12':'QUAD 2','M13':'QUAD 1','M14':'QUAD 2','M15':'QUAD 1','M16':'QUAD 2','M17':'QUAD 1','M18':'QUAD 2','M19':'QUAD 1','M20':'QUAD 2','M21':'QUAD 1','M22':'QUAD 2','M23':'QUAD 1','M24':'QUAD 2','N1':'QUAD 3','N2':'QUAD 4','N3':'QUAD 3','N4':'QUAD 4','N5':'QUAD 3','N6':'QUAD 4','N7':'QUAD 3','N8':'QUAD 4','N9':'QUAD 3','N10':'QUAD 4','N11':'QUAD 3','N12':'QUAD 4','N13':'QUAD 3','N14':'QUAD 4','N15':'QUAD 3','N16':'QUAD 4','N17':'QUAD 3','N18':'QUAD 4','N19':'QUAD 3','N20':'QUAD 4','N21':'QUAD 3','N22':'QUAD 4','N23':'QUAD 3','N24':'QUAD 4','O1':'QUAD 1','O2':'QUAD 2','O3':'QUAD 1','O4':'QUAD 2','O5':'QUAD 1','O6':'QUAD 2','O7':'QUAD 1','O8':'QUAD 2','O9':'QUAD 1','O10':'QUAD 2','O11':'QUAD 1','O12':'QUAD 2','O13':'QUAD 1','O14':'QUAD 2','O15':'QUAD 1','O16':'QUAD 2','O17':'QUAD 1','O18':'QUAD 2','O19':'QUAD 1','O20':'QUAD 2','O21':'QUAD 1','O22':'QUAD 2','O23':'QUAD 1','O24':'QUAD 2','P1':'QUAD 3','P2':'QUAD 4','P3':'QUAD 3','P4':'QUAD 4','P5':'QUAD 3','P6':'QUAD 4','P7':'QUAD 3','P8':'QUAD 4','P9':'QUAD 3','P10':'QUAD 4','P11':'QUAD 3','P12':'QUAD 4','P13':'QUAD 3','P14':'QUAD 4','P15':'QUAD 3','P16':'QUAD 4','P17':'QUAD 3','P18':'QUAD 4','P19':'QUAD 3','P20':'QUAD 4','P21':'QUAD 3','P22':'QUAD 4','P23':'QUAD 3','P24':'QUAD 4',}



#select the method to use for the types of data to calculate and which placvements of control materials for OQ or other 
#experiment types

def normalise_values(final):
    final['Well'] = final['Row_ID']+final['Col_ID']
    final ["norm_RNaseP"] =  final["VIC_RFU"] / final["ROX_RFU"]
    final ["norm_N_Cov"] =  final["FAM_RFU"] / final ["ROX_RFU"]
    final['order'] = np.arange(len(final))  
    final['norm_zscore'] = (final.ROX_RFU - final.ROX_RFU.mean())/final.ROX_RFU.std(ddof=0)
    final['cfo_zscore'] = (final.VIC_RFU - final.VIC_RFU.mean())/final.VIC_RFU.std(ddof=0)
    final['fam_zscore'] = (final.FAM_RFU - final.FAM_RFU.mean())/final.FAM_RFU.std(ddof=0)
    final['control'] = final['Well'].map(controls).fillna('live')
    
#convert araya files to date - time and week. Eventually - change to iso-datetime function when current is depreciated.
def araya_date_time(final):
    final["file_name"]=final['File_root'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) +'.csv'
    final['file_name']=final['file_name'].str.lstrip("0")
    final['date_time'] = final.file_name.str.split('_', n=1).str[0]
    #breakout information in to sensible filter groups 
    final.date_time.str.slice(start = -6)
    final['date_time'] =  pd.to_datetime(final['date_time'], format='%Y%m%d%H%M%S')
    final['date'] = final['date_time'].apply( lambda final : 
    datetime(year=final.year, month=final.month, day=final.day))
    final['Week'] = final['date_time'].dt.week


path = st.text_input('Please copy and paste the full path to the Araya Files you would like to view: ')
path = Path(path)


st.text(path)

#(Convert dataframes in to metric ePCR - use comp.head() to gather headers for df. zscores are calculated for the files imported - not useful if data is non-linear.Order is time /date ordered but can be file name ordered) 


files = glob.glob(str(path)+ '\*.csv')
st.text(files)
arrays = ArayaManager(files)


    
comp = arrays.concatenate_dataframes()
normalise_values(comp)
araya_date_time(comp)


conditions = [
    (comp['norm_N_Cov'] <= 4.0) & (comp['norm_RNaseP'] > 2.0),
    (comp['norm_N_Cov'] > 4.0) & (comp['norm_N_Cov'] <= 9.0) & (comp['norm_RNaseP'] >1.0),
    (comp['norm_N_Cov'] >= 9.0) & (comp['norm_RNaseP'] >=1.0),
    (comp['norm_N_Cov'] >= 9.0) & (comp['norm_RNaseP']<= 1.0),
    (comp['norm_N_Cov'] <= 4.0) & (comp['norm_RNaseP'] <=2.0),
    (comp['norm_N_Cov'] > 3.0) & (comp['norm_N_Cov'] <= 9.0) & (comp['norm_RNaseP'] <1.0)]

# create a list of the values we want to assign for each condition
values = ['Negative_sample', 'PLOD', 'N_Cov_Positive_Sample', 'Control_N_Cov', 'No_Call','Background_PLOD']

# create a new column and use np.select to assign values to it using our lists as arguments
comp['Result'] = np.select(conditions, values)

#Import xml files for DP and DJ 

st.title('ePCR viewer')

st.header('Select plate and dye from drop down left to change data view')
st.subheader('lower threshold set at mean - 3*SD - Mid point - Mean Average - upper set at Mean + 3*SD')

#comp=pd.read_csv(uploaded_file)



#percentiles
def Q25(x):
    return x.quantile(0.25)

def Q50(x):
    return x.quantile(0.5)

def Q75(x):
    return x.quantile(0.75)


def ROXCV(df1):
    stats_ROX = df1.groupby(['Run_ID'])['ROX_RFU'].agg(['count', 'mean','std','min',Q25, Q50, Q75, 'max'])
    print('-'*30)
    CI95_hi_ROX = []
    CI95_lo_ROX = []
    CV_run_ROX = []
    for i in stats_ROX.index:
        c,m,s,t,u,q1,q2,v =(stats_ROX.loc[i])
        CI95_hi_ROX.append(m + 1.95*s/math.sqrt(c))
        CI95_lo_ROX.append(m - 1.95*s/math.sqrt(c))
        CV_run_ROX.append(s/m*100)
        
    stats_ROX['CI95% low ROX'] = CI95_lo_ROX
    stats_ROX['CI95% hi ROX'] = CI95_hi_ROX
    stats_ROX['ROX CV%'] = CV_run_ROX
    stats_ROX = stats_ROX.reset_index()
    return(stats_ROX)

stats_ROX = ROXCV(comp)




stats_FAM = comp.groupby(['Run_ID','Result'])['FAM_RFU'].agg(['count', 'mean','std','min',Q25, Q50, Q75, 'max'])
#print(stats_FAM)
print('-'*30)

CI95_hi_FAM = []
CI95_lo_FAM = []
CV_ruFAM_RFU = []

for i in stats_FAM.index:
    c,m,s,t,u,q1,q2,v = round(stats_FAM.loc[i])
    CI95_hi_FAM.append(m + 1.95*s/math.sqrt(c))
    CI95_lo_FAM.append(m - 1.95*s/math.sqrt(c))
    #CV_ruFAM_RFU.append(100 -(s/m*100))

stats_FAM['ci95_lo_FAM'] = CI95_lo_FAM
stats_FAM['ci95_hi_FAM'] = CI95_hi_FAM
#stats_FAM['CV%_FAM'] = CV_ruFAM_RFU

print(stats_FAM)


print('-'*30)

print('nomralised_FAM_N1N2 by sample type')

stats_nFAM = comp.groupby(['Run_ID','Result'])['norm_N_Cov'].agg(['count', 'mean','min', 'std',Q25, Q50, Q75, 'max'])

print('-'*30)

CI95_hi_nFAM = []
CI95_lo_nFAM = []
CV_run_nFAM = []

for i in stats_nFAM.index:
    c,m,s,t,u,q1,q2,v =(stats_nFAM.loc[i])
    CI95_hi_nFAM.append(m + 1.95*s/math.sqrt(c))
    CI95_lo_nFAM.append(m - 1.95*s/math.sqrt(c))
    CV_run_nFAM.append(100 - (s/m*100))

stats_nFAM['CI 95% low nFAM'] = CI95_lo_nFAM
stats_nFAM['CI 95_hi_nFAM'] = CI95_hi_nFAM
stats_nFAM['CV%_nFAM'] = CV_run_nFAM
#stats_nFAM['%Percent_detected'] = result['N1N2_detected'] / TOT*100
print(stats_nFAM)


fig2b = px.scatter(comp, x= 'norm_RNaseP', y = 'norm_N_Cov',color = 'Result')

#fig2b.show()
#fig2b.write_html('comp_N3.html')


fig1bbnbb = px.scatter(comp, x= 'order', y = 'norm_RNaseP', color = 'Result')
fig1bbnbb.update_yaxes(range=[0, 4])
#fig1bbnbb.show()
#fig1bbnbb.write_html('comp_N3__monitor_normRNaseP.html')

figROX = px.scatter(comp, x= 'order', y = 'ROX_RFU', color = 'Result', title = 'Dispense Trace ROX')
figROX.update_yaxes(gridwidth = 0.0002, gridcolor ='grey')

figROX.add_trace(go.Scatter(
    x=[comp.order.min(), comp.order.max()],
    y=[1600, 1600],
    mode="lines",
    name="1600  RFU Lower Cutoff Limit",
    text=["LCL"],
    #text=["ROX 1600 lower cutoff"],
    textposition="top center",
    line=dict(color="grey")
))
#figrp.show()


figrp = px.scatter(comp, x= 'ROX_RFU', y = 'FAM_RFU' ,color = 'Result')
figrp.update_xaxes(range=[1000, 6000])
figrp.update_yaxes(range=[0, 50000])


figrp.add_trace(go.Scatter(
    x=[1600, 1600],
    y=[50000, -100],
    mode="lines",
    name="1600  RFU Lower Cutoff Limit",
    text=["LCL"],
    #text=["ROX 1600 lower cutoff"],
    textposition="top center",
    line=dict(color="grey")
))
#figrp.show()

print('CFO_RFU by sample type')

stats_CFO = comp.groupby(['Run_ID','Result'])['VIC_RFU'].agg(['count', 'mean','std','min',Q25, Q50, Q75, 'max'])

CI95_hi_CFO = []
CI95_lo_CFO = []
CV_run_CFO = []

for i in stats_CFO.index:
    c,m,s,t,u,q1,q2,v = round(stats_CFO.loc[i])
    CI95_hi_CFO.append(m + 1.95*s/math.sqrt(c))
    CI95_lo_CFO.append(m - 1.95*s/math.sqrt(c))
    CV_run_CFO.append(100 -(s/m*100))

stats_CFO['ci95_lo_CFO'] = CI95_lo_CFO
stats_CFO['ci95_hi_CFO'] = CI95_hi_CFO
stats_CFO['CV%_CFO'] = CV_run_CFO

print(stats_CFO)




print('-'*30)

print('normalised_CFO_RNaseP by sample type')

stats_nCFO = comp.groupby(['Run_ID','Result'])['norm_RNaseP'].agg(['count', 'mean','std','min',Q25, Q50, Q75, 'max'])
print(stats_nCFO)
print('-'*30)

CI95_hi_nCFO = []
CI95_lo_nCFO = []
CV_run_nCFO = []

for i in stats_nCFO.index:
    c,m,s,t,u,q1,q2,v = (stats_nCFO.loc[i])
    CI95_hi_nCFO.append(m + 1.95*s/math.sqrt(c))
    CI95_lo_nCFO.append(m - 1.95*s/math.sqrt(c))
    CV_run_nCFO.append((s/m*100))

stats_nCFO['ci95_lo_nCFO'] = CI95_lo_nCFO
stats_nCFO['ci95_hi_nCFO'] = CI95_hi_nCFO
stats_nCFO['CV%_nCFO'] = CV_run_nCFO

print(stats_nCFO)

def N_cov_process(df,t):
    figN1 = px.scatter(df, x= 'order', y = 'norm_N_Cov' ,color = 'Result', title = 'N1 N2 Calls ' + str(t))
    figN1.add_trace(go.Scatter(
        y=[10, 10],
        x=[df.order.min(), df.order.max()],
        mode="lines+markers+text",
        name="Lower_10_Positive_Boundary",
        text=["10"],
        #text=["ROX 1600 lower cutoff"],
        textposition="top center",
        line=dict(color="red")
    ))



    figN1.add_trace(go.Scatter(
         y=[9, 9],
         x=[df.order.min(), df.order.max()],
         mode="lines+markers+text",
         name="Lower_9_Positive_Boundary",
         text=["9"],
         textposition="top center"))



    figN1.update_xaxes(showgrid = True, gridwidth = 0.0002, gridcolor = 'grey')
    figN1.update_yaxes(range=[0, 15],showgrid = True, gridwidth = 0.0002, gridcolor = 'grey')
    st.plotly_chart(figN1, use_container_width=True)


def heat_map(df1, dye_choice, plate_choice):
    
    y = df1['Row_ID']
    x = df1['Col_ID']
    z = df1[dye_choice]
    mean = z.mean()
    sd = z.std()
    cv = sd/mean*100
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='magma'))
    fig.update_layout(title= ('HEATMAP: '+ str(plate_choice) + ' ' + str(dye_choice) +' mean ' + str(int(mean)) + ' stdev ' + str(int(sd))+ ' CV% ' +str(int(cv))),
                      xaxis_nticks=24,
                      yaxis_nticks = 16)
    fig.update_yaxes(autorange="reversed")
    fig.data[0].update(zmin = mean - 3*sd, zmid = mean, zmax = mean + 3* sd)
    
    st.plotly_chart(fig, use_container_width=True)

    
    


dye = ['ROX_RFU', 'FAM_RFU', 'VIC_RFU', 'norm_RNaseP', 'norm_N_Cov']


plate = comp['Run_ID'].unique()
plate_choice = st.sidebar.selectbox('Select plate to analyse:', plate)
dye_choice = st.sidebar.selectbox('Select dye to analyse:', dye)
select = comp[(comp.Run_ID == plate_choice)]
heat_map(select, dye_choice, plate_choice)
N_cov_process(select, plate_choice)

# Plot!

#selectable versions first - side by side - ROX FAM / Cartesian score plot then heat map then ROX dispense Plot
#as with heatmap - use selectables for either dye or plate 
#Below the selectable - show all processed data in the folder - find someway to make it non- changable
#after import - it will take a lot of time if lots of files. Assess the number of files and set 


col1, col2 = st.columns(2)

with col1:
	st.plotly_chart(figrp, use_container_width=True)
with col2:
	st.plotly_chart(fig2b, use_container_width=True)


st.plotly_chart(figROX,use_container_width = True)

st.plotly_chart(fig1bbnbb, use_container_width=True)
st.table(stats_ROX)

N_cov_process(comp,'')
st.table(stats_nFAM)
st.table(stats_nCFO)
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")