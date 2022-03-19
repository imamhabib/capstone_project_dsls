#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from numpy import int64
import seaborn as sns
import matplotlib.pyplot as plt


# In[34]:


dir = 'DJPK/'
data_program_2018 = pd.read_csv(dir+'Program_2018.csv', sep=';')
#data_program_2019 = pd.read_csv(dir+'Program_2019.csv', sep=';')
#data_program_2020 = pd.read_csv(dir+'Program_2020.csv', sep=';')
#data_program_2021 = pd.read_csv(dir+'Program_2021.csv', sep=';')
data_realisasi_2018 = pd.read_csv(dir+'Realisasi_2018.csv', sep=';')
#data_realisasi_2019 = pd.read_csv(dir+'Realisasi_2019.csv', sep=';')
#data_realisasi_2020 = pd.read_csv(dir+'Realisasi_2020.csv', sep=';')


# In[35]:


#menyamakan nama kolom
data_program_2018 = data_program_2018.rename({'Akun Analisis': 'AkunAnalisis'}, axis=1)

data_realisasi_2018 = data_realisasi_2018.rename({' nilaianggaran ': 'Nilaianggaran'}, axis=1)
data_realisasi_2018 = data_realisasi_2018.rename({'Akun Analisis': 'AkunAnalisis'}, axis=1)
#data_realisasi_2019 = data_realisasi_2019.rename({'nilaianggaran': 'Nilaianggaran'}, axis=1)
#data_realisasi_2020 = data_realisasi_2020.rename({'nilaianggaran': 'Nilaianggaran'}, axis=1)


# In[36]:


data_realisasi_2018['Tahunanggaran'] = 2018


# In[37]:


data_program_2018 = data_program_2018.dropna(axis=0, subset=['Nilaianggaran'])
data_realisasi_2018 = data_realisasi_2018.dropna(axis=0, subset=['Nilaianggaran'])


# In[38]:


kode_pemdax = data_program_2018.query('Namaprogram.isnull() & Kodepemda.str.len() >6',engine='python')[['Kodepemda','Namaprogram']]


# In[39]:


data_program_2018['Namaprogram'].fillna(kode_pemdax['Kodepemda'],inplace=True)


# In[40]:


kode_pemda_salah = data_program_2018.query('Kodepemda.notnull() & Kodepemda.str.len() <6',engine='python').drop_duplicates(subset=['Kodepemda'])[['Kodepemda','standarpemda']]


# In[41]:


#select unique kode pemda
kode_pemda = data_program_2018.query('Kodepemda.notnull() & Kodepemda.str.len() <6',engine='python').drop_duplicates(subset=['Kodepemda'])[['Kodepemda','standarpemda']]


# In[42]:


#select unique kode pemda
kode_pemda_realisasi = data_realisasi_2018.query('Kodepemda.notnull() & Kodepemda.str.len() <6',engine='python').drop_duplicates(subset=['Kodepemda'])[['Kodepemda','standarpemda']]


# In[43]:


join_kode_pemda = pd.merge(
    left=data_program_2018,
    right=kode_pemda,
    left_on='standarpemda',
    right_on='standarpemda',
    how='left'
)

join_kode_pemda_realisasi = pd.merge(
    left=data_realisasi_2018,
    right=kode_pemda_realisasi,
    left_on='standarpemda',
    right_on='standarpemda',
    how='left'
)


# In[44]:


data_program_2018['Kodepemda'] = join_kode_pemda['Kodepemda_y'].values
data_realisasi_2018['Kodepemda'] = join_kode_pemda_realisasi['Kodepemda_y'].values


# In[45]:


data_realisasi_2018 = data_realisasi_2018[data_realisasi_2018.standarkelompok != "All"]


# In[46]:


data_realisasi_2018 = data_realisasi_2018[data_realisasi_2018.standarkelompok.notnull()]


# In[47]:


data_program_2018_null = data_program_2018.query('AkunAnalisis.isnull()',engine='python')
data_realisasi_2018_null = data_realisasi_2018.query('AkunAnalisis.isnull()',engine='python')


# In[48]:


#select unique akun analisis program
kode_akun_analisis_program = data_program_2018.query('AkunAnalisis.notnull()',engine='python').drop_duplicates(subset=['AkunAnalisis','standarjenis','standarkelompok'])[['AkunAnalisis','standarjenis','standarkelompok']]

#select unique akun analisis realisasi
kode_akun_analisis_realisasi = data_realisasi_2018.query('AkunAnalisis.notnull()',engine='python').drop_duplicates(subset=['AkunAnalisis','standarjenis','standarkelompok'])[['AkunAnalisis','standarjenis','standarkelompok']]


# In[49]:


join_akun_analisis = pd.merge(
    left=data_program_2018,
    right=kode_akun_analisis_program,
    left_on='standarjenis',
    right_on='standarjenis',
    how='left'
)

join_akun_analisis_realisasi = pd.merge(
    left=data_realisasi_2018,
    right=kode_akun_analisis_realisasi,
    left_on='standarjenis',
    right_on='standarjenis',
    how='left'
)


# In[50]:


data_program_2018['AkunAnalisis'] = join_akun_analisis['AkunAnalisis_y'].values
data_realisasi_2018['AkunAnalisis'] = join_akun_analisis_realisasi['AkunAnalisis_y'].values


# In[51]:


#convert nilai_anggaran
data_program_2018['Nilaianggaran'].astype('int64')

#convert nilai_anggaran
data_realisasi_2018['Nilaianggaran'].astype('int64')


# In[52]:


data = {
    "JenisKelompok":['41. Pendapatan Asli Daerah','42. Dana Perimbangan','43. Lain-Lain Pendapatan Daerah yang Sah',
                     '51. Belanja Tidak Langsung','52. Belanja Langsung','61. Penerimaan Pembiayaan Daerah',
                    '62. Pengeluaran Pembiayaan Daerah'],
    "JenisAPBD":['Pendapatan','Pendapatan','Pendapatan','Belanja','Belanja','Pembiayaan','Pembiayaan']}
Jenis_APBD = pd.DataFrame(data=data)
Jenis_APBD


# In[53]:


join_jenis_apbd_program = pd.merge(
    left=data_program_2018,
    right=Jenis_APBD,
    left_on='standarkelompok',
    right_on='JenisKelompok',
    how='left'
)

join_jenis_apbd_realisasi = pd.merge(
    left=data_realisasi_2018,
    right=Jenis_APBD,
    left_on='standarkelompok',
    right_on='JenisKelompok',
    how='left'
)


# In[54]:


data_program_2018['JenisAPBD'] = join_jenis_apbd_program['JenisAPBD'].values
data_realisasi_2018['JenisAPBD'] = join_jenis_apbd_realisasi['JenisAPBD'].values


# In[55]:


data_program_2018.to_csv('data_clean_program_2018.csv',sep=";",index=False)
data_realisasi_2018.to_csv('data_clean_realisasi_2018.csv',sep=";",index=False)


# In[ ]:




