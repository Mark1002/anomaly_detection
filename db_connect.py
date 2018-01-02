# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:17:34 2018

@author: mark
"""

import jaydebeapi
import pandas as pd

conn = jaydebeapi.connect('oracle.jdbc.driver.OracleDriver', 
                          'jdbc:oracle:thin:@test.iems.com.tw:12632:iemsdb',
                          {'user': "INS_RC_EMSQC", 'password': "Ji394su3"},
                          'ojdbc8.jar')

curs = conn.cursor()
curs.execute("select * from meterinfo2 where meterid=8395 and reporttime"
             ">= to_Date('2017-01-01 00:00:00','yyyy-mm-dd HH24:mi:ss')")
df = pd.DataFrame(curs.fetchall())
df.columns = [attr[0] for attr in curs.description[0:len(curs.description)]]
df.to_csv('CSV/light/light2017.csv', index=False)
curs.close()
conn.close()
