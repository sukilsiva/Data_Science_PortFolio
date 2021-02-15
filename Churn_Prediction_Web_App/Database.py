import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://root:idntknowpassword#404@localhost:3333/churnapp')

df = pd.read_sql_table("data/customerdata", engine)
df

data = pd.read_csv('userdata.csv')
data.head()
 
data.to_sql(name="customerdata", con=engine, index=False, if_exists='replace')
