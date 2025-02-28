import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Read the signin information from the file
signin = pd.read_csv("signin.txt", header=None).iloc[:, 0].values

# Create the SQLAlchemy engine
engine = create_engine('postgresql+psycopg2://{user}:{password}@localhost:5432/{dbname}?host={host}'.format(
    user=signin[0],
    password=signin[1],
    dbname='mimiciv',
    host='/var/run/postgresql'
))

# Query the database using pandas, no need to manage connection
triage_table = pd.read_sql_query("SELECT * FROM mimiciv_ed.triage;", con=engine)
print(len(triage_table))
