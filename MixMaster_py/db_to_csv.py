import pandas as pd
import sqlite3

subject = "003920"
dbcon = sqlite3.connect("daily_stock.db")
data = pd.read_sql("select * from '{}';".format(subject), dbcon)

with open(subject+".csv", "wt") as f:
    print(data.to_csv(index=False,line_terminator='\n'),file=f)