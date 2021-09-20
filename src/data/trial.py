import pandas as pd

tb = pd.read_csv('/home/desktop0/files/data_merge/prod/Resumen_Denuncia.csv')
tb = tb.tail(10)
tb.to_csv('/home/desktop0/files/data_merge/prod/Resumen_Denuncia.csv', index=False)