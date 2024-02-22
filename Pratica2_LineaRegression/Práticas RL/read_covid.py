import pandas as pd 
from collections import defaultdict
from datetime import (date,datetime)

dir = 'data/'

s_regiao = 'regiao'
s_estado = 'estado'
s_data = 'data'
s_casos_novos = 'casosNovos'
s_casos_acumulados = 'casosAcumulado'
s_obitos_novos = 'obitosNovos'
s_obitos_acumulados = 'obitosAcumulado'

class ReadCovid:
    tempo = 0
    casos_novos = 1
    casos_acumulados = 2
    obitos_novos = 3
    obitos_acumulados = 4
    
    def read_csv(self):
        csv = None
        url = dir + 'covid.csv' 
        
        csv = pd.read_csv(url, sep=';')
            
        # Pega o cabecalho do arquivo
        print(csv.head(0))

        d = date(2020, 1, 30)
        inc = 0
        BR = defaultdict(list)  
        for i in range(len(csv[s_regiao])):
            if (csv[s_estado][i] != csv[s_estado][i]): # Se igula a NaN dados do Brasil                
                #_date = datetime.strptime(csv[s_data][i], '%Y-%m-%d').date()
                #data[est][self.tempo].append(_date.strftime('%d/%m'))
                BR[self.tempo].append(inc)
                BR[self.casos_novos].append(csv[s_casos_novos][i])
                BR[self.casos_acumulados].append(csv[s_casos_acumulados][i])
                BR[self.obitos_novos].append(csv[s_obitos_novos][i])
                BR[self.obitos_acumulados].append(csv[s_obitos_acumulados][i])
                inc += 1
                
        return BR