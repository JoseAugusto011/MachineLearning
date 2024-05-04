import xlrd
import csv 

dir = 'data/'

class ReadSeries:
    
    def read_income_csv(self, data, time):
        url = dir+'renda_Brasil_2012_2020.csv'
        with open(url, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=';')
            lc = 0;
            for row in rows:
                if lc == 1:
                    for r in row[1:] :
                        time.append(r) 
                if lc == 2:
                    for r in row[1:] :
                        data.append(int(r))
                lc += 1
    
    
    def read_energia_xls(self, data, time):
        # Abre o arquivo
        xls = xlrd.open_workbook(dir + 'ENERGIA.XLS')
        # Pega a primeira planilha do arquivo
        plan = xls.sheets()[0]
        # Para i de zero ao numero de linhas da planilha
        read = False
        t = 1
        for i in range(1, plan.nrows):
                # Le os valores nas linhas da planilha
                row = plan.row_values(i)
                if(row[0] == 1968.0):
                  read = True
                if(row[0] == 1979.0):
                  read = False
                if(read):
                  data.append(float(row[2]/1000))
                  time.append(t)
                  t +=1

    def read_IPI_xls(self, data, time):
        # Abre o arquivo
        xls = xlrd.open_workbook(dir + 'IPI.xls')
        # Pega a primeira planilha do arquivo
        plan = xls.sheets()[0]
        # Para i de zero ao numero de linhas da planilha
        read = False
        t = 1
        numNote = 0
        i=1
        while(i<plan.nrows):
                numNote -= 1
                # Le os valores nas linhas da planilha
                row = plan.row_values(i)
                if(row[0] == 1972.0):
                    numNote = 6
                    
          
                if(numNote == 0):
                  numNote += 1
                  data.append(float(row[1]))
                  time.append(t)
                  t += 1
                i += 1


    def read_temperatura_xls(self, data1, data2, time):
        # Abre o arquivo
        xls = xlrd.open_workbook(dir + 'temperatura.xls')
        # Pega a primeira planilha do arquivo
        plan = xls.sheets()[0]
       
        t = 1
        for i in range(1, plan.nrows):
                # Le os valores nas linhas da planilha
                row = plan.row_values(i)
                data1.append(float(row[1]))
                data2.append(float(row[2]))
                time.append(t)
                t +=1