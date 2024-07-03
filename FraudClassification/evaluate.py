import numpy as np
import pandas as pd
from plotnine import *
import locale
from scipy import stats #importo stats para implementar el ks


#Defino una nueva función para obtener las métricas

#Esta función recibe el dataframe con la probabilidad, el umbral para clasificar y el nombre del target

#Como resultado nos devuelve las métricas del modelo: recall, presision, ks y falsos positivos y
# metricas de negocio: % de rechazo, % de fraude prevenido, y una formula de balance en la que:
# el monto de los verdaderos positivos y falsos negativos suman 100%
# el monto de los falsos positivos suma el % de comision

def evaluate(df, umbral, target):
    
    #Calculo el TPV total de fraude
    
    tpv_fraude =  sum(df.loc[df['target']== 1,'Amount'])
    
    #Calculo el total de transacciones
    
    tx_tot = df.size
    
    #Guardo el dataframe en un nuevo objeto
    reclasif = df.copy()
    
     
    #reclasifico los casos con el nuevo umbral
    reclasif['new_predict'] = (
       reclasif
    .assign(new_predict=lambda x: pd.np.where(x['pred'] >= umbral, 1, 0))
    ['new_predict']
    )
    
    
    ####################################################
    ########        METRICAS DE IMPACTO         ########
    ####################################################
    
    print("------IMPACTO DEL MODELO--------")
    vp_amount_r = reclasif.query('target == 1 & new_predict == 1')['Amount'].sum()
    
    # Impacto de falsos positivos
    fp_amount_r = reclasif.query('target == 0 & new_predict == 1')['Amount'].sum() * 0.06
    
    # Impacto de falsos negativos
    fn_amount_r = reclasif.query('target == 1 & new_predict == 0')['Amount'].sum()
    
    # Impacto de verdaderos negativos
    vn_amount_r = reclasif.query('target == 0 & new_predict == 0')['Amount'].sum() * 0.06
    
    # Obtengo una fracción de balance 
    costos = (fp_amount_r + fn_amount_r)
    
    balance = (vp_amount_r) - costos
    indice =  costos / vp_amount_r
    
    
    #Armo una nueva matriz de confusión
    
    cm_2 = reclasif.groupby(['target', 'new_predict']).size().reset_index(name='n')
      
    
    #recalculo las métricas
    recall= round(cm_2.loc[3,'n']/sum(cm_2.loc[2:3,'n'])*100,2)
    falsos_positivos = round(cm_2.loc[1,'n']/sum(cm_2.loc[0:1,'n'])*100,2)
    precision = round(cm_2.loc[3, 'n'] * 100 / cm_2.loc[1:3, 'n'].sum(), 2)
    ks = round(stats.ks_2samp(reclasif.pred[reclasif.target == 0], reclasif.pred[reclasif.target == 1]).statistic, 2) * 100
    rechazo_tot = round((cm_2.loc[1, 'n'] + cm_2.loc[3, 'n'])*100/tx_tot,3)
    tpv_fraude_prev = round(vp_amount_r*100/(vp_amount_r+fn_amount_r),2)
    #Imprimo los valores
    
    # Imprimo los valores
    print("------------RECALL-------------")
    print(recall)
    print("-------FALSOS POSITIVOS--------")
    print(falsos_positivos)
    print("-----------PRECISION-----------")
    print(precision)
    print("--------------KS---------------")
    print(ks)
    print("-------TASA RECHAZO TOTAL------")
    print(rechazo_tot)
    print("-----TPV FRAUDE PREV. TOTAL----")
    print(tpv_fraude_prev)
    
    
    # Creo una función para darle formato a los valores de impacto
    import locale
    
    def comma_format(number):
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')  # Establece el local español
        return locale.format_string("%.2f", number, grouping=True)

    print(f"Ganancia Verdaderos Positivos {comma_format(vp_amount_r)} \n"
          f"Costo Falsos Positivos {comma_format(fp_amount_r)} \n"
          f"Costo Falsos Negativos: {comma_format(fn_amount_r)} \n"
          f"Ganancia Verdaderos Negativos: {comma_format(vn_amount_r)}\n"
          f"Balance: {comma_format(balance)}\n"
          f"Razon Costos/Ganancia: {comma_format(indice)}"
          )
    # Armo una matriz de confusion
    cm_2.iloc[:, 0] = cm_2.iloc[:, 0].astype('category')
    cm_2.iloc[:, 1] = cm_2.iloc[:, 1].astype('category')
    cm_2['n'] = cm_2['n'].astype('category')

    graf = (ggplot(cm_2, aes(x= cm_2.iloc[:, 0], y= cm_2.iloc[:, 1], fill='n')) +
            geom_tile(color="white") +
            geom_text(aes(label='n'), color="white", size=9) +
            labs(x="TARGET", y="PREDICT", title='Matriz de confusión') +
            scale_fill_manual(values=["#DA89FE", "#50007B", "#50007B", "#DA89FE"]) +
            theme(panel_grid=element_blank()) +
            coord_flip() +
            guides(fill=False))

    return [reclasif, graf, cm_2]

