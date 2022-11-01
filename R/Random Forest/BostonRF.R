
library(tidyverse)
library(tidymodels)
library(lubridate)
library(MASS)

#Usamos el dataset boston de la libreria tree

glimpse(Boston)

# CRIM: tasa de criminalidad per cápita por ciudad
# ZN: proporción de suelo residencial zonificado para lotes de más de 25,000 pies cuadrados.
# INDUS: proporción de acres comerciales no minoristas por ciudad
# CHAS: variable ficticia del río Charles (= 1 si el tramo limita con el río; 0 en caso contrario)
# NOX: concentración de óxidos nítricos (partes por 10 millones)
# RM: promedio de cuartos por vivienda
# age: proporción de unidades ocupadas por sus propietarios construidas antes de 1940
# DIS: distancias ponderadas a cinco centros de empleo de Boston
# RAD: índice de accesibilidad a las carreteras radiales
# TAX: tasa de impuesto a la propiedad de valor total por $ 10,000
# PTRATIO: ratio alumno-profesor por municipio
# BLACK: 1000(Bk - 0.63)^2 donde Bk es la proporción de negros por ciudad
# LSTAT: % estado inferior de la población
# MEDV: valor medio de las viviendas ocupadas por sus propietarios en $ 1000

##%######################################################%##
#                                                          #
####           Objetivo predecir el precio de           ####
####            la vivienda con un modelo RF            ####
#                                                          #
##%######################################################%##

#Exploramos los datos
View(summarytools::dfSummary(Boston,style = "grid"))


#Vemos datos perdidos

missing <- data.frame(columna= "columna", missing= 0)

for(column in names(Boston)){

missing[column,2] <- sum(is.na(Boston[,column]))

}

rownames_to_column(missing)

#Comprobamos colinearidad con una matriz de correlacion

mat_cor <- dplyr::select(Boston,-medv) %>% cor(method="pearson") %>% round(digits=2) 



library(corrplot)

corrplot(mat_cor, type="upper", order="hclust", tl.col="black", tl.srt=45)

#Hay colinearidad entre rad y tax. Me quedo con rad

#No hay falta preprocesar los predictores en random forest


##%######################################################%##
#                                                          #
####                   Entrenamiento                    ####
#                                                          #
##%######################################################%##

#dividimos el dataset

splits <- initial_split(dplyr::select(Boston,-tax),strata = medv,prop = 0.8)

boston_train <- training(splits)
boston_test <- testing(splits)


summary(boston_train$medv)
summary(boston_test$medv)
#Creamos el modelo

ranger_spec <-
  rand_forest(mtry = tune(), min_n = tune(),trees = 1000) %>%
  set_engine('ranger') %>%
  set_mode('regression')

#Creamos el workflow de entrenamiento

model.fit <- workflow()%>%
  add_model(ranger_spec)%>%
  add_formula(medv ~ .)

#Creamos el set de validacion

set.seed(234)

trees_folds <- vfold_cv(boston_train)

#Ajustamos el modelo

doParallel::registerDoParallel()

set.seed(345)

tune_res <- tune_grid(
  model.fit,
  resamples = trees_folds,
  grid = 10
)

library(dplyr)

tune_res %>%
  collect_metrics()%>%
  filter(.metric== "rmse")%>%
  dplyr::select(mtry,min_n,mean) %>%
  pivot_longer(-mean,names_to = "parameters")%>%
  ggplot(aes(x= value, y= mean))+
  geom_point()+
  facet_wrap(~parameters,scales = "free_x")

#Armo un grid de optimizacion

grid <- grid_regular(
  min_n(range=c(2,10)),
  mtry(range=c(4,10)),
  levels = 5
)

grid 

# Reajusto el modelo

model_grid <- tune_grid(
  model.fit,
  resamples = trees_folds,
  grid = grid
)

#Muestro el mejor
model_grid %>%
  show_best(metric = "rmse")

#Selecciono el mejor modelo

best_model <- model_grid %>%
  select_best(metric = "rmse")

best_model

#Creamos el modelo final

boston_final <- finalize_model(x = ranger_spec,
                               parameters = best_model)

#Ajustamos con los datos de test

model_prod <- workflow()%>%
  add_model(boston_final)%>%
  add_formula(medv ~ .)


ult_eval <- model_prod %>%
  last_fit(splits)

#Miramos las metricas

ult_eval %>%
  collect_metrics()


#Vemos la importancia de las variables

library(vip)


boston_final %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(medv ~ .,boston_train) %>%
  vip(geom = "point")

