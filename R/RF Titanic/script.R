
##%######################################################%##
#                                                          #
####       Objetivo: implementar un modelo Random       ####
####        Forest de clasificacion para Titanic        ####
#                                                          #
##%######################################################%##

#Cargo las librerias

library(tidyverse)
library(tidymodels)
library(dplyr)

#Cargo el dataset


ruta <- "C:/Users/hernan.hernandez/Documents/Documents/MachineLearning/python/Random Forest Clasificador/Data/"


titanic <- readr::read_csv(paste0(ruta,"titanic.csv"))

ruta_pc_personal <- "C:/Users/usuario/Documents/MachineLearning/python/Random Forest Clasificador/Data/" 

titanic <- readr::read_csv(paste0(ruta_pc_personal,"titanic.csv"))

#Analisis exploratorio

head(titanic)


titanic %>%
  group_by(Sex,Survived)%>%
  summarise(n= n()) %>%
  ggplot(aes(x= as.factor(Survived), y= n, fill= Sex)) +
  geom_bar(stat= "identity", position= "dodge")

titanic %>%
  group_by(Pclass,Survived)%>%
  summarise(n= n()) %>%
  ggplot(aes(x= as.factor(Survived), y= n, fill= as.factor(Pclass))) +
  geom_bar(stat= "identity", position= "dodge")

titanic %>%
  glimpse()


#Manejo de missing

sapply(titanic, function(x) sum(is.na(x)))
  

#Creo dos variables
## Viaja en fiamilia
## Cambio survived a character

titanic <- titanic %>%
  mutate(EnFamilia = case_when(SibSp + Parch > 0 ~ 1,
                               TRUE ~ 0)) %>%
  select(-c(SibSp,Parch)) %>%
  mutate(Survived = as.character(Survived))

##%######################################################%##
#                                                          #
####             Analizamos la correlación              ####
####             entre variables numéricas              ####
#                                                          #
##%######################################################%##

#Comprobamos colinearidad con una matriz de correlacion
titanic %>%
  glimpse()

titanic_num <- titanic %>% select_if(is.numeric)

mat_cor <- titanic_num %>% cor(method="pearson",use = "complete.obs") %>% round(digits=2) 

library(corrplot)

corrplot(mat_cor, type="upper", order="hclust", tl.col="black", tl.srt=45)

##%######################################################%##
#                                                          #
####          Analizamos la correlación entre           ####
####             las variables no numericas             ####
#                                                          #
##%######################################################%##

titanic_nom <- titanic %>% select_if(is.character)

options(scipen = 1000000)


survived_sex <- table(titanic_nom$Survived,titanic_nom$Sex)

chisq.test(survived_sex)

survived_emb <- table(titanic_nom$Survived,titanic_nom$Embarked)


chisq.test(survived_emb)


#Con un valor de p por debajo de 0.001 se rechaza la H0 de independencia
#por lo que tanto embarked como sexo estan relacionadas significativamente
# con survived

##%######################################################%##
#                                                          #
####                  Preprocesamiento                  ####
#                                                          #
##%######################################################%##


#Dividimos el dataset

#Dividimos el dataset ----


titanic <- titanic[,-c(4,7,8,9)]

titanic$Sex <- as.factor(titanic$Sex) #Para que funcione el impute mode
titanic$Embarked <- as.factor(titanic$Embarked) #Para que funcione el impute mode
titanic$EnFamilia <- as.factor(titanic$EnFamilia)

set.seed(123)

split <- initial_split(titanic,strata = Survived)

train <- training(split)

test <- testing(split)


train %>%
  glimpse()

#armo la receta
## inputo los NA de las variables numericas con el valor de la media
## inputo los NA de las variables categoricas con el valor más frecuente
## creo variables dummy para las categoricas

names(titanic)

sapply(titanic, function(x) sum(is.na(x)))

titanic_rec <- recipe(Survived ~ ., data = train) %>%
  update_role(PassengerId, new_role = "ID") %>%
  step_impute_mean(Age)%>% #Imputo los NA con la media de Edad
  step_impute_mode(Embarked) %>% #Imputo los Na con la categoria más frecuente
  step_dummy(all_nominal(), -all_outcomes())#One hot encoding todas las nominales


#Aprendo la receta

train_prep <- prep(titanic_rec)

juiced <- bake(train_prep,new_data = NULL)

View(juiced)


#Creo y ajusto el primer modelo

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

##%######################################################%##
#                                                          #
####                 Creamos un worflow                 ####
#                                                          #
##%######################################################%##

tune_titanic_wf <- workflow() %>%
  add_recipe(titanic_rec)%>%
  add_model(tune_spec)

##%######################################################%##
#                                                          #
####                Ajustamos el modelo                 ####
####         y optimizamos los hiperparametros          ####
#                                                          #
##%######################################################%##

##%######################################################%##
#                                                          #
####            Optimizacion hiperparámetros            ####
#                                                          #
##%######################################################%##

# Validacion cruzada ----

set.seed(234)

trees_folds <- vfold_cv(train)

# Entrenamos varios modelos ----

doParallel::registerDoParallel()

set.seed(345)

tune_res <- tune_grid(
  tune_titanic_wf,
  resamples = trees_folds,
  grid = 10
)


tune_res %>%
  collect_metrics() %>%
  filter(.metric== "roc_auc")%>%
  select(mtry,min_n,mean) %>%
  pivot_longer(-mean) %>%
  ggplot(aes(x= value, y= mean))+
  geom_line()+
  facet_wrap(.~name,scales = "free_x")

## Probamos varias combinaciones de hipeparametros

rf_grid <- grid_regular(
  mtry(range = c(1, 6)),
  min_n(range = c(2, 10)),
  levels = 5
)

rf_grid

##Reajustamos el modelo

set.seed(456)

tune_res_reg <- tune_grid(
  tune_titanic_wf,
  resamples = trees_folds,
  grid = rf_grid
)

# Vemos las métricas 

tune_res_reg %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

View(tune_res_reg %>%
       collect_metrics())

best_parameter <- tune_res_reg %>%
  select_best(metric = "roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_parameter
)

final_rf




#Ajustamos el modelo final

##Armamos el workflow


titanic_final <- workflow() %>%
  add_recipe(titanic_rec)%>%
  add_model(final_rf)


#Ajustamos el modelo



titanic_final_fit <- titanic_final %>%
  last_fit(split)

titanic_final_fit %>%
  collect_metrics()

#Unimos la prediccion con el archivo original


View(titanic_final_fit %>%
  collect_predictions() %>%
  mutate(correcto= case_when(Survived == .pred_class ~ 1,
                             TRUE ~ 0)) %>%
  select(-Survived)%>%
  bind_cols(test))

#Vemos las variables según la importancia

library(vip)

final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Survived ~ .,
      data = juice(train_prep) %>% select(-PassengerId)
  ) %>%
  vip(geom = "point")


