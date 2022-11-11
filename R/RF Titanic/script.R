
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

