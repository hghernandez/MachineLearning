
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
## Titulo

titanic <- titanic %>%
  mutate(EnFamilia = case_when(SibSp + Parch > 0 ~ 1,
                               TRUE ~ 0)) %>%
  select(-c(SibSp,Parch))

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

mat_cor <- dplyr::select(titanic_num,-Survived) %>% cor(method="pearson",use = "complete.obs") %>% round(digits=2) 

library(corrplot)

corrplot(mat_cor, type="upper", order="hclust", tl.col="black", tl.srt=45)

##%######################################################%##
#                                                          #
####          Analizamos la correlación entre           ####
####             las variables no numericas             ####
#                                                          #
##%######################################################%##

titanic_nom <- titanic %>% select_if(is.character)

crosstable::crosstable(titanic_nom,cols = Sex,by = Embarked,
                       funs=mean, test=TRUE)
