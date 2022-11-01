library(tidyverse)
library(tidymodels)
library(openxlsx)

#Objetivo predecir si compra o no según sexo, salario y edad

#Cargo el archivo

data <- read_csv("Social_Network_Ads.csv")


#Exploramos el archivo

#Compra según sexo
data %>%
  group_by(Purchased,Gender) %>%
  summarise(n= n_distinct(`User ID`))%>%
  mutate(p= round(n*100/sum(n),2))

#Compra según edad
View(data %>%
  group_by(Age) %>%
  summarise(n= n_distinct(`User ID`))%>%
  mutate(p= round(n*100/sum(n),2)))


View(data %>%
       mutate(gredad= case_when(Age <= 30~ "1.Menor 30",
                                TRUE ~ "2.Mayor 30"))%>%
       group_by(Purchased,gredad)%>%
       summarise(n= n_distinct(`User ID`))%>%
       mutate(p= round(n*100/sum(n),2)))

#Exploramos salario

between(10,9,11)

View(data %>%
       mutate(group_salary= case_when(between(EstimatedSalary,0,43000)== TRUE ~ "1.<43k",
                                between(EstimatedSalary,43000,70000)== TRUE ~ "2.>=43k y <70k",
                                between(EstimatedSalary,70000,88000)== TRUE ~ "3.>=43k y <88k",
                                TRUE ~ "4.>88k"))%>%
       group_by(Purchased,group_salary)%>%
       summarise(n= n_distinct(`User ID`))%>%
       mutate(p= round(n*100/sum(n),2)))

summary(data)

#Creamos el df final

data <- data %>%
  mutate(Purchased = as.factor(Purchased))%>%
  na.omit()

# Dividimos el dataset ----
set.seed(123)

split_data <- initial_split(data,strata = Purchased)

train <- training(split_data)
test <- testing(split_data)


table(train$Purchased)
table(test$Purchased)

#Creamos la receta ----

tree_rec <- recipe(Purchased ~ ., data = train) %>%
  update_role(`User ID`, new_role = "ID") %>%
  step_dummy(all_nominal(), -all_outcomes())#One hot encoding todas las nominales
  
# Proprocesamos los datos segun la receta

trees_prep <- prep(tree_rec)


#Obtenemos el df preprocesado

trees_juiced <- bake(trees_prep,new_data = NULL)

#Creamos el modelo ----

tune_spec <- rand_forest(mtry = tune(),
                           min_n = tune(),
                           trees = 1000)%>%
  set_mode("classification") %>%
  set_engine("ranger")

#Creamos un workflow ----

model_wf <- workflow() %>%
  add_recipe(tree_rec)%>%
  add_model(tune_spec)


#Hacemos la validacion cruzada

set.seed(234)

trees_folds <- vfold_cv(train)

#Entrenamos el modelo ----

doParallel::registerDoParallel()

set.seed(345)

tune_res <- tune_grid(
  model_wf,
  resamples = trees_folds,
  grid = 20
)

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean,mtry,min_n)%>%
   pivot_longer(mtry:min_n,
                names_to = "parametros",
                values_to = "value") %>%
  ggplot(aes(x= value, y= mean, color= parametros))+
  geom_point()+
  facet_wrap(~parametros, scales = "free_x")


#Armamos una cuadricula para optimizar

grid <- grid_regular(
  mtry(range=c(2,3)),
  min_n(range=c(2,8)),
  levels = 5
)

#levels es la cantidad de valores de min_n por cada valor de mtry

View(grid)

#Optimizamos los hiperparametros

set.seed(456)

regular_res <- tune_grid(
  model_wf,
  resamples = trees_folds,
  grid = grid
)


regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

regular_res$.metrics

#Elegimos el mejor modelo

#Siempre debemos seleccionar el modelo por roc_auc, ya que
#un modelo puede tener buena accuracy pero mala sensibilidad o especificidad

View(regular_res%>%
       collect_metrics())

best_model <- select_best(regular_res,metric = "accuracy")

#Creamos el modelo final

model_final <- finalize_model(
               tune_spec,
               best_model)


#Vemos la importancia de las variables en el modelo

library(vip)


model_final %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Purchased ~ .,
      data = juice(trees_prep) %>% select(-`User ID`)
  ) %>%
  vip(geom = "point")


##%######################################################%##
#                                                          #
####             Creamos el workflow final              ####
#                                                          #
##%######################################################%##

#Instanciamos el modelo productivo

model_prod <- workflow() %>%
  add_recipe(tree_rec)%>%
  add_model(model_final)


#Obtenemos las métricas del modelo productivo

eval <- model_prod %>%
  last_fit(split_data)

#Extraemos las metricas

eval %>%
  collect_metrics()


#unimos el dataframe original con las predicciones

df.final <- eval %>%
 collect_predictions()%>%
  mutate(correct = case_when(
  Purchased == .pred_class ~ "Correct",
  TRUE ~ "Incorrect"
))%>%
  bind_cols(test)


df.final %>% 
  group_by(Age,correct) %>%
  summarise(n= n_distinct(`User ID`))%>%
  ggplot(aes(x= Age, y= n, color= correct))+
  geom_point()


df.final %>% 
  select(EstimatedSalary,correct) %>%
  summarise(n= n_distinct(`User ID`))%>%
  ggplot(aes(x= EstimatedSalary, y= n, color= correct))+
  geom_point()
