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



# remuestreo por bootstrap para evaluar el modelo


boot <- bootstraps(train)
boot


#Especificamos el modelo

glm_spec <- logistic_reg()%>%
  set_engine("glm")


#Instanciamos el modelo

rl_model <- workflow()%>%
  add_formula(Purchased ~.)

#Ajusto el modelo

glm.fit <- rl_model %>%
  add_model(glm_spec)%>%
  fit_resamples(
    resamples= boot,
    control = control_resamples(save_pred = TRUE)
  )


glm.fit

#Analizamos las métricas de error

glm.fit %>%
  collect_metrics()


#Matriz de confusión

glm.fit %>%
  conf_mat_resampled()


#Creamos la curva ROC

View(glm.fit %>%
       collect_predictions())

glm.fit %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Purchased, .pred_0) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal()



#Ajuste final con los datos de test

glm_final <- rl_model  %>%
  add_model(glm_spec) %>%
  last_fit(split_data)

#Vemos las metricas de error

glm_final %>%
  collect_metrics()


#vemos las predicciones

View(glm_final %>%
  collect_predictions())

#vemos la matriz de confusion

glm_final %>%
  collect_predictions()%>%
  conf_mat(Purchased, .pred_class)

#Obtengo los coieficientes 

glm_final$.workflow[[1]] %>%
  tidy(exponentiate = TRUE)


library(palmerpenguins)

View(penguins)
