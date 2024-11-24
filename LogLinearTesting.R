library(dplyr)

data = read.csv("C:/Users/karaa/Downloads/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv")

# Lets look at the association between Race and Hospitalization
# using a Log-Linear Model

table(data$race)
df = table(data$race, data$hosp_yn)        
df = as.data.frame(df)
colnames(df) = c("Race", "Hospitalized", "Count")
print(df)

# loglinear WITH interaction terms

loglinear_model = glm(Count ~ Race * Hospitalized, 
                      family = poisson(),
                      data = df)

summary(loglinear_model)

#loglinear WITHOUT interaction terms

simple_loglinear = glm(Count ~ Race + Hospitalized,
                       family = poisson(),
                       data = df)

summary(simple_loglinear)

# compare the two models fit
AIC(loglinear_model, simple_loglinear)

deviance(loglinear_model)
df.residual(loglinear_model)

rel_freq_table = sapply(data, function(x) table(x)/nrow(data))
