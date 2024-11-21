data = read.csv("C:/Users/karaa/Downloads/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv")

library(dplyr)

# Recode 'yes' to 1 and 'no' to 0
data <- data %>%
  mutate(death_numeric = recode(death_yn, "Yes" = 1, "No" = 0))

grouped_data = data %>% 
  select(sex, death_numeric) %>% 
  mutate(sex = recode(sex, "Female" = 1, "Male" = 0))

chisq_table = table(grouped_data$sex, grouped_data$death_numeric)

# Are the death rates among female patients and male 
# patients significantly different?

# Ho: Death rates and patient sex are independent from each other.
# Ha: Death rates and patient sex have an association.

# p_value >= 0.05 -> Fail to reject null hypothesis
# p_value < 0.05 -> Reject the null hypothesis

chi_sq = chisq.test(chisq_table)

print(chi_sq)

# p_value = 8.892e-16 < 0.05. Therefore, we reject the null hypothesis,
# because there is evidence to support that death rates and patient sex
# have an association.