## Import library
library(plyr)
library(dplyr)
library(arulesCBA)
library(knitr)
library(kableExtra)

## Import data
df <- read.csv("https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_yes_no.csv")

## Encode "Churn" as 0/1
df <- df %>%
    mutate(Churn = ifelse(Churn == "No",0,1)) %>%
    mutate(Churn = as.factor(Churn))

## Discretize "MonthlyCharges" with respect to "Churn"/"No Churn" label and assign to new column in dataframe
df$Binned_MonthlyCharges <- discretizeDF.supervised(Churn ~ ., df[, c('MonthlyCharges', 'Churn')], method='mdlp')$MonthlyCharges

## Rename the levels based on knowledge of min/max monthly charges
df$Binned_MonthlyCharges = revalue(df$Binned_MonthlyCharges,
                                    c("[-Inf,29.4)"="$0-29.4",
                                    "[29.4,56)"="$29.4-56",
                                    "[56,68.8)"="$56-68.8",
                                    "[68.8,107)"="$68.8-107",
                                    "[107, Inf]" = "$107-118.75"))

write.csv(df, './disc_churn.csv', row.names=FALSE)