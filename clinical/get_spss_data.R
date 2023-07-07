library(tidyverse)
library(haven)

## run from directory [/loctmp2/data/vog20246/BubblesDNN/clinical] ## or setwd('X:/BubblesDNN/clinical')
protocolFile  <- '../img/dsetComposite/BubblesProtocolComposite2.txt'
spssFile      <- 'Bubble_AQC_BDI-II.sav' #./Bubbles_SPSSDatei_30.03.2023.sav'

# vplist <- read.csv(protocolFile, sep = '\t', header = F) %>%
#   as_tibble() %>%
#   select(vp = V2, group = V5) %>%
#   mutate(vp = str_replace(vp, 'S', '0')) %>% # convert to three digit code
#   unique() %>%
#   arrange(vp) 

df  <- read_sav(spssFile, encoding = 'latin1') %>%
  select(vp=VPNummer, group = Kontrollgruppe_vs_NSSI, age=Alter2, BDI = BDI2_sum, AQC_DIF, AQC_DDF, AQC_sum) %>%
  mutate(group = as_factor(group))

sumscore <- read_csv('../sumscore.csv', col_names = 'sumscore')
df %>% mutate(sumscore = sumscore$sumscore)


write_csv(df, 'clinicalcsv.csv')

n = 42
r = .305
t = r / (sqrt((1-r^2)/(n-2)))

df2 <- df %>%
  filter(group == 'NSSI')
plot(sumscore~BDI,data = df2)
plot(sumscore~AQC_DIF,data = df2)
plot(sumscore~AQC_DDF,data = df2)
plot(sumscore~AQC_sum,data = df2)

cor.test(df$sumscore,df$BDI)
cor.test(df$sumscore,df$AQC_DDF)
cor.test(df$sumscore,df$AQC_DIF)
cor.test(df$sumscore,df$AQC_sum)




ggplot(df, aes(x = group,
                y = BDI,
                color = group)) +
  geom_jitter(position = position_jitterdodge(0.4), size = 3) +
  stat_summary(size = 3, fun = 'mean',   col='red', shape = 95)+
  scale_color_manual(values = c("grey40", "goldenrod1")) +
  theme_classic() + 
  ylim(0, 60) +
  theme(legend.position="none")
  labs(x = 'Groups', y = 'BDI Score')

t.test(BDI~group, data=df)
aggregate(df$BDI, by = list(df$group), FUN = mean, na.rm=T)


ggplot(df, aes(x = group,
               y = AQC_DIF,
               color = group)) +
  geom_jitter(position = position_jitterdodge(0.4), size = 3) +
  stat_summary(size = 3, fun = 'mean',   col='red', shape = 95)+
  scale_color_manual(values = c("grey40", "goldenrod1")) +
  theme_classic() + 
  ylim(0, 2) +
  labs(x = 'Groups', y = 'AQC_DIF')

t.test(AQC_DIF~group, data=df)
aggregate(df$AQC_DIF, by = list(df$group), FUN = mean, na.rm=T)


ggplot(df, aes(x = group,
               y = AQC_DDF,
               color = group)) +
  geom_jitter(position = position_jitterdodge(0.4), size = 3) +
  stat_summary(size = 3, fun = 'mean',   col='red', shape = 95)+
  scale_color_manual(values = c("grey40", "goldenrod1")) +
  theme_classic() + 
  ylim(0, 2) +
  labs(x = 'Groups', y = 'AQC_DDF')


t.test(AQC_DDF~group, data=df)
aggregate(df$AQC_DDF, by = list(df$group), FUN = mean, na.rm=T)


ggplot(df, aes(x = group,
               y = AQC_sum,
               color = group)) +
  geom_jitter(position = position_jitterdodge(0.4), size = 3) +
  stat_summary(size = 3, fun = 'mean',   col='red', shape = 95)+
  scale_color_manual(values = c("grey40", "goldenrod1")) +
  theme_classic() + 
  theme(legend.position="none")
  ylim(0, 2) +
  labs(x = 'Groups', y = 'AQC_sum')



t.test(AQC_sum~group, data=df)
aggregate(df$AQC_sum, by = list(df$group), FUN = mean, na.rm=T)

t.test(iq~group, data=df)
aggregate(df$iq, by = list(df$group), FUN = mean, na.rm=T)

t.test(age~group, data=df)
aggregate(df$age, by = list(df$group), FUN = mean, na.rm=T)
