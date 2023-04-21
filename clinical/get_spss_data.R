library(tidyverse)
library(haven)

## run from directory [/loctmp2/data/vog20246/BubblesDNN/clinical] ## or setwd('X:/BubblesDNN/clinical')
protocolFile  <- '../img/dsetComposite/BubblesProtocolComposite2.txt'
spssFile      <- './Bubbles_SPSSDatei_30.03.2023.sav'

vplist <- read.csv(protocolFile, sep = '\t', header = F) %>%
  as_tibble() %>%
  select(vp = V2, group = V5) %>%
  mutate(vp = str_replace(vp, 'S', '0')) %>% # convert to three digit code
  unique() %>%
  arrange(vp) 

spss  <- read_sav(spssFile, encoding = 'latin1') %>%
  select(vp=VPNummer, age=Alter, groupSPSS=Kontrollgruppe_vs_NSSI, iq=IQPunkte, hand=`HÃ¤ndigkeit`)

joined = left_join(vplist, spss, by = 'vp')

