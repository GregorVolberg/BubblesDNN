library(tidyverse)
library(ez)
library(ggplot2)
library(viridis) # for plot
library(hrbrthemes) # for plot

df_happy <- bind_rows(read_csv('./results/probs_for_res50c_happy.csv'),
                      read_csv('./results/probs_for_res50e_happy.csv'),
                      .id = 'model') %>%
            rename('orig_trial' = '...1') %>%
            mutate(condition  = fct_recode(as_factor(condition),
                                           'emotion' = '1', 'neutral' = '3'),
                   model     = fct_recode(as_factor(model),
                                          'res50c' = '1',
                                          'res50e' = '2'),
                   vp        = as_factor(vp),
                   group     = as_factor(group),
                   p         = ifelse(condition == 'emotion', prob_happy, 1-prob_happy),
                   emotion   = 'happy') %>%
                   select(-prob_happy)

df_sad <- bind_rows(read_csv('./results/probs_for_res50c_sad.csv'),
                      read_csv('./results/probs_for_res50e_sad.csv'),
                      .id = 'model') %>%
  rename('orig_trial' = '...1') %>%
  mutate(condition  = fct_recode(as_factor(condition),
                                 'emotion' = '5', 'neutral' = '7'),
         model     = fct_recode(as_factor(model),
                                'res50c' = '1',
                                'res50e' = '2'),
         vp        = as_factor(vp),
         group     = as_factor(group),
         p         = ifelse(condition == 'emotion', prob_happy, 1-prob_happy),
         emotion   = 'sad') %>%
         select(-prob_happy)

df_all = rbind(df_happy, df_sad)

ezANOVA(data = df_all, wid = vp, dv = p, 
        within = .(model, condition, emotion), between = group)
stats = ezStats(data = df_all, wid = vp, dv = p, 
                within = .(model, condition, emotion))

# https://r-graph-gallery.com/48-grouped-barplot-with-ggplot2
# Graph
stats %>%
  mutate(model = fct_recode(model, "Trained on controls" = "res50c",
                            "Trained on patients" = "res50e")) %>%
  ggplot(., aes(fill=condition, y=Mean, x=emotion)) + 
  geom_bar(position="dodge", stat="identity") +
  scale_fill_viridis(discrete = T, option = "E") +
  ggtitle("Model accuracy") +
  facet_wrap(~model) +
  theme_ipsum() +
  theme(legend.position="top") +
  theme(legend.title=element_blank()) + 
  labs(y= "Mean p for target class",
       x = "Condition")
