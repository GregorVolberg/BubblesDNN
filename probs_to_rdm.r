library(tidyverse)
df = read_csv('./probs_for_weights_HCvsNC.h5.csv') %>%
  rename('orig_trial' = '...1') %>%
  mutate(condition = as_factor(condition),
         condition  = fct_recode(condition, 'happy' = '1', 'neutral' = '2'),
         vp = as_factor(vp))


df %>% select(vp, condition, prob_happy) %>%
  aggregate(., by = list("condition", "vp"), FUN = "mean")

df2 = df %>% 
  group_by(vp, condition) %>%
  summarize(mean_phappy = mean(prob_happy),
            group = unique(group)) %>%
  ungroup()

plot(df2$mean_phappy[df2$group=="control"], type="l", col="blue", ylim = c(0.5, 1))
points(df2$mean_phappy[df2$group=="experimental"], type='l', col='red') # nolint





plot(df2
  