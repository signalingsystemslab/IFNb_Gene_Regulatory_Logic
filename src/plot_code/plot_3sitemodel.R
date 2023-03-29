library(tidyverse)
library(RColorBrewer)
setwd("~/IFN_paper/src/3-site-model")

# Load data
rmsd_df = read_csv("../data/ModelB1_rmsd.csv",col_names = c("rmsd","dset"))
params_df = read_csv("../data/ModelB1_parameters.csv",col_names = c("t1","t2","t3","t4","t5","t6","dset")) %>%
  pivot_longer(c("t1","t2","t3","t4","t5","t6"), names_to = "parameter",values_to = "val")

par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
            "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
            expression("t6: IRF/NF"*kappa*"B"))
# aic_df = read_csv("../data/ModelB1_aic.csv",col_names = c("aic","dset"))

eval_df = rmsd_df %>% mutate(aic = log(rmsd^2)*7+2*6) %>%
  pivot_longer(c("rmsd","aic"), names_to = "measurement",values_to = "val")

dummy=bind_cols(dset=c(1,2,3,4),measurement=c("aic","aic","rmsd","rmsd"),val=c(-35,-10,0,0.2))

p=ggplot(eval_df,aes(x=factor(0),y=val)) +
  geom_boxplot(outlier.size = 3) +
  # geom_jitter(data=eval_df[eval_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=eval_df[eval_df$dset==0,], color="red",size=3)+
  facet_wrap(~measurement, scales = "free_y") +
  geom_blank(data=dummy) + 
  theme_bw() +
  ggtitle(expression("Model "*beta*"1 fits for each dataset")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.title.y=element_blank(),
        axis.ticks.x=element_blank())
ggsave(p, filename = str_c("../3-site-model/figs/B1_all_data_fits.png"),
       height = 6, width = 9)

# val_names = c(`t1`="t1: IRF", `t2` = "t2: IRF", `t3` = expression("t3: NF"*kappa*"B"),
#               `t4`="t4: IRF/IRF", `t5` = expression("t5: IRF/NF"*kappa*"B"), `t6` = expression("t6: IRF/NF"*kappa*"B"))

p=ggplot(params_df,aes(x=parameter,y=val)) +
  geom_boxplot(outlier.size = 3) +
  # geom_jitter(data=eval_df[eval_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=params_df[params_df$dset==0,], color="red",size=3)+
  # facet_wrap(~measurement, scales = "free_y") +
  # geom_blank(data=dummy) + 
  theme_bw() +
  scale_x_discrete(labels=par_names) +
  ggtitle(expression("Model "*beta*"1 parameters from best fit model for each data set"))

ggsave(p, filename = str_c("../3-site-model/figs/B1_all_parameters.png"),
       height = 6, width = 9)


all_df = read_csv("../data/ModelB1_parameters.csv",col_names = c("t1","t2","t3","t4","t5","t6","dset")) %>%
  full_join(rmsd_df,by="dset") %>% mutate(aic = log(rmsd^2)*7+2*6) %>%
  mutate(good = aic<quantile(all_df$aic,0.25),
         bad = aic>quantile(all_df$aic,0.75),
         keep=good+bad) %>%
  filter(keep==1) %>%
  pivot_longer(c("t1","t2","t3","t4","t5","t6"), names_to = "parameter",values_to = "val")

# all_df[all_df$good+all_df$bad,]

p=ggplot(all_df,aes(x=good,y=val)) +
  geom_boxplot(outlier.size = 3) +
  # geom_jitter(data=eval_df[eval_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  # geom_point(data=all_df[all_df$dset==0,], color="red",size=3)+
  facet_wrap(~parameter) +
  # geom_blank(data=dummy) + 
  theme_bw() +
  scale_x_discrete(labels=c("worst","best")) +
  theme(axis.title.x=element_blank())+
  ggtitle(expression("Model "*beta*"1 parameters from best fit model for best/worst datasets"))

ggsave(p, filename = str_c("../3-site-model/figs/B1_best_worst_parameters.png"),
       height = 6, width = 9)

# TODO: Plot residuals for each parameter like Frank's fig 2C