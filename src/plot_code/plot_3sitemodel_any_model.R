library(tidyverse)
library(RColorBrewer)
library(ggforce)
setwd("~/IFN_paper/src/3-site-model")

args=commandArgs(TRUE)
model_name=args[1] # Needs to include 1,2,3, or 4
model_name=str_extract(model_name,"\\d")

# get parameter names from model number
t_pars = c("t1","t2","t3","t4","t5","t6")

if (model_name=="1") {
  param_names = t_pars
  par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
              "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
              expression("t6: IRF/NF"*kappa*"B"))
} else if (model_name=="2") {
  param_names = c("KI2",t_pars)
  par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
              "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
              expression("t6: IRF/NF"*kappa*"B"))
} else if (model_name=="3") {
  param_names = c("C",t_pars)
  par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
              "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
              expression("t6: IRF/NF"*kappa*"B"))
} else if (model_name=="4") {
  param_names = c("KI2","C",t_pars)
  par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
              "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
              expression("t6: IRF/NF"*kappa*"B"))
}

# Load data
rmsd_df = read_csv(str_c("../data/ModelB",model_name,"_rmsd.csv"),col_names = c("rmsd","dset"))
params_df = read_csv(str_c("../data/ModelB",model_name,"_parameters.csv"),col_names = c(param_names,"dset")) %>%
  pivot_longer(param_names, names_to = "parameter",values_to = "val") %>%
  mutate(par_type = case_when(
    parameter %in% t_pars ~ "t_parameter",
    TRUE ~ "Other"))
resid_df = read_csv(str_c("../data/ModelB",model_name,"_res.csv"),col_names = c("pt1","pt2","pt3","pt4","pt5","pt6","pt7","dset")) %>%
  pivot_longer(c("pt1","pt2","pt3","pt4","pt5","pt6","pt7"), names_to = "data_point",values_to = "residual")

#  Verify residual and RMSD agree
x= resid_df %>% mutate(rsq = residual^2) %>% group_by(dset) %>% summarize(rss = sum(rsq)) %>% mutate(rmsd = sqrt(rss/7))
cat("Does RMSD agree with residual df? \t")
cat(all(abs(x$rmsd-rmsd_df$rmsd)<0.00001))
cat("\n\n")

#  Continue analysis

dpt_names=c("WT_LPS",expression("NF"*kappa*"B KO LPS"),
            "IRF 2KO LPS","WT pIC",expression("NF"*kappa*"B KO pIC"),
            "IRF 2KO pIC", "IRF 3KO pIC")

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
  ggtitle(bquote("Model "*beta*.(model_name)*" fits for each dataset")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.title.y=element_blank(),
        axis.ticks.x=element_blank())
ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_all_data_fits.png"),
       height = 6, width = 9)

# val_names = c(`t1`="t1: IRF", `t2` = "t2: IRF", `t3` = expression("t3: NF"*kappa*"B"),
#               `t4`="t4: IRF/IRF", `t5` = expression("t5: IRF/NF"*kappa*"B"), `t6` = expression("t6: IRF/NF"*kappa*"B"))

p=ggplot(params_df,aes(x=parameter,y=val)) +
  geom_boxplot(outlier.size = 3) +
  # geom_jitter(data=eval_df[eval_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=params_df[params_df$dset==0,], color="red",size=3)+
  ggforce::facet_row(~par_type, scales = "free", space = "free") +
  # geom_blank(data=dummy) + 
  theme_bw() +
  scale_x_discrete(labels=par_names) +
  ggtitle(bquote("Model "*beta*.(model_name)*" parameters from best fit model for each data set"))

ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_all_parameters.png"),
       height = 6, width = 9)

p=ggplot(resid_df,aes(x=data_point,y=residual)) +
  geom_boxplot(outlier.size = 3) +
  geom_jitter(data=resid_df[resid_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=resid_df[resid_df$dset==0,], color="red",size=3)+
  # facet_wrap(~measurement, scales = "free_y") +
  expand_limits(y =c(-.3,.4)) +
  theme_bw() +
  scale_x_discrete(labels=dpt_names) +
  ggtitle(bquote("Model "*beta*.(model_name)*" residuals from best fit model for each data set"))

ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_all_residuals.png"),
       height = 6, width = 9)


all_df = read_csv(str_c("../data/ModelB",model_name,"_parameters.csv"),col_names = c(param_names,"dset")) %>%
  full_join(rmsd_df,by="dset") %>% mutate(aic = log(rmsd^2)*7+2*6)
all_df=all_df %>%
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
  ggtitle(bquote("Model "*beta*.(model_name)*" parameters from best fit model for best/worst datasets"))

ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_best_worst_parameters.png"),
       height = 6, width = 9)
