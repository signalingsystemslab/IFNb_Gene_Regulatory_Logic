library(tidyverse)
library(RColorBrewer)
library(ggforce)
setwd("~/IFN_paper/src/3-site-model")

args=commandArgs(TRUE)
model_name=args[1] # Needs to include 1,2,3, or 4
model_name=str_extract(model_name,"\\d")

# get parameter names from model number
t_pars = c("t1","t2","t3","t4","t5","t6")
par_names=c("t1: IRF","t2: IRF",expression("t3: NF"*kappa*"B"),
            "t4: IRF/IRF",expression("t5: IRF/NF"*kappa*"B"),
            expression("t6: IRF/NF"*kappa*"B"))

if (model_name=="1") {
  param_names = t_pars

} else if (model_name=="2") {
  param_names = c("KI2",t_pars)

} else if (model_name=="3") {
  param_names = c("C",t_pars)

} else if (model_name=="4") {
  param_names = c("KI2","C",t_pars)

}
nparams = length(param_names)

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

eval_df = rmsd_df %>% mutate(aic = 7*log(rmsd^2)+2*nparams) %>%
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

# labels messed up
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


# only t parameters
params_df1 = params_df %>%
  filter(parameter %in% t_pars)

p=ggplot(params_df1,aes(x=parameter,y=val)) +
  geom_boxplot(outlier.size = 3) +
  # geom_jitter(data=eval_df[eval_df$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=params_df1[params_df1$dset==0,], color="red",size=3)+
  # ggforce::facet_row(~par_type, scales = "free", space = "free") +
  # geom_blank(data=dummy) + 
  theme_bw() +
  scale_x_discrete(labels=par_names) +
  ggtitle(bquote("Model "*beta*.(model_name)*" transcription parameters from best fit model for each data set"))

ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_t_parameters_boxplot.png"),
       height = 5, width = 6)

# only non t-parameters
if (model_name=="2") {
  params_df2 = params_df %>%
    filter(par_type=="Other") %>%
    mutate(log_par = log10(val))
  
  p=ggplot(params_df2,aes(x=parameter,y=log_par)) +
    # geom_boxplot(outlier.size = 3) +
    geom_jitter(data=params_df2[params_df2$dset!=0,], color="black",size=3,width=rel(0.1))+
    geom_point(data=params_df2[params_df2$dset==0,], color="red",size=3)+
    # ggforce::facet_row(~par_type, scales = "free", space = "free") +
    expand_limits(y=c(2,-7)) +
    theme_bw() +
    scale_x_discrete(labels=c(expression(K[I2]/K[I1]))) +
    ggtitle(bquote("Model "*beta*.(model_name)*" other parameters"))
  
  ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_other_parameters_boxplot.png"),
         height = 6, width = 3)
} else if (model_name=="3"){
  params_df2 = params_df %>%
    filter(par_type=="Other")
  
  p=ggplot(params_df2,aes(x=parameter,y=val)) +
    # geom_boxplot(outlier.size = 3) +
    geom_jitter(data=params_df2[params_df2$dset!=0,], color="black",size=3,width=rel(0.1))+
    geom_point(data=params_df2[params_df2$dset==0,], color="red",size=3)+
    # ggforce::facet_row(~par_type, scales = "free", space = "free") +
    # expand_limits(y=c(2,-7)) +
    theme_bw() +
    scale_x_discrete(labels=c("C")) +
    ggtitle(bquote("Model "*beta*.(model_name)*" other parameters"))
  
  ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_other_parameters_boxplot.png"),
         height = 6, width = 3)
} else if (model_name=="4"){
  params_df2 = params_df %>%
    filter(par_type=="Other")  %>%
    mutate(log_par = log10(val),
           parameter = factor(parameter, levels = c("KI2","C")))
  
  p=ggplot(params_df2,aes(x=parameter,y=log_par)) +
    # geom_boxplot(outlier.size = 3) +
    geom_line(aes(group=dset), alpha = 0.5, color="grey") +
    geom_point(data=params_df2[params_df2$dset!=0,], color="black",size=3)+
    geom_point(data=params_df2[params_df2$dset==0,], color="red",size=3)+
    expand_limits(y=c(2,-7)) +
    theme_bw() +
    # scale_x_continuous(labels=parameter) +
    # scale_x_discrete(labels=c(expression(K[I2]/K[I1]),"C")) +
    ggtitle(bquote("Model "*beta*.(model_name)*" other parameters"))
  
  ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_other_parameters_boxplot.png"),
         height = 6, width = 3)
  
  params_df3 = params_df %>%
    filter(par_type=="Other")  %>%
    mutate(log_val = log10(val)) %>%
    select(!val) %>%
    pivot_wider(names_from = parameter, values_from = log_val)
  
  p=ggplot(params_df3,aes(x=KI2,y=C)) +
    # geom_boxplot(outlier.size = 3) +
    # geom_line(aes(group=dset), alpha = 0.5, color="grey") +
    geom_point(data=params_df3[params_df3$dset!=0,], color="black",size=3)+
    geom_point(data=params_df3[params_df3$dset==0,], color="red",size=3)+
    theme_bw() +
    # scale_x_continuous(labels=parameter) +
    # scale_x_discrete(labels=c(expression(K[I2]/K[I1]),"C")) +
    ggtitle(bquote("Model "*beta*.(model_name)*" other parameters, log 10"))
  
  ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_other_parameters_scatter.png"),
         height = 6, width = 6)
  
}


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


par_df = read_csv(str_c("../data/ModelB",model_name,"_parameters.csv"),col_names = c(param_names,"dset")) %>%
  full_join(rmsd_df,by="dset") %>% mutate(aic = 7*log(rmsd^2)+2*nparams)
all_df=par_df %>%
  mutate(good = aic<quantile(par_df$aic,0.25),
         bad = aic>quantile(par_df$aic,0.75),
         keep=good+bad) %>%
  filter(keep==1) %>%
  pivot_longer(all_of(t_pars), names_to = "parameter",values_to = "val")

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

# Parameter box plots
t_params = par_df %>%
  select(all_of(t_pars),dset) %>%
  pivot_longer(all_of(t_pars), values_to = "val", names_to = "param") %>%
  group_by(param) %>%
  summarize(mean = mean(val), err = sd(val)/sqrt(n())) %>%
  mutate(lower = pmax(mean-err,0),
         upper=pmin(mean+err,1))

p=ggplot(t_params, aes(x=param, y=mean)) +
  geom_col()+
  geom_errorbar(aes(ymin=lower,ymax=upper))+
  theme_bw() +
  expand_limits(y=1) +
  scale_x_discrete(labels=par_names) +
  ylab("Relative transcription activity") +
  ggtitle(bquote("Model "*beta*.(model_name)))
ggsave(p, filename = str_c("../3-site-model/figs/B",model_name,"_t_parameters_barplot.png"),
       height = 5, width = 5)

