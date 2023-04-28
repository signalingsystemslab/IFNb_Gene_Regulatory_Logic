# Comparing the different versions of the 3-site model
library(tidyverse)
library(RColorBrewer)
library(ggforce)
setwd("~/IFN_paper/src/3-site-model")

#### Residuals ####
pt_names=c("pt1","pt2","pt3","pt4","pt5","pt6","pt7")

# Plotting residuals like in Frank's figure
read_residuals = function(number){
  res = read_csv(str_c("../data/ModelB",number,"_res.csv"),col_names = c(pt_names,"dset"),show_col_types = FALSE)
  res = bind_cols(res,"model"=number)
  return(res)
}

res = bind_rows(read_residuals("1"),read_residuals("2"),read_residuals("3"),read_residuals("4"))
res_long = res %>%
  pivot_longer(cols=all_of(pt_names),names_to = "data_point", values_to = "residual")

p=ggplot(res_long, aes(x=model,y=residual)) +
  geom_boxplot(outlier.color = "grey") +
  expand_limits(y =c(-.3,.4)) +
  facet_wrap(~data_point,nrow = 1) +
  geom_point(data=res_long[res_long$dset==0,]) +
  theme_bw()
ggsave(p, filename = str_c("../3-site-model/figs/all_models_all_residuals.png"),
       height = 3, width = 6)

# Least residual
res_long_max= res_long %>%
  group_by(dset,model) %>%
  filter(abs(residual)==max(abs(residual)))

jitterer = position_jitter(seed = 5,width=0.2) 
p=ggplot(res_long_max, aes(x=model,y=residual^2)) +
  geom_boxplot(outlier.shape = NA, fill="grey") +
  geom_point(data=res_long_max[res_long_max$dset!=0,],position = jitterer, aes(color=data_point),size=2) +
  geom_point(data=res_long_max[res_long_max$dset==0,], aes(color=data_point),size=2) +
  scale_color_brewer(palette="Purples")+
  geom_point(data=res_long_max[res_long_max$dset==0,], shape=1,
             size=2,color="red") +
  ggtitle("All models comparing worst fit point for each dataset") +
  theme_bw()

ggsave(p, filename = str_c("../3-site-model/figs/all_models_max_residuals.png"),
       height = 3, width = 4)

#### RMSD plot ####
rmsd_1 = read_csv(str_c("../data/ModelB","1","_rmsd.csv"),col_names = c("rmsd1","dset"))
rmsd_2 = read_csv(str_c("../data/ModelB","2","_rmsd.csv"),col_names = c("rmsd2","dset"))
rmsd_3 = read_csv(str_c("../data/ModelB","3","_rmsd.csv"),col_names = c("rmsd3","dset"))
rmsd_4 = read_csv(str_c("../data/ModelB","4","_rmsd.csv"),col_names = c("rmsd4","dset"))

rmsd=full_join(rmsd_1,rmsd_2, by="dset")
rmsd=full_join(rmsd,rmsd_3, by="dset")
rmsd=full_join(rmsd,rmsd_4, by="dset")

# # Normalizing values to model 1 for each dataset
# rmsd_rel = rmsd %>%
#   mutate("rmsd2"=rmsd2/rmsd1,
#          "rmsd3"=rmsd3/rmsd1,
#          "rmsd4"=rmsd4/rmsd1) %>%
#   select(!rmsd1) %>%
#   pivot_longer(cols=c("rmsd2","rmsd3","rmsd4"),names_to = "model",
#                values_to = "rmsd")
# 
# ggplot(rmsd_rel, aes(x=model,y=rmsd)) +
#   geom_boxplot() +
#   theme_bw()


rmsd_all = rmsd %>%
  pivot_longer(cols=c("rmsd1","rmsd2","rmsd3","rmsd4"),names_to = "model",
               values_to = "rmsd") %>%
  mutate(aic = case_when(
    model == "rmsd1" ~ 7*log(rmsd^2)+2*6,
    model == "rmsd4" ~  7*log(rmsd^2)+2*8,
    TRUE ~  7*log(rmsd^2)+2*7))


p= ggplot(rmsd_all, aes(x=model,y=rmsd)) +
  geom_boxplot(outlier.size = 3) +
  geom_jitter(data=rmsd_all[rmsd_all$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=rmsd_all[rmsd_all$dset==0,], color="red",size=3)+
  theme_bw()+
  scale_x_discrete(labels=c(bquote("Model "*beta*"1"),
                            bquote("Model "*beta*"2"),
                            bquote("Model "*beta*"3"),
                            bquote("Model "*beta*"4"))) +
  ggtitle("All models RMSD")
ggsave(p, filename = str_c("../3-site-model/figs/all_models_RMSD.png"),
       height = 4, width = 4)

p= ggplot(rmsd_all, aes(x=model,y=aic)) +
  geom_boxplot(outlier.size = 3) +
  geom_jitter(data=rmsd_all[rmsd_all$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=rmsd_all[rmsd_all$dset==0,], color="red",size=3)+
  theme_bw()+
  scale_x_discrete(labels=c(bquote("Model "*beta*"1"),
                            bquote("Model "*beta*"2"),
                            bquote("Model "*beta*"3"),
                            bquote("Model "*beta*"4"))) +
  ggtitle("All models AIC")
ggsave(p, filename = str_c("../3-site-model/figs/all_models_AIC.png"),
       height = 4, width = 4)

# Compare to 2-site model
rmsd_two_exp = read_csv(str_c("../data/exp_data_mins.csv")) %>%
  mutate(dset = 0)
rmsd_two_site = read_csv(str_c("../data/syn_data_mins.csv")) %>%
  mutate(dset = rep(c(1:99),length.out = n())) %>%
  bind_rows(rmsd_two_exp) %>%
  mutate(model = str_c("twosite",model)) %>%
  rename("rmsd" = "minRMSD") %>%
  dplyr::select(c("rmsd","dset","model"))

rmsd_both_models = rmsd %>%
  pivot_longer(cols=c("rmsd1","rmsd2","rmsd3","rmsd4"),names_to = "model",
               values_to = "rmsd") %>%
  bind_rows(rmsd_two_site) %>%
  mutate(aic = case_when(
    model == "rmsd1" ~ 7*log(rmsd^2)+2*6,
    model == "rmsd4" ~  7*log(rmsd^2)+2*8,
    str_detect(model, "^twosite") ~ 7*log(rmsd^2)+2*1,
    TRUE ~  7*log(rmsd^2)+2*7))

p= ggplot(rmsd_both_models, aes(x=model,y=rmsd)) +
  geom_boxplot(outlier.size = 3) +
  geom_jitter(data=rmsd_both_models[rmsd_both_models$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=rmsd_both_models[rmsd_both_models$dset==0,], color="#EF4A5F",size=3)+
  theme_bw()+
  scale_x_discrete(labels=c(bquote("Model "*beta*"1"),
                            bquote("Model "*beta*"2"),
                            bquote("Model "*beta*"3"),
                            bquote("Model "*beta*"4"),
                            "2-site IRF",
                            bquote("2-site NF"*kappa*"B"),
                            "2-site AND",
                            "2-site OR")) +
  ggtitle("All models RMSD 2-site and 3-site") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(p, filename = str_c("../3-site-model/figs/all_models_RMSD_2site_3site.png"),
       height = 4, width = 5)

p= ggplot(rmsd_both_models, aes(x=model,y=aic)) +
  geom_boxplot(outlier.size = 3) +
  geom_jitter(data=rmsd_both_models[rmsd_both_models$dset!=0,], color="black",alpha=0.3,size=3,width=rel(0.1))+
  geom_point(data=rmsd_both_models[rmsd_both_models$dset==0,], color="red",size=3)+
  theme_bw()+
  scale_x_discrete(labels=c(bquote("Model "*beta*"1"),
                            bquote("Model "*beta*"2"),
                            bquote("Model "*beta*"3"),
                            bquote("Model "*beta*"4"),
                            "2-site IRF",
                            bquote("2-site NF"*kappa*"B"),
                            "2-site AND",
                            "2-site OR")) +
  ggtitle("All models AIC 2-site and 3-site")
ggsave(p, filename = str_c("../3-site-model/figs/all_models_AIC_2site_3site.png"),
       height = 4, width = 5)


# Parameters
read_parameters = function(model_name){
  t_pars = c("t1","t2","t3","t4","t5","t6")
  if (model_name=="1") {
    param_names = t_pars
    
  } else if (model_name=="2") {
    param_names = c("KI2",t_pars)
    
  } else if (model_name=="3") {
    param_names = c("C",t_pars)
    
  } else if (model_name=="4") {
    param_names = c("KI2","C",t_pars)
    
  }
  df = read_csv(str_c("../data/ModelB",model_name,"_parameters.csv"),col_names = c(param_names,"dset"),
                show_col_types = FALSE) %>%
    pivot_longer(param_names, names_to = "parameter",values_to = "val") %>%
    mutate(par_type = case_when(
      parameter %in% t_pars ~ "t_parameter",
      TRUE ~ "Other"))
  return(df)
}

params_1 = read_parameters("1")
params_2 = read_parameters("2")
params_3 = read_parameters("3")
params_4 = read_parameters("4")
