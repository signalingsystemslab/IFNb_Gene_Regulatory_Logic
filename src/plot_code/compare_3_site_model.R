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

# #### RMSD plot ####
# #  Normalizing values to model 1 for each dataset
# rmsd_1 = read_csv(str_c("../data/ModelB","1","_rmsd.csv"),col_names = c("rmsd1","dset"))
# rmsd_2 = read_csv(str_c("../data/ModelB","2","_rmsd.csv"),col_names = c("rmsd2","dset"))
# rmsd_3 = read_csv(str_c("../data/ModelB","3","_rmsd.csv"),col_names = c("rmsd3","dset"))
# rmsd_4 = read_csv(str_c("../data/ModelB","4","_rmsd.csv"),col_names = c("rmsd4","dset"))
# 
# rmsd=full_join(rmsd_1,rmsd_2, by="dset")
# rmsd=full_join(rmsd,rmsd_3, by="dset")
# rmsd=full_join(rmsd,rmsd_4, by="dset")
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
# 
# 
# rmsd_all = rmsd %>%
#   pivot_longer(cols=c("rmsd1","rmsd2","rmsd3","rmsd4"),names_to = "model",
#                values_to = "rmsd")
# 
# ggplot(rmsd_all, aes(x=model,y=rmsd)) +
#   geom_boxplot() +
#   theme_bw()

