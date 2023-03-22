library(tidyverse)
library(RColorBrewer)
setwd("~/IFN_paper/src/2-site-model")

## Plot Synthetic and experimental data
data_pts = read_csv("../data/syn_data.csv")
syn_pts = data_pts %>% filter(exp != 0)
exp_pts = data_pts %>% filter(exp == 0)

p = ggplot(syn_pts,  aes(x=IRF, y = NFkB, fill=IFNb, color=IFNb)) +
  geom_point(size=2, alpha = 0.7, shape=21) +
  geom_point(data = exp_pts, size=5, alpha = 1, shape=21, color="black", stroke=1) +
  scale_fill_distiller(palette = "RdYlBu", direction =-1) +
  scale_color_distiller(palette = "RdYlBu", direction =-1) +
  ylab(expression("[NF"*kappa*"B], normalized")) + xlab("[IRF], normalized") +
  labs(color = expression("IFN"*beta*" mRNA"), fill = expression("IFN"*beta*" mRNA")) +
  theme_bw() +
  theme(axis.text=element_text(size = rel(1.5)), axis.title = element_text(size = rel(1.5)),
        legend.title = element_text(size = rel(1.5)), legend.text = element_text(size = rel(1.1)))
ggsave(p, filename = str_c("../2-site-model/figs/all_data_points.png"),
       height = 6, width = 9)

## Plot best C/corresponding RMSD for all models
opt_par_exp = read_csv("../data/exp_data_mins.csv") %>%
  mutate(model = factor(model, levels=c("IRF","NFkB","AND","OR"))) %>% 
  mutate(group = row_number(), bestC=log10(bestC)) %>%
  rename(logbestC=bestC) %>%
  pivot_longer(c(minRMSD, logbestC), values_to = "val", names_to = "measure")
opt_par_syn = read_csv("../data/syn_data_mins.csv")  %>%
  mutate(model = factor(model, levels=c("IRF","NFkB","AND","OR"))) %>% 
  mutate(group = row_number(), bestC=log10(bestC)) %>%
  rename(logbestC=bestC) %>%
  pivot_longer(c(minRMSD, logbestC), values_to = "val", names_to = "measure")
opt_par_all = bind_rows(opt_par_exp, opt_par_syn)
# model_names = c(`3`="AND", `2`="NFkB", `1`="IRF", `4`="OR")
val_names = c(`logbestC`="log10(C) of best fit", `minRMSD` = "RMSD of best fit")

# p=ggplot(opt_par_syn, aes(x=measure, y=val)) +
#   geom_line(alpha=0.5, aes(group=group)) +
#   geom_point(alpha=0.5) +
#   geom_boxplot(data= opt_par_all, alpha=0.5) +
#   geom_line(data=opt_par_exp, color="red", aes(group=group)) +
#   geom_point(data=opt_par_exp, color="red") +
#   facet_grid(~model, labeller = as_labeller(model_names)) + 
#   theme_bw()  +
#   theme(strip.placement = "outside",
#       strip.background = element_rect(fill=NA,colour="grey50"),
#       panel.spacing=unit(0,"cm"), 
#       axis.title.x=element_blank())
# ggsave(p, filename = str_c("../2-site-model/figs/best_fits.png"),
#        height = 6, width = 9)


p=ggplot(opt_par_syn, aes(x=model, y=val)) +
  geom_point(alpha=0.5) +
  geom_boxplot(data= opt_par_all, alpha=0.5) +
  geom_point(data=opt_par_exp, color="red") +
  facet_wrap(~measure, nrow=2, scales="free_y", labeller = as_labeller(val_names)) + 
  theme_bw()  +
  # scale_x_discrete(labels=model_names) +
  expand_limits(y = 0) +
  expand_limits(y = 1) +
  theme(strip.placement = "outside",
        strip.background = element_rect(fill=NA,colour="grey50"),
        panel.spacing=unit(0,"cm"))
ggsave(p, filename = str_c("../2-site-model/figs/best_fits.png"),
       height = 6, width = 9)

