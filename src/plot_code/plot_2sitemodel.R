library(tidyverse)
library(RColorBrewer)

## Plot Synthetic and experimental data
data_pts = read_csv("../data/syn_data.csv")
syn_pts = data_pts %>% filter(exp != 0)
exp_pts = data_pts %>% filter(exp == 0)

p = ggplot(syn_pts,  aes(x=IRF, y = NFkB, fill=IFNb, color=IFNb)) +
  geom_point(size=2, alpha = 0.7, shape=21) +
  geom_point(data = exp_pts, size=5, alpha = 1, shape=21, color="black", stroke=1) +
  scale_fill_distiller(palette = "RdYlBu", direction =-1) +
  scale_color_distiller(palette = "RdYlBu", direction =-1) +
  xlab(expression("[NF"*kappa*"B], normalized")) + ylab("[IRF], normalized") +
  labs(color = expression("IFN"*beta*" mRNA"), fill = expression("IFN"*beta*" mRNA")) +
  theme_bw() +
  theme(axis.text=element_text(size = rel(1.5)), axis.title = element_text(size = rel(1.5)),
        legend.title = element_text(size = rel(1.5)), legend.text = element_text(size = rel(1.1)))
ggsave(p, filename = str_c("../2-site-model/figs/all_data_points.png"),
       height = 6, width = 9)

## Plot best C/corresponding RMSD for all models
opt_par_exp = read_csv("../data/exp_data_mins.csv") %>%
  mutate(model = as_factor(model)) %>% 
  mutate(group = row_number()) %>%
  pivot_longer(c(minRMSD, bestC), values_to = "val", names_to = "measure")
opt_par_syn = read_csv("../data/syn_data_mins.csv") %>%
  mutate(model = as_factor(model)) %>% 
  mutate(group = row_number()) %>%
  pivot_longer(c(minRMSD, bestC), values_to = "val", names_to = "measure")
opt_par_all = bind_rows(opt_par_exp, opt_par_syn)
facet_names = c(`1`="AND", `2`="NFkB", `3`="IRF", `4`="OR")

p=ggplot(opt_par_syn, aes(x=measure, y=log10(val))) +
  geom_line(alpha=0.5, aes(group=group)) +
  geom_point(alpha=0.5) +
  geom_boxplot(data= opt_par_all, alpha=0.5) +
  geom_line(data=opt_par_exp, color="red", aes(group=group)) +
  geom_point(data=opt_par_exp, color="red") +
  facet_grid(~model, labeller = as_labeller(facet_names)) + 
  theme_bw()  +
  theme(strip.placement = "outside",
      strip.background = element_rect(fill=NA,colour="grey50"),
      panel.spacing=unit(0,"cm"), 
      axis.title.x=element_blank())
ggsave(p, filename = str_c("../2-site-model/figs/best_fits.png"),
       height = 6, width = 9)
