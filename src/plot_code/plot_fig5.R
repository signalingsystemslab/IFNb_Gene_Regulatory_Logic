#######################
# process code 
#######################
datafolder = "../data/"

#######################
# plot code
#######################
library(gplots)

load(file=paste(datafolder,"mef.RData",sep = ""))

color.wt <- "#262626"
color.ac <- "#FF3030"
color.abc <- "#FF7F00"
color.ac50 <- "#8B814C"

### Box plot 
idx <- c(2,10,6,14)
plot_data <- log2(plotdata3[,idx]/plotdata3[,1])
colnames(plot_data) <- NULL
#setEPS()
#postscript(paste(figfolder,"/subfigs/fig5_mef_box.eps",sep=""),
#           horizontal = F,paper="special",width=2.8578,height = 2.1632 )
par(mar=c(1,1,1,1)-.5)
boxplot(plot_data[,2:4],horizontal=F,col=c(color.ac50,color.ac,color.abc),axes=F,names=NULL,cex=.85,outline=F,ylim=c(-2,2))
axis(1,at=seq(1,3,by=1),tick = T,labels = F,lty = 1,tck=0.02) #tck=-.008,
axis(2,at=seq(-2,1,by=1),tick = T,labels = F,lty = 1,tck=0.02) #tck=-.008,
box()
#dev.off()
t.test(log2(plotdata3[,idx[2]]/plotdata3[,1]),log2(plotdata3[,idx[3]]/plotdata3[,1]),paired = T,alternative = "greater")
t.test(log2(plotdata3[,idx[3]]/plotdata3[,1]),log2(plotdata3[,idx[4]]/plotdata3[,1]),paired = T,alternative = "greater")




##The ranked bar graph
plotdata8 <- log2(plotdata7[,c(2,6)])
idx <- order(plotdata8[,1],decreasing = T)
#setEPS()
#postscript(paste(figfolder,"/subfigs/fig5_whole_ISGs.eps",sep=""),
#                 horizontal = F,paper="special",width=5,height = 2.5 )
par(mar=c(1,1,1,1)-.5)
barplot(t(plotdata8[idx,]),beside = T,col = c(color.ac50,color.abc),
        names.arg = rep("",length(plotdata8[idx,])),border = NA,
        xlim=c(15, length(idx)*3)-7,ylim=c(-2.5,6),axes =F) #,legend = c("",""),
#args.legend = list(x="top",bty="n"))
box()
axis(2,at=seq(-2,6,by=2),labels  = F,tck = .012)
axis(1,at=seq(1,221,by=3)+1,tick = T,labels = F,tck=.008,lty = 1)
grid(nx=NA,ny=NULL,col=gray(.5),lty=2)
#dev.off()


