X = as.matrix(read.table('DATAFILE',header=F))
X = t(scale(X,center=T,scale=F))
p = ncol(X)
n = nrow(X)

index<-seq(1,150)
labels<-read.table('LABELFILE',header=F,as.is=c(1))[,1]
labelint<-as.integer(as.factor(labels))
possible_shapes<-1
colors<-floor((labelint-1)/possible_shapes)+1
shapes<-floor((labelint-1)%%possible_shapes)+1
sequence<-seq(0,(p-1))
files<-read.table('CLUSTERFILE',as.is=c(1),header=F)
gamma = seq(1,length(files[,1]))

## Plot the cluster path
library(ggplot2)
svdX = svd(X)
pc_x<-1
pc_y<-2
pc = svdX$u[,c(pc_x,pc_y),drop=FALSE]
pc.df = as.data.frame(t(pc)%*%X)
nGamma = length(gamma)
df.paths = data.frame(x=c(),y=c(), group=c())
for (j in 1:nGamma) {
  U<-t(scale(as.matrix(read.table(file=files[j,],header=F)),center=T,scale=F))
  cat(paste("Projecting",files[j,],"\n"))
  pcs = t(pc)%*%U
  x = pcs[1,]
  y = pcs[2,]
  df = data.frame(x=x,y=y, group=1:p)
  df.paths = rbind(df.paths,df)
}
X_data = as.data.frame(t(X)%*%pc)
colnames(X_data) = c("x","y")
X_data$Species = labels
X_data$Shapes = shapes
X_data$Colors = colors
unique_shapes<-unique(shapes)
unique_colors<-unique(colors)
total_shapes=length(unique_shapes)
total_colors=length(unique_colors)
color_vec<-as.vector(t(matrix(rep(unique_colors,total_shapes),total_colors,total_shapes)))
data_plot = ggplot(data=df.paths,aes(x=x,y=y))
data_plot = data_plot + geom_point(data=X_data,aes(x=x,y=y,colour=Species),alpha=1,size=5)
data_plot = data_plot + scale_colour_manual(values=color_vec)
data_plot = data_plot + geom_path(aes(group=group),alpha=1,size=1)
data_plot = data_plot + xlab(paste('Principal Component 1')) + ylab(paste('Principal Component 2'))
#data_plot + theme(axis.line=element_blank(),axis.text.x=element_blank(), axis.text.y=element_blank(),axis.ticks=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(), panel.background=element_blank(),panel.border=element_blank(),panel.grid.major=element_blank(), panel.grid.minor=element_blank(),plot.background=element_blank(),legend.position="none") 
ggsave(filename='OUTPUTPDF',width=13,height=10,scale=1,limitsize=F)
