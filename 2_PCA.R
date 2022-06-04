#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
library(ggpubr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(beepr)
library(TCGAbiolinks)

#------------------------------------------------------------------------------
# Data information
#------------------------------------------------------------------------------
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")

projects <- TCGAbiolinks:::getGDCprojects()
names(projects)
listProjectCancer<-projects[grep("TCGA", projects$id),c("id","tumor","name")]
listProjectCancer$tumor

# typeData="Exp"
typeData="Met"
# typeData="miRNA"

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    message("*beep*")
    Sys.sleep(freq)
  }
}
beeper(1)

#------------------------------------------------------------------------------
# Loop configuration
#------------------------------------------------------------------------------

PCA_graph=FALSE
# PCA_graph=TRUE
counter=0
start=1
end=33
# end=length(listProjectCancer$tumor)

# dataPCA is to store information about samples and PC explaned variance
dataPCA<-data.frame(Project=character(),
                    PC1=double(),
                    PC2=double(),
                    numSamples=integer()
)

#------------------------------------------------------------------------------
# Loop
#------------------------------------------------------------------------------
counter=0
for (project in listProjectCancer$tumor[start:end]){
  global.start.time<-proc.time()[3]
  # project="BRCA"
  # project=listProjectCancer$tumor[start:end]
  #------------------------------------------------------------------------------
  # Title
  #------------------------------------------------------------------------------
  
  counter=counter+1
  message      ("\n*-----------------------------------*")
  message(paste(" - INITIATING PCA ANALYSIS OF ",project,sep=""))
  message      ("*-----------------------------------*")
  message(paste(" - Total PCA: ",counter,"/",length(listProjectCancer$tumor[start:end]),sep=""))
  message      ("*-----------------------------------*")
  loadFile=paste("data_",typeData,"/data",typeData,"_",project,".Rdata",sep="")


  #------------------------------------------------------------------------------
  # Loading data
  #------------------------------------------------------------------------------
  
  message(paste(" - Information - ",project," - Uploaded file: ",loadFile,sep=""))
  load(file = loadFile)
  message(paste("   *Done* ",sep=""))
  #-------------
  
  # reviso clase de estructura del objeto
  message(paste(" - Information - Data dimension: ",dim(data)[1]," x ",dim(data)[2]," (genes/probes x muestras)",sep=""))
  class(data)
  dim(data)
  
  # names of probe/gene
  probe_names<-rownames(data)
  head(probe_names)
  
  # rm(data)
  
  #------------------------------------------------------------------------------
  # PCA analysis
  #------------------------------------------------------------------------------
  
  # https://towardsdatascience.com/principal-component-analysis-pca-101-using-r-361f4c53a9ff
  #message(paste(" - Information - ",project," - Initialice Principal Componente Analysis",sep=""))
  message(paste(" - Information - Initialice Principal Componente Analysis",sep=""))
  start.time <- proc.time()[3]
  pca_results<-prcomp(data,
                      center = TRUE,
                      scale. = if(typeData!="Met") TRUE else FALSE
  )
  # ?prcomp
  message(paste("   *Done* - Time: ",round(proc.time()[3]-start.time,3),"s",sep=""))
  
  pca_sum<-summary(pca_results)
  # pca_sum
  
  #------------------------------------------------------------------------------
  # Análisis de las componentes principales
  #------------------------------------------------------------------------------
  
  # Porcentaje que explica las PC1 y PC2
  # ncol(pca_sum$importance)
  # elijo si quiero mostrar 10 PC o el máximo que haya
  lastCol<-if(ncol(pca_sum$importance)>10) 10 else ncol(pca_sum$importance)
  var_exp<-pca_sum$importance[2,1:lastCol]*100
  # var_exp
  message(paste(" - Information - ",project," - PCA variance for:\n PC1 = ",var_exp[1], "%\n PC2 = ",var_exp[2],"%",sep=""))
  
  #gráfico sobre la varianza explicada por cada PC
  # library(factoextra)
  # fviz_eig(pca_results)
  
  # Coordenadas de las 2 componentes principales (PC1 & PC2)
  pc_scores<-pca_results$x[,1:2]
  message(paste(" - Information - ",project," - pc_scores numbers: ",dim(pc_scores)[1],sep=""))
  
  #------------------------------------------------------------------------------
  # Cargo datos para analizar el plot - OPCIONAL según valor de PCA_graph
  #------------------------------------------------------------------------------
  

  if(PCA_graph){
    message(paste(" - Information - ",project," - Creating general cancer image in pdf",sep=""))
    dir.create("plots/PCA_plot",
               showWarnings=FALSE,
               recursive=TRUE)
    ?dir.create()
    pltCancer <- ggplot(data = as.data.frame(pc_scores), aes(x=PC1,y=PC2)) + 
      geom_point(size=0.001) + 
      theme_bw() + 
      xlab(paste0("PC1 ",var_exp[1]," %")) +
      ylab(paste0("PC2 ",var_exp[2]," %"))
    print(pltCancer)
    plotName=paste("plots/PCA_plot/PCA",typeData,"_",project,"_plot.pdf",sep="")
    plotName
    ggsave(plotName, plot = pltCancer)
    dev.off()
    message(paste("   *Done* ",sep=""))
  }

  #------------------------------------------------------------------------------
  # Confirmación que los nombres de los genes son los mismos en todo momento
  #------------------------------------------------------------------------------
  if(!(table(rownames(data)==rownames(pc_scores)))){
    message("Data genes do not match pc scores - STOPPING ANALYSIS")
    break
  }
  
  #------------------------------------------------------------------------------
  # Choose type of dataset to save
  #------------------------------------------------------------------------------
  
  filename = paste("data_",typeData,"/PCA",typeData,"_",project,".Rdata",sep="")
  
  message(paste(" - Saving - PCA file:",filename))
  save(pc_scores, file = filename)
  
  dataPCA[counter,1]<-project
  dataPCA[counter,2]<-var_exp[1]
  dataPCA[counter,3]<-var_exp[2]
  dataPCA[counter,4]<-ncol(data)
  write.table(dataPCA, file = paste("logs/PCA_",typeData,"_log.txt",sep=""))
  rm(list=c("pc_scores","data"))
  
  beeper(1)
}
message(paste("  *FINISH PCA* ",sep=""))
message(paste("        total time: ",round(proc.time()[3]-global.start.time,2),sep=""))
beeper(1,type=3)
filename = paste("data_",typeData,"/PCASummary",typeData,".Rdata",sep="")
message(paste(" - Saving - Summary of PCA in file:",filename))
save(dataPCA, file = filename)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Visualization of PC variance explaned per project
#------------------------------------------------------------------------------
# Loading saved summary of PC on .RData or .txt

# loadPCASummary = paste("data_",typeData,"/PCASummary",typeData,".Rdata",sep="")
loadPCASummary = paste("logs/PCA_Meth_log.txt",sep="")
message(paste(" - Information - Uploaded file: ",loadPCASummary,sep=""))

dataPCA<-read.table(loadPCASummary, header = TRUE, sep = "", dec = ".")
# load(file = loadPCASummary)
message(paste("   *Done* ",sep=""))

dat_long <- dataPCA[,-4] %>%
              gather("PC", "Variance", -Project)

plot1<-ggplot(dat_long,aes(x=Project,y=Variance,fill=factor(PC)))+
  geom_bar(stat="identity",position="dodge")+
  #labs(fill='Principal\nComponent') +
  ylab("Explained variance")+
  # xlab("Project") +
  xlab("") +
  scale_y_continuous(limits = c(0,max(dat_long$Variance)*1.1), expand = c(0, 0)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

plot2<-ggplot(dat_long,aes(x=PC,y=Variance,fill=factor(PC)))+
  geom_bar(stat="identity",position="dodge")+
  #labs(fill='Principal\nComponent') +
  ylab("Explained variance")+
  xlab("Project") +
  scale_y_continuous(limits = c(0,max(dat_long$Variance)*1.1), expand = c(0, 0)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

ggarrange(plot1, plot2, 
          labels = c("Principal Components per project", "Global Principal Componens"),
          ncol = 1, nrow = 2)

savePCASummary=paste("plots/PCA_plot/PCA_",typeData,"_Summary.png",sep="")
ggsave(savePCASummary)

#------------------------------------------------------------------------------



