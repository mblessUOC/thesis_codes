
#--------------------------------------------------------------------------------
# Código para descargar los datos de miRNA del projecto TCGA
#--------------------------------------------------------------------------------


# TCGAanalyze_Preprocessing: Preprocessing of Gene Expression data (IlluminaHiSeq_RNASeqV2)
# You can easily search TCGA samples, download and prepare a matrix of gene expression.
#--------------------------------------------------------------------------------
# Libraries
#--------------------------------------------------------------------------------

library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(beepr)
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")

#--------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------
beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    message("*beep*")
    Sys.sleep(freq)
  }
}
beeper(1)
#--------------------------------------------------------------------------------
# Variables for the loop
#--------------------------------------------------------------------------------
# summary of all GDC project
projects <- TCGAbiolinks:::getGDCprojects()

# extreact information only from TCGA
listProjectCancer<-projects[grep("TCGA", projects$id),c("id","tumor","name")]

# dataframe para guardar genes extraidos y el resultado final

dataFiltered<-data.frame(Project=listProjectCancer$id,
                         Sample.Size=rep(NA,length(listProjectCancer$id)),
                         Fin.Gene.Size=rep(NA,length(listProjectCancer$id))
)

# define a variable with missing values so it won't look for samples barcodes
listSamples<- rlang::missing_arg()

start=1
end=33
counter=0
#------------------------------------------------------------------------------
# Loop que descarga los datos y los filtra
#------------------------------------------------------------------------------

for (project in listProjectCancer$id[start:end]){
  dataFiltered$Sample.Size[counter]="Calculating"
  dataFiltered$Fin.Gene.Size[counter]="Calculating"
  # project=listProjectCancer$id[start]; project
  counter=counter+1
  message      ("\n*-----------------------------------------------------------*")
  message(paste(" - INITIATING DONWLOAD and PREPROCESSING OF ",project,sep=""))
  message(paste(" - Total projects: ",counter,"/",length(listProjectCancer$id[start:end]),sep=""))
  message      ("*-------------------------------------------------------------*")
  cancer.Name <- sub("TCGA-","",project)
  
  #------------------------------------------------------------------------------
  # GDCquery
  #------------------------------------------------------------------------------
  
  message("\n - Information - GDCquery function\n")
  
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <<- 1 
    attempt <- attempt + 1
    message(paste("   * Attempt GDCquery: ",attempt," attempt\n",sep=""))
    check<-tryCatch({
      
      #--- Original code
    queryDown.miR <- GDCquery(project = project, 
                          # data.category = "Gene expression",
                          data.category = "Transcriptome Profiling",
                          # data.type = "miRNA gene quantification",
                          data.type = "miRNA Expression Quantification",
                          barcode=listSamples,
                          # file.type = "hg38.mirna",
                          #workflow.type = "",
                          legacy = FALSE)
    check<<-0
    
    }, 
    error=function(e){
      #--- Code to repeate if error appears
      message("\n ******* Original error message:\n")
      message(e)
      message("\n Waiting 15 seconds before trying again...")
      Sys.sleep(15)
      # return(1)
      check<<-1
    }
    )
    # print(paste("Print check -",check,"-",sep=""))
    if(attempt>=3){
      dataFiltered[counter,2:3]<-"Error GDC query" 
      next
    }
  }
  
  message("\n - Information - GDCdownload function\n")

# getResults(queryDown.miR)
  
  #------------------------------------------------------------------------------
  # GDCdownload
  #------------------------------------------------------------------------------
  
  # ?GDCdownload
  # dir.create("data/GDCdata")
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <- 1 
    attempt <- attempt + 1
    message(paste("   * Attempt GDCDownload: ",attempt," attempt\n",sep=""))
    tryCatch({
      
      #--- Code to repeate if error appears


    GDCdownload(queryDown.miR,
                directory = "data/GDCdata",
                method = "api",
                files.per.chunk	= 10)
      #--- Code to repeate if error appears
      check<<-0
    }, 
    error=function(e){
      message("\n ******* Original error message:\n")
      message(e)
      message("\n\n ******* Trying client method:\n")
      GDCdownload(queryDown.miR,
                  directory = "data/GDCdata",
                  # method = "api",
                  method = "client",
                  files.per.chunk	= 10)
      message("\n Waiting 15 seconds before trying again...")
      Sys.sleep(15)
      check<<-1
    }
    )
    # print(paste("Print check -",check,"-",sep=""))
  }
  if(attempt>=3){
    dataFiltered[counter,2:3]<-"Error GDC download" 
    next
  }
  
  #------------------------------------------------------------------------------
  # GDCprepare
  #------------------------------------------------------------------------------  
  
  message("\n - Information - GDCprepare function\n")
  
  check <- 1
  attempt <- 0
  while( check==1 && attempt < 3 ) {
    check<-1
    attempt <- attempt + 1
    message(paste("   * Attempt GDCprepare: ",attempt," attempt\n",sep=""))
    tryCatch({
      #--- Code to repeate if error appears
      

    # ?GDCprepare
    data.miRNA <- GDCprepare(query = queryDown.miR, 
                               summarizedExperiment = TRUE,
                               directory = "data/GDCdata" )
    
    check<<-0
    }, 
    error=function(e){
      message("\n ******* Original error message:\n")
      message(e)
      message("\n Waiting 15 seconds before trying again...")
      Sys.sleep(15)
      check<<-1
    }
    )
    # print(paste("Print check -",check,"-",sep=""))
  }
  if(attempt>=3){
    dataFiltered[counter,2:3]<-"Error GDC prepare" 
    next
  }
  # original<-data.miRNA
  # data.miRNA<-original
  
  #------------------------------------------------------------------------------
  # Extract columns with reads_per_million information
  #------------------------------------------------------------------------------
  # Los datos extraidos de prepare están organizados en el dataframe de la siguiente manera:
  #  - filas: corresponden a cada miRNA
  #  - columnas: tiene la informaciónd de diferentes unidades para cada muestra, es decir que cada 3
  #              columnas tendré diferentes tipos de mediciones, por lo tanto hay que extraer las que uno
  #              desea utilizar.
  
  # The normalised expression (reads_per_million_miRNA_mapped_"barcode") of a panel of miRNAs is given for every sample. 
  
  message(" - Información - Generando matriz con \"reads per million\"")
  message("   * Extraction of columns with \"reads_per_million_miRNA_mapped\" ")

  # name in column miRNA_ID will be cut to rownames
  # La primera columna de data.miRNA corresponde al nombre de miRNA
  rownames(data.miRNA) <- data.miRNA$miRNA_ID

  # ejercicio que utiliza reads_per_million 
  # https://davetang.org/muse/2013/11/12/analysing-mirna/
  # filter columns that measured reads_per_million data 
  # el término "mapped" puede hacer referencia que dichos miRNA fueron complementarios a genes.
  
  # Extraigo wl nombre de aquellas columnas que contengan "reads_per_million"
  read_countData <-  colnames(data.miRNA)[grep("reads_per_million", colnames(data.miRNA))]
  
  # selecciono las columnas encontradas anteriormente y filtro data.miRNA
  data.miRNA <- data.miRNA[,read_countData]
  
  # sustituyo el nombre reads_per_million_miRNA_mapped+"barcode" por "barcode"
  colnames(data.miRNA) <- gsub("reads_per_million_miRNA_mapped_","", colnames(data.miRNA))
  # ejemplo del procesado
  # head(data.miRNA[,1:5])
  
  message(paste("\n - Información data - ",project,":",sep=""))
  message(paste("   * Muestras: ",ncol(data.miRNA),sep=""))
  message(paste("   * miRNA: ",nrow(data.miRNA),sep=""))
  dataFiltered$Sample.Size[counter]=ncol(data.miRNA)
  dataFiltered$Fin.Gene.Size[counter]=nrow(data.miRNA)
  #------------------------------------------------------------------------------
  # Metadata
  #------------------------------------------------------------------------------
  
  # metadata a partir del queryDown.miR
  metadata<-getResults(queryDown.miR)
  filenameMeta = paste("data_miRNA/metadatamiRNA_",cancer.Name,".Rdata",sep="")
  message(paste("\n - Saving - Metadata file:",filenameMeta))
  save(metadata, file = filenameMeta)

  #------------------------------------------------------------------------------
  # Info frecuencia muestras
  #------------------------------------------------------------------------------
  # tipo de tumor
  table(metadata$sample_type)
  message(paste(" - Información - Frecuencia de muestras: "))
  message(paste("   * ",unique(metadata$sample_type),": ",table(metadata$sample_type),"\n",sep=""))
 
  
  

  # barcode of patient and samples
  metadata$cases.submitter_id[1:3] #subject
  metadata$cases[1:3] #samples
  
  # Check if metadata and data.miRNA have the same barcodes at each position
  table(metadata$cases==colnames(data.miRNA))
  
  # project
  unique(metadata$project)

  # whole metadata for the first 2 samples
  t(metadata[1:2,])

    #------------------------------------------------------------------------------
  # Transformación log2
  #------------------------------------------------------------------------------

  # Transformación log2 https://www.biostars.org/p/100926/
  # Indeed microarray values and RPKM/FPKM values are better correlated when 
  # log-transformed. The reason for it is that the distribution of RPKM/FPKM values
  # is skewed, and by log-transforming it we could bring it closer to normal
  # distribution. It is needless to say that many statistical tests require
  # normally-distributed data.

  data<-log2(data.miRNA+1) #se agregar el 1 para evitar transformar el 0 y que de -Inf. log2(0)=-Inf / log2(1)=0

  filenameData = paste("data_miRNA/datamiRNA_",cancer.Name,".Rdata",sep="")
  message(paste(" - Saving - Data file:",filenameData))
  save(data, file = filenameData)
  message(paste("          * Done - ",project," *\n",sep=""))
  
  write.table(dataFiltered, file = "Download_miRNA_log.txt")
  beeper(1)
}
message      ("\n*------------------------------------------------*")
message(paste("          * FINISH Descarga datos miRNA *",sep=""))
beeper(1,type=3)
