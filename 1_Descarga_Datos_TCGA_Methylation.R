#-------------------------------------------------------
# Codigo que descarga los datos .idat sobre los 
# exprimentos de metilación. NO hay funcion GDCPrepare.
#-------------------------------------------------------


# TCGAanalyze_Preprocessing: Preprocessing of Gene Expression data (IlluminaHiSeq_RNASeqV2)
# You can easily search TCGA samples, download and prepare a matrix of gene expression.
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(beepr)
library(sesame)

projects <- TCGAbiolinks:::getGDCprojects()
listProjectCancer <- projects[grep("TCGA", projects$id),c("id","tumor","name")]
# listProjectCancer$id

downloadStatus <- data.frame(Project = rep(NA,length(listProjectCancer$id)),
                             QueryStatus = rep(NA,length(listProjectCancer$id)),
                             DownloadStatus = rep(NA,length(listProjectCancer$id))
                             )
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    Sys.sleep(freq)
  }
}
beeper(1)


start=1
end=33
counter=0

for (project in listProjectCancer$id[start:end]){
  # project = listProjectCancer$id[start:end]
  counter=counter+1
  downloadStatus$Project[counter] = project
  beeper(1)
  message      ("\n*-----------------------------------*")
  message(paste(" - INITIATING DONWLOAD OF ",project,sep=""))
  message(paste(" - Total projects: ",counter,"/",length(listProjectCancer$id[start:end]),sep=""))
  message      ("*-----------------------------------*\n ")
  # project="TCGA-COAD"
  #nombre de la base de datos de cancer
  cancer.Name<-sub("TCGA-","",project)
  
  # define a variable with missing values so it won't look for samples barcodes
  listSamples <- rlang::missing_arg()

  #-----------------------------------
  # STEP 1: Search, download, prepare |
  #-----------------------------------
  # 1.1 - DNA methylation
  # ----------------------------------
  # DNA methylation aligned to hg38
  
  # ----------------------------------------------------------------------------
  # GDCquery - Configuro consulta de los datos a descargar
  # ----------------------------------------------------------------------------
  
  message("\n - Information - GDCquery function\n")
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <<- 1 
    attempt <- attempt + 1
    message(paste("** Attempt GDCDownload: ",attempt," attempt\n",sep=""))
    check<-tryCatch({
        
        #--- Original code
      query_met <- GDCquery(project = project,
                            data.category = "DNA Methylation",
                            data.type = "Masked Intensities", 
                            legacy = FALSE,
                            file.type = ".idat",
                            platform = "Illumina Human Methylation 450"
                            # barcode =  listSamples
      )
      # getResults(query_met)
    # getResults(query_met,cols=c("cases","file_name"))
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
      downloadStatus[counter,2:3]<-"Error GDC query" 
      next
    }
  }
  
  # n_sample<-length(query_met$results[[1]]$cases)
  # message(paste("Information - Samples found for ",cancer.Name,": ",n_sample,sep=""))
  # unique(query_met$results[[1]]$data_type)
  
  # ----------------------------------------------------------------------------
  # GDCdownload - Descargo los datos del servidor
  # ----------------------------------------------------------------------------
  message("\n - Information - GDCdownload function\n")
  
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <- 1 
    attempt <- attempt + 1
    message(paste("** Attempt GDCDownload: ",attempt," attempt\n",sep=""))
    tryCatch({
      #--- Code to repeate if error appears
      GDCdownload(query_met,
                  directory = "data/GDCdata",
                  method = "api",
                  files.per.chunk	= 6
                  )
        check<<-0
      }, 
      error=function(e){
        message("\n ******* Original error message:\n")
        message(e)
        message("\n\n ******* Trying client method:\n")
        GDCdownload(query_met,
                    directory = "data/GDCdata",
                    # method = "api",
                    method = "client",
                    files.per.chunk	= 6)
        message("\n Waiting 15 seconds before trying again...")
        Sys.sleep(15)
        check<<-1
      }
    )
    # print(paste("Print check -",check,"-",sep=""))
  }
  if(attempt>=3){
    downloadStatus[counter,2:3]<-"Error GDC download" 
    next
  }

  beeper(2)
  message(paste("\nDONE DONWLOAD PROJECT ",project," - Total: ",counter,"/",length(listProjectCancer$id[start:end]),"\n \n              ++++++++++++++++ \n",sep=""))
}
beeper(10)
beep(0)
?beep
