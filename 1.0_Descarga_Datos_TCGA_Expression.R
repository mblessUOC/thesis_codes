# TCGAanalyze_Preprocessing: Preprocessing of Gene Expression data (IlluminaHiSeq_RNASeqV2)
# You can easily search TCGA samples, download and prepare a matrix of gene expression.

#Ejemplo 1: https://bioconductor.org/packages/devel/bioc/vignettes/TCGAbiolinks/inst/doc/download_prepare.html#Gene_expression:_aligned_against_hg38
#Ejemplo 2:
#-------------------------------------------------------------------------------
# Librerias
#-------------------------------------------------------------------------------
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(beepr)

#-------------------------------------------------------------------------------
# Variables
#-------------------------------------------------------------------------------

setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")

projects <- TCGAbiolinks:::getGDCprojects()
listProjectCancer<-projects[grep("TCGA", projects$id),c("id","tumor","name")]
# listProjectCancer$id
getProjectSummary(listProjectCancer$id[1])


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
#-------------------------------------------------------------------------------
# Loop
#-------------------------------------------------------------------------------
dataFiltered<-data.frame(Project=listProjectCancer$id,
                         Sample.Size=rep(NA,length(listProjectCancer$id)),
                         Init.Gene.Size=rep(NA,length(listProjectCancer$id)),
                         Filt.Chrx.ChrY.ChrM=rep(NA,length(listProjectCancer$id)),
                         Fin.Gene.Size=rep(NA,length(listProjectCancer$id))
)

start=1
end=1
counter=start-1
for (project in listProjectCancer$id[start:end]){
  # project=listProjectCancer$id[start]
  counter=counter+1
  message      ("\n*--------------------------------------------*")
  message(paste(" - INITIATING DONWLOAD OF ",project,sep=""))
  message(paste(" - Total projects: ",counter,"/",length(listProjectCancer$id),sep=""))
  message      (  "*--------------------------------------------*")
  # project="TCGA-COAD"
  #nombre de la base de datos de cancer
  cancer.Name<-sub("TCGA-","",project)
  
  #----------
  #get information from the project
  #----------
  TCGAbiolinks:::getProjectSummary(project)
  #----------
  
  #----------
  # listSample
  #----------
  # You can define a list of samples to query and download providing relative TCGA barcodes.
  # listSamples <- c("TCGA-E9-A1NG-11A-52R-A14M-07","TCGA-BH-A1FC-11A-32R-A13Q-07",
  #                  "TCGA-A7-A13G-11A-51R-A13Q-07","TCGA-BH-A0DK-11A-13R-A089-07",
  #                  "TCGA-E9-A1RH-11A-34R-A169-07","TCGA-BH-A0AU-01A-11R-A12P-07",
  #                  "TCGA-C8-A1HJ-01A-11R-A13Q-07","TCGA-A7-A13D-01A-13R-A12P-07",
  #                  "TCGA-A2-A0CV-01A-31R-A115-07","TCGA-AQ-A0Y5-01A-11R-A14M-07")
  # define a variable with missing values so it won't look for samples barcodes
  listSamples<- rlang::missing_arg()
  #----------
  
  #----------
  # New updates in TCGA database
  #----------
  # HTSeq - FPKM-UQ is not available anymore
  # https://www.biostars.org/p/9516907/#9517068
  # https://github.com/BioinformaticsFMRP/TCGAbiolinks/issues/495
  
  
  
  # Query platform Illumina HiSeq with a list of barcode 
  # query <- GDCquery(project = "TCGA-BRCA",
  #                   data.category = "Gene expression",
  #                   data.type = "Gene expression quantification",
  #                   experimental.strategy = "RNA-Seq",
  #                   # platform = "Illumina HiSeq",
  #                   file.type = "results",
  #                   barcode = listSamples,
  #                   legacy = TRUE)
  
  # ?GDCquery
  # query.exp <- GDCquery(project = project,
  #                       data.category = "Transcriptome Profiling",
  #                       data.type = "Gene Expression Quantification", 
  #                       workflow.type = "HTSeq - FPKM",
  #                       # barcode = listSamples
  
  ##### https://docs.gdc.cancer.gov/Data/Release_Notes/Data_Release_Notes/#data-release-320
  # Only workflow.type = "STAR - Counts" is available for downloading. It also contains
  # information of FKPM and TPM when assays() apply to object
  
  #----------
  
  # ?GDCquery
  message("\n - Information - GDCquery function\n")
  
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <<- 1 
    attempt <- attempt + 1
    message(paste("** Attempt GDCDownload: ",attempt," attempt\n",sep=""))
    check<-tryCatch({
      
      #--- Original code
      query.exp <- GDCquery(project = project,
                            data.category = "Transcriptome Profiling", 
                            data.type = "Gene Expression Quantification", 
                            workflow.type = "STAR - Counts",
                            barcode =  listSamples
                            )
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
      dataFiltered[counter,2:5]<-"Error GDC query" 
      next
    }
  }
  
  # Numebers of genes for genome version hg38
  # dim(TCGAbiolinks:::get.GRCh.bioMart("hg38"))
  
  # dir.create("data/GDCdata")
  # ?GDCdownload
  # GDCdownload(query.exp)
  
  message("\n - Information - GDCdownload function\n")
  
  #-------------
  # Loop which reitarate the code if there is an error
  # check <- value to check if there was an error or not. If no error the results of
  #          tryCatch() will be NULL, otherwise it will be 1
  # attempt <- will measure the time to try to execute the code
  #-------------
  
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <- 1 
    attempt <- attempt + 1
    message(paste("** Attempt GDCDownload: ",attempt," attempt\n",sep=""))
    tryCatch({
        
      #--- Code to repeate if error appears
      GDCdownload(query.exp,
                  directory = "data/GDCdata",
                  method = "api",
                  # method = "client",
                  files.per.chunk	= 10)
      #--- Code to repeate if error appears
      check<<-0
      }, 
      error=function(e){
        message("\n ******* Original error message:\n")
        message(e)
        message("\n\n ******* Trying client method:\n")
        GDCdownload(query.exp,
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
    
    dataFiltered[counter,2:5]<-"Error GDC download" 
    next
  }
  #genes que no pudieron ser mapeados con el genoma de referencia no se agregaron
  # ?GDCprepare
  
  message("\n - Information - GDCprepare function\n")
  
  check <- 1
  attempt <- 0
  while( check==1 && attempt < 3 ) {
    check<-1
    attempt <- attempt + 1
    message(paste("** Attempt GDCprepare: ",attempt," attempt\n",sep=""))
    tryCatch({
      #--- Code to repeate if error appears
      
      RNAseq <- GDCprepare(query.exp,
                           directory = "data/GDCdata",
                           summarizedExperiment = TRUE) 
      #--- Code to repeate if error appears
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
    dataFiltered[counter,2:5]<-"Error GDC prepare" 
    next
  }
      
      
  # "summarizedExperiment = TRUE" has gene information (start, end and others) and sample information.
  #---------
  # maybe an explanation to unstranded fpkm
  # https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-015-1876-7
  #---------
  
  #---------
  # Resume of the SummarizedExperiment object
  #---------
  # Info clase de datos
  class(RNAseq)
  
  # Infor datos en RNAseq
  RNAseq
  # dim(RNAseq)# dim1: genes; dim2: muestras
  
  # Información de los tipos de recuentos disponibles
  assayNames(RNAseq)
  # Types of counts: unstranded stranded_first stranded_second fpkm_unstrand tpm_unstrand
  # maybe an explanation to fpkm_unstrand
  # https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-015-1876-7
  
  # ?assays
  # head(assay(RNAseq))
  # colData(RNAseq)
  
  head(rowData(RNAseq))
  
  # extraigo datos de fpkm_unstrand
  data<- assays(RNAseq)$fpkm_unstrand
  dim(data)
  message(paste("\n - Information datos:"))
  message(paste(  "   * Gene size: ",dim(data)[1],sep=""))
  message(paste(  "   * Sample size: ",dim(data)[2],"\n",sep=""))
  
  # actualizo dataFiltered
  dataFiltered[counter,"Sample.Size"] = dim(data)[2]
  dataFiltered[counter,"Init.Gene.Size"] = dim(data)[1]
  #-----------------
  # Explicación de por qué casi el 5% de los genes no se mapean:ç
  #-----------------
  # https://support.bioconductor.org/p/111325/
  # https://rpubs.com/tiagochst/TCGAbiolinks_mapping_genes
  
  # genes que no fueron mapeados en el genoma de referencia
  # ?GDCprepare
  if(FALSE){
    RNAseq_notMapped <- GDCprepare(query.exp,
                                   directory = "data/GDCdata",
                                   summarizedExperiment = FALSE) #se dejan aquellos  
  }
  
  # 
  # # Accessing www.ensembl.org to get gene information
  # gene.information <- TCGAbiolinks::get.GRCh.bioMart("hg38")
  # 
  # # How many are not mapped in the in the database
  # RNAseq_notMapped$X1[!gsub("\\.[0-9]*$","",BRCARnaseq_notMapped$X1) %in% gene.information$ensembl_gene_id] %>% length
  #-----------------
  # getwd() 
  
  #-----------------
  # Información obtenida del objecto SummarizedExperiment
  #-----------------
  
  # metadata de los datos
  # colData(RNAseq)
  
  #resumen de los datos
  # names(colData(RNAseq))
  
  # filenameMeta = paste("data_Exp/metadataExp_",cancer.Name,".Rdata",sep="")
  # message(paste("Saving - Metadata file:",filenameMeta))
  # save(metadata, file = filenameMeta)
  
  # metadata a partir del query.met
  metadata<-getResults(query.exp)
  filenameMeta = paste("data_Exp/metadataExp_",cancer.Name,".Rdata",sep="")
  message(paste("\n - Saving - Metadata file:",filenameMeta),"\n")
  save(metadata, file = filenameMeta)
  
  # tipo de tumor:
  #   - solid tissue normal (NT): muestra extraido del tejido sano 
  #   - primary solid tumor (TP): muestra tumoral
  unique(colData(RNAseq)$definition)
  unique(colData(RNAseq)$shortLetterCode)
  
  # barcode of patient and samples
  metadata$cases.submitter_id[1:3]
  metadata$cases[1:3]
  unique(colData(RNAseq)$patient)[1:3]
  unique(colData(RNAseq)$barcode)[1:3]
  
  # whole metadata for the first 2 samples
  t(metadata[1:2,])
  
  # checking sample size in metadata and dataset
  sizes <- length(metadata$cases) == ncol(data)
  message(paste("\n - Informaation - Sample size in metadata = sample dataset?: ",sizes,"\n"))
  
  #-----------------
  
  # extraigo datos gene_id (nombres de los genes)
  data
  rownames(data)
  gene_data<-as.data.frame(rowRanges(RNAseq))
  table(rownames(data)==gene_data$gene_id)
  
  #extraigo ubicación cromosomática de cada gen
  head(gene_data)
  table(gene_data$seqnames)
  # names(gene_data)
  # gene_data[gene_data$seqnames=="chrM",][1:3,]
  
  # chrM: gene information from Mitochondria
  # extraigo listado de genes ensembl en el chrY y chrM
  # seqname: cromosomas -> chrY
  
  # library(tidyverse)
  chrDel <- c("chrX", "chrY", "chrM")
  ensembl_gene_chrX.Y.M <- gene_data %>%
    filter(seqnames %in% chrDel) %>% # filter every row that match strings in chrDel  
    select(gene_id)
  message(paste("\n - Filtering - Deletion genes in chrX, chrY and chrM:",nrow(ensembl_gene_chrX.Y.M),"\n"))
  dataFiltered[counter,"Filt.Chrx.ChrY.ChrM"]=nrow(ensembl_gene_chrX.Y.M)
  
  # Genero dataframe con valores FPKM de cada gen para cada muestra
  # para averiguar las opciones utiliar names(assays(RNAseq)) 
  
  # Elimino genes del chrY y chrM
  # class(ensembl_gene_chrY.M)
  # type(ensembl_gene_chrY.M)
  # class(data.frame(row.names(data)))
  # type(row.names(data))
  # dim(ensembl_gene_chrY.M)
  # dim(data.frame(row.names(data)))
  
  data<-data[!(  row.names(data) %in% ensembl_gene_chrX.Y.M[,1] ),]
  dataFiltered[counter,"Fin.Gene.Size"] = dim(data)[1]
  message(paste("\n - Information - Final Gene size: ",dataFiltered[counter,"Fin.Gene.Size"],"\n",sep=""))
  
  
  # Transformación log2 https://www.biostars.org/p/100926/
  # Indeed microarray values and RPKM/FPKM values are better correlated when 
  # log-transformed. The reason for it is that the distribution of RPKM/FPKM values
  # is skewed, and by log-transforming it we could bring it closer to normal
  # distribution. It is needless to say that many statistical tests require
  # normally-distributed data.
  
  data <- log2( data + 1 ) #se agregar el 1 para evitar transformar el 0 y que de -Inf. log2(0)=-Inf / log2(1)=0
  message(paste("\n -- Information dataFiltered --"))
  message(paste("\n",names(dataFiltered),": ",dataFiltered[counter,]))
  print(dataFiltered)
  
  filenameData = paste("data_Exp/dataExp_",cancer.Name,".Rdata",sep="")
  message(paste("\n - Saving - Data file:",filenameData,"\n"))
  save(data, file = filenameData)
  message(paste("\nDONE DONWLOAD PROJECT ",project," - Total: ",counter,"/",length(listProjectCancer$id),"\n \n              ++++++++++++++++ \n",sep=""))
  beeper(1)
  write.table(dataFiltered, file = paste("logs/Download_RNA_log.txt",sep=""))
}
filenameFilter = paste("data_Exp/dataFileterd_Exp.Rdata",sep="")
message(paste("\n - Saving - Data file:",filenameFilter,"\n"))
save(dataFiltered, file = filenameFilter)
beeper(1,type = 3)


# convert START-count (raw) to FPKM

# fpkm = function (counts, effective_lengths) {
  # exp(log(counts) - log(effective_lengths) - log(sum(counts)) + log(1E9))
# }
