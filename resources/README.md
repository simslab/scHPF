## Gene Files

Two column, tab-delimited text file of ENSEMBL gene ids and names with protein coding, T-cell receptor constant or immunoglobulin constant biotypes  in the GENCODE main annotation for [human](https://www.gencodegenes.org/human/) or [mouse](https://www.gencodegenes.org/mouse/).

### Included files
Human: `gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.stripped.txt`   
Mouse: `gencode.vM19.annotation.gene_l1l2.pc_TRC_IGC.stripped.txt`

### Generating gene files
Files were generated from GENCODE GTFs as follows:
```
# Select genes with feature gene and level 1 or 2
awk '{if($3=="gene" && $0~"level (1|2);"){print $0}}' gencode.v29.annotation.gtf > gencode.v29.annotation.gene_l1l2.gtf 

# Only include biotypes protein_coding, TR_C_g* and IG_C_g*
awk '{if($12~"TR_C_g" || $12~"IG_C_g" || $12~"protein_coding"){print $0}}' gencode.v29.annotation.gene_l1l2.gtf > gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.gtf

# Retrieve ENSEMBL gene id and name
awk '{{OFS="\t"}{gsub(/"/, "", $10); gsub(/;/, "", $10); gsub(/"/, "", $14); gsub(/;/, "", $14); print $10, $14}}' gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.gtf > gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.stripped.txt'")")}}'
```
