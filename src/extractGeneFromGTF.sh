genome_gtf=$1
output=$2
cat $genome_gtf | perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' > $output