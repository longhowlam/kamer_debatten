# Tweede Kamer speeches

Alle Tweede Kamer parlementaire debatten van 1995 t/m juni-2019 zijn in een data set verzameld. Dat is het werk van:

Rauh, Christian; Schwalbach, Jan, 2020, "0_RauhSchwalbach_2020_ParlSpeechV2_ReleaseNote.pdf", The ParlSpeech V2 data set: Full-text corpora of 6.3 million parliamentary speeches in the key legislative chambers of nine representative democracies, https://doi.org/10.7910/DVN/L4OAKN/C2TWCZ, Harvard Dataverse, V1

Voor meer informatie [zie deze site](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN).

The original data on the webiste was an R rds file, I imported that in R and saved it as csv, then zipped it. But the zip file was too large to put on GitHub so I used the command line tool split to create 24 Mb chuncks. CorpusTweedeKamera .... CorpusTweedekamerl. You need to put those chuncks to one zip again.

## Analyze mogelijkheden

* Metadata anlyse
* Topic modeling
* Word2Vec
* CLassificatie naar partij
* en veel meer......