# Dutch Parlement speeches

![](images/kamer.png)

Alle Tweede Kamer parlementaire debatten van 1995 t/m juni-2019 zijn in een data set verzameld. Dat is het werk van:

**Rauh, Christian; Schwalbach**, Jan, 2020, "0_RauhSchwalbach_2020_ParlSpeechV2_ReleaseNote.pdf", The ParlSpeech V2 data set: Full-text corpora of 6.3 million parliamentary speeches in the key legislative chambers of nine representative democracies, https://doi.org/10.7910/DVN/L4OAKN/C2TWCZ, Harvard Dataverse, V1

Voor meer informatie [zie deze site](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN).

De data was verzameld in een R rds file. Deze heb ik in een csv file gezet en gezipped. De zip is te groot om in zijn geheel op GitHub te zetten, dus heb ik de command line tool `split` gebruikt om 24 Mb chuncks te maken: CorpusTweedeKamera .... CorpusTweedekamerl. Deze kan je weer concatenaten tot 1 zip en inlezen.


## Analyze mogelijkheden

Er zijn tal van analyze mogelijkheden op deze data set:

* Metadata analyse
* Topic modeling
* Word2Vec
* Text Classificatie model (Target is politieke partij)
* TF Sentence Embedding
* Transformers text generation, [see here](https://github.com/huggingface/transformers)
* en nog andere dingen

## Metadata analyse
