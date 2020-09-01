# Dutch Parlement speeches

![](images/kamer.png)

Alle Tweede Kamer parlementaire debatten van januari-1995 t/m juni-2019 zijn in een data set verzameld. Dat is het werk van:

**Rauh, Christian; Schwalbach**, Jan, 2020, "0_RauhSchwalbach_2020_ParlSpeechV2_ReleaseNote.pdf", The ParlSpeech V2 data set: Full-text corpora of 6.3 million parliamentary speeches in the key legislative chambers of nine representative democracies, https://doi.org/10.7910/DVN/L4OAKN/C2TWCZ, Harvard Dataverse, V1

Voor meer informatie [zie deze site](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN).

De data was verzameld in een R rds file. Deze heb ik in een csv file gezet en gezipped. De zip is te groot om in zijn geheel op GitHub te zetten, dus heb ik de command line tool `split` gebruikt om 24 Mb chuncks te maken: CorpusTweedeKamera .... CorpusTweedekamerl. Deze kan je weer concatenaten tot 1 zip en inlezen.

De data set bestaat uit ruim 1.1 mln. regels. Elke regel bevat de naam van de partij, de naam van de spreker, de datum en tekst.

![](images/data.png)

### **Analyze mogelijkheden**

Er zijn tal van analyze mogelijkheden op deze data set:

* Metadata analyse
* Topic modeling
* Word2Vec
* Text Classificatie model (Target is politieke partij)
* TF Sentence Embedding
* Transformers text generation, [see here](https://github.com/huggingface/transformers)
* en nog andere dingen

***

***

<br/>

<br/>


# Metadata analyse

### **dagelijkse aantallen**

De parlement-speech data beslaat een periode van januari 1995 tot en met juni 2019. Onderstaande grafiek geeft het aantal speeches per dag aan.

![](images/aantal_per_dag.png)

Het schommelt rond de 500 per dag in de begin jaren waarna het iets oploopt.


### **top sprekers**

De top sprekers worden weergegeven in de volgende grafiek. Het zijn Rutte, Halsema en Pechthold, gevolgd door anderen.....

![](images/per_spreker.png)


### **gemiddeld aantal woorden per partij**

Als we kijken naar het gemiddeld aantal woorden van de speeches per partij zien we de 'gelovige' partijen SGP en GPV er boven uitsteken met 211 en 214 woorden gemiddeld.  

![](images/aantal_woorden.png)

