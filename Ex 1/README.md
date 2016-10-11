Per generare i dati di training e di test potete usare la classe GenPolyData passando il grado del polinomio target e la std del componente rumore (noise). Poi usate il metodo genera per creare i file di training e test con le coppie. Per esempio:
>>> g = GenPolyData(5,0.1)
>>> g.genera(50,'dati.p5.std01.tr') # genera 50 coppie di train
>>> g.genera(1000,'dati.p5.std01.te') # genera 1000 coppie di test

Quindi potete utilizzare la classe Regressor per l'apprendimento e il test come esemplificato nella funzione do_prova(), vedere il codice.