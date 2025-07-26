# Guida Operativa e Teorica Completa: Workflow Classificazione Rischio Criminalità NYC (DMML)
## GUIDA TEORICA E OPERATIVA (per ogni fase)

### Fase 1: Model Selection & Baseline

- **Obiettivo**: Definire i modelli candidati da valutare e stabilire un baseline di performance.
- **Notebook**: `Modeling.ipynb`
- **Riferimento Teorico**: `detailed_summary.md` - Sezione 3 (Algoritmi di Classificazione Principali).
- **Task**:
    - [X] **Definire Baseline**: Implementare un `DummyClassifier` (es. strategia 'stratified' o 'most_frequent') per avere un riferimento minimo di performance. Valutarlo con CV sul training set (come nella Fase 2).
    - [X] **Selezionare Modelli Candidati**: Scegliere una varietà di modelli (es. 2-5) con caratteristiche diverse:
        - Lineare: `LogisticRegression`
        - Basato su Alberi: `DecisionTreeClassifier` (principalmente come base per ensemble)
        - Ensemble: `RandomForestClassifier`, `GradientBoostingClassifier` (o `XGBoost`/`LightGBM` se installati)
        - SVM: `SVC` (con kernel lineare e/o RBF)
        - Naive Bayes: `GaussianNB` (o altri a seconda delle feature)
        - k-NN: `KNeighborsClassifier`
    - [X] **Creare Pipeline**: Per *ogni* modello candidato, creare una `Pipeline` (da `sklearn.pipeline` o `imblearn.pipeline`) che includa:
        - **Scaling**: `StandardScaler` (quasi sempre necessario, specialmente per LR, SVM, kNN).
        - **Altri Preprocessing (Opzionale)**: Se si vogliono testare effetti di PCA, feature selection specifica per modello, etc., includerli nella pipeline.
        - **Classificatore**: L'estimatore finale.
    - [X] **Organizzare Pipeline**: Mettere le pipeline create in un dizionario per una facile iterazione nella fase successiva (es. `pipelines = {'LR': pipeline_lr, 'RF': pipeline_rf, ...}`).

---

### Fase 2: Cross-Validation Candidati

- **Obiettivo**: Valutare e confrontare le performance di tutte le pipeline definite (baseline, modelli base, modelli con gestione sbilanciamento) usando la cross-validation sul training set.
- **Notebook**: `Modeling.ipynb`
- **Riferimento Teorico**: `detailed_summary.md` - Sezioni 4.2 (Metriche) e 4.1 (Strategie di Valutazione - CV).
- **Task**:
    - [X] **Definire Schema CV**: Istanziare `StratifiedKFold` (da `sklearn.model_selection`) con `n_splits` (es. 5 o 10), `shuffle=True`, e `random_state` per riproducibilità. `skf = StratifiedKFold(...)`.
    - [X] **Definire Metriche**: Scegliere le metriche appropriate per la valutazione, specialmente considerando lo sbilanciamento. Usare un dizionario per `scoring`. Metriche raccomandate:
        - `'accuracy'`
        - `'balanced_accuracy'`
        - `'f1_weighted'` (media F1 pesata per supporto classi)
        - `'f1_macro'` (media F1 non pesata, tratta tutte le classi ugualmente)
        - `'f1_minority'` (F1 score specifico per la classe minoritaria - richiede creazione scorer custom, vedi sotto)
        - `'roc_auc_ovr_weighted'` (AUC ROC media pesata, per multi-classe) o `'roc_auc'` (per binario)
        - `'average_precision_weighted'` (AUC PR media pesata) o `'average_precision'` (per binario, spesso più informativa di ROC AUC con sbilanciamento)
        - `'mcc'` (Matthews Correlation Coefficient)
    - [X] **Controllo tier**: Usa come metro di valutazione anche di quanti tier si sbaglia, gli errori inaccettabili sono quelli basso/alto e alto/basso, gli altri sono accettabili.  
    - [X] **Creare Scorer Custom (se necessario)**: Per metriche specifiche per classe (es. F1 sulla minoranza), usare `make_scorer` da `sklearn.metrics`. Esempio: `f1_minority_scorer = make_scorer(f1_score, pos_label=minority_class_label)`. Aggiungere al dizionario `scoring`.
    - [X] **Eseguire Cross-Validation**: Iterare sul dizionario `pipelines`. Per ogni pipeline:
        - Usare `cross_validate` (preferibile a `cross_val_score` per ottenere più metriche e tempi).
        - Passare `X_train`, `y_train`, `cv=skf`, `scoring=scoring_dict`, `n_jobs=-1` (per parallelizzare), `return_train_score=False` (di solito non serve).
        - Gestire eventuali errori (`try-except`).
    - [X] **Salvare Risultati**: Memorizzare i risultati di `cross_validate` (che è un dizionario di array) per ogni pipeline in un altro dizionario (es. `all_cv_outputs`).
    - [X] **Scrivere Risultati su File**: Salvare `all_cv_outputs` (in formato JSON o CSV) al termine di ogni ciclo di valutazione, per poterlo recuperare in futuro.
    - [X] **Caricare Risultati da File**: In caso di ulteriori analisi o valutazioni, importare il file salvato e riassegnarlo a `all_cv_outputs` o a un nuovo dizionario.
    - [X] **Analizzare Risultati Preliminari**: Calcolare la media e la deviazione standard per ogni metrica e per ogni pipeline. Presentare i risultati in una tabella (es. Pandas DataFrame) per un confronto iniziale. Identificare le pipeline più promettenti.
    - [X] **Risolvere bug**: Risolvere il bug dei NaN nelle metriche di valutazione.
    - [X] **Capire per quali modelli è necessario encoding e co e per quali no**: Alcuni modelli non richiedono encoding, mentre altri sì. Ad esempio, `LogisticRegression` e `RandomForestClassifier` non richiedono encoding, mentre `SVC` e `KNeighborsClassifier` sì. Questo è importante per evitare errori durante la valutazione.
    - [X] **Includere nuovi modelli**: Aggiungere nuovi modelli alla pipeline, come `XGBoost` e `LightGBM`, per migliorare le prestazioni del modello. Questi modelli sono noti per la loro velocità e accuratezza.
    - [X] **Sistema split**: Sistemare lo split dei dati per fare in modo che i risultati non siano troppo pompati -> bisogna fare in modo che StratifiedKFold non sia sullo stesso spazio‐tempo su cui hai costruito il target
---

### Fase 3: Analisi Statistica Risultati CV

- **Obiettivo**: Determinare se le differenze osservate nelle performance medie dei modelli durante la CV (Fase 2) sono statisticamente significative o potrebbero essere dovute al caso.
- **Notebook**: `Modeling.ipynb`
- **Riferimento Teorico**: `detailed_summary.md` - Sezione 5.2 (Confronto Modelli Basato su Test Statistici).
- **Task**:
    - [ ] **Identificare Modelli da Confrontare**: Selezionare le coppie di modelli/pipeline più interessanti da confrontare (es. i top 2-3 performer, o un modello con e senza una specifica tecnica di gestione sbilanciamento).
    - [ ] **Recuperare Punteggi per Fold**: Assicurarsi di avere i punteggi della metrica chiave (es. 'test_f1_minority') per *ogni fold* della CV per i modelli selezionati. Questi sono negli array restituiti da `cross_validate` (es. `all_cv_outputs['PipelineName']['test_f1_minority']`).
    - [ ] **Scegliere Test Statistico**:
        - Dato che i modelli sono valutati sugli stessi fold CV, i campioni sono **appaiati**.
        - **Wilcoxon Signed-Rank Test**: Test non parametrico robusto, buona scelta generale quando non si può assumere normalità delle differenze (comune con pochi fold come 5 o 10). Ipotesi nulla (H0): La mediana delle differenze tra i punteggi appaiati è zero. Usare `scipy.stats.wilcoxon`.
        - **t-test Appaiato**: Test parametrico. Richiede che le differenze tra i punteggi appaiati siano approssimativamente normali. Meno robusto di Wilcoxon se l'assunzione è violata. Usare `scipy.stats.ttest_rel`.
    - [ ] **Eseguire Test e Interpretare p-value**:
        - Per ogni coppia selezionata, eseguire il test statistico scelto sui vettori dei punteggi per fold.
        - Confrontare il p-value ottenuto con un livello di significatività alpha (es. 0.05).
        - **Se p-value < alpha**: Rifiutiamo H0. C'è evidenza statistica che le performance dei due modelli sono diverse sulla metrica scelta.
        - **Se p-value >= alpha**: Non rifiutiamo H0. Non c'è sufficiente evidenza statistica per concludere che le performance siano diverse (potrebbero esserlo, ma il test non è abbastanza potente per rilevarlo con questi dati/fold).
    - [ ] **Documentare Risultati**: Riportare i risultati dei test statistici, indicando quali differenze sono significative. Questo aiuta a prendere decisioni più informate su quale modello procedere a ottimizzare.

---

### Fase 4: Hyperparameter Tuning

- **Obiettivo**: Ottimizzare gli iperparametri della/e pipeline selezionata/e (i migliori candidati dalla Fase 2/3) per massimizzare la performance (stimata tramite CV sul training set).
- **Notebook**: `Modeling.ipynb`
- **Riferimento Teorico**: `detailed_summary.md` - Sezione 5.1 (Selezione del Modello e Ottimizzazione Iperparametri).
- **Task**:
    - [ ] **Selezionare Pipeline da Ottimizzare**: Scegliere 1 o 2 pipeline migliori identificate nelle fasi precedenti, basandosi sui risultati CV e sull'analisi statistica.
    - [ ] **Identificare Iperparametri Chiave**: Per ogni pipeline, identificare gli iperparametri più importanti da ottimizzare. Questi includono sia quelli del classificatore (es. `C`, `gamma` per SVC; `n_estimators`, `max_depth` per RandomForest) sia quelli degli step di preprocessing o resampling (es. `k_neighbors` per SMOTE; `n_components` per PCA). Usare la sintassi `nome_step__nome_parametro` (es. `'classifier__C'`, `'sampling__k_neighbors'`).
    - [ ] **Definire Spazio di Ricerca (`param_distributions` o `param_grid`)**:
        - **Randomized Search (`RandomizedSearchCV`)**: Generalmente preferito per efficienza. Definire distribuzioni da cui campionare per ogni iperparametro (es. `loguniform` per `C` e `gamma`, `randint` per `n_estimators`, lista discreta per `kernel`).
        - **Grid Search (`GridSearchCV`)**: Solo se lo spazio di ricerca è piccolo. Definire valori discreti per ogni iperparametro.
    - [ ] **Scegliere Metrica di Ottimizzazione (`refit_metric`)**: Selezionare la singola metrica che si vuole massimizzare durante il tuning (es. `'f1_minority'`, `'average_precision'`, `'mcc'`). Deve essere una delle chiavi del dizionario `scoring_dict` usato in Fase 2.
    - [ ] **Configurare ed Eseguire SearchCV**:
        - Istanziare `RandomizedSearchCV` (o `GridSearchCV`) da `sklearn.model_selection`.
        - Passare: `estimator=pipeline_da_ottimizzare`, `param_distributions` (o `param_grid`), `n_iter` (numero di combinazioni da provare per Random Search, es. 50-100), `cv=skf` (lo stesso schema CV usato prima), `scoring=scoring_dict` (per tracciare tutte le metriche), `refit=refit_metric` (fondamentale! Riaddestra il miglior modello trovato sull'intero `X_train`, `y_train` usando questa metrica per decidere qual è il migliore), `random_state` (per riproducibilità di Random Search), `n_jobs=-1`.
        - Eseguire `.fit(X_train, y_train)`.
    - [ ] **Analizzare Risultati Tuning**:
        - Accedere ai migliori parametri trovati: `search.best_params_`.
        - Accedere al miglior score CV ottenuto durante il tuning (per la metrica `refit`): `search.best_score_`.
        - Accedere alla pipeline riaddestrata con i migliori parametri: `search.best_estimator_`. Questo è il modello finale pronto per la valutazione sul test set.
        - Opzionale: Esaminare `search.cv_results_` (convertire in DataFrame) per vedere le performance di tutte le combinazioni provate per tutte le metriche in `scoring_dict`.
    - [ ] **Salvare Risultati e Miglior Modello**: Salvare `best_params_`, `best_score_` e, soprattutto, `best_estimator_` (vedi Fase 9).

---

### Fase 5: Final Model Training

- **Obiettivo**: Addestrare la pipeline finale (con gli iperparametri ottimizzati) sull'intero set di training (`X_train`, `y_train`).
- **Notebook**: `Modeling.ipynb`
- **Riferimento Teorico**: N/A (è un passo operativo)
- **Task**:
    - [ ] **Se `refit=True` (o `refit='metrica'`) in `SearchCV`**: Questo passo è **già stato fatto automaticamente** da `GridSearchCV` o `RandomizedSearchCV` alla fine del `.fit()`. Il modello finale ottimizzato e addestrato su tutto `X_train`, `y_train` è disponibile nell'attributo `.best_estimator_` dell'oggetto `SearchCV` fittato.
    - [ ] **Se `refit=False` o tuning manuale** (sconsigliato):
        - Creare una nuova istanza della pipeline.
        - Impostare manualmente gli iperparametri migliori trovati (`.set_params(**best_params)`).
        - Addestrare questa pipeline sull'intero `X_train`, `y_train` (`pipeline.fit(X_train, y_train)`).
    - [ ] **Verifica**: Assicurarsi di avere l'oggetto pipeline finale correttamente addestrato (es. `final_model = search.best_estimator_`).

**Nota**: Usare `refit='metrica_scelta'` è la pratica standard e più conveniente.

---

### Fase 6: Test Set Evaluation

- **Obiettivo**: Valutare le performance del modello finale (addestrato e ottimizzato) **una sola volta** sul set di test (`X_test`, `y_test`), che non è mai stato usato durante l'addestramento o il tuning. Questo fornisce una stima realistica di come il modello generalizza a dati nuovi e mai visti.
- **Notebook**: `Modeling.ipynb` (sezione finale) o un notebook dedicato alla valutazione (`Evaluation.ipynb`).
- **Riferimento Teorico**: `detailed_summary.md` - Sezione 4.2 (Metriche di Valutazione).
- **Task**:
    - [ ] **Caricare Modello Finale**: Se non già in memoria, caricare la pipeline salvata dalla directory versionata (`final_model = joblib.load('saved_model/v1.0/final_pipeline.joblib')`).
    - [ ] **Preparare Test Set**: Assicurarsi che `X_test` abbia le stesse feature (nello stesso ordine) usate per addestrare il modello. Se `X_test` è stato caricato da file, verificare le colonne rispetto a `feature_names.json` caricato dalla stessa directory.
    - [ ] **Ottenere Predizioni**:
        - Predizioni di classe: `y_pred_test = final_model.predict(X_test)`.
        - Probabilità (se il modello le supporta e sono utili per metriche come AUC): `y_proba_test = final_model.predict_proba(X_test)`. Estrarre le probabilità per la classe positiva/di interesse (es. `y_score_test = y_proba_test[:, minority_class_label]`).
    - [ ] **Calcolare Metriche Complete**: Calcolare tutte le metriche di valutazione rilevanti confrontando `y_test` (verità) con `y_pred_test` (e `y_score_test` per AUC). Usare le stesse metriche definite in `scoring_dict` (Fase 2) per confronto, ma ora calcolate sul test set.
        - `confusion_matrix`
        - `classification_report(digits=3)` (include Precision, Recall, F1 per classe e medie)
        - `accuracy_score`
        - `balanced_accuracy_score`
        - `matthews_corrcoef`
        - `cohen_kappa_score`
        - `roc_auc_score` (usando `y_score_test`)
        - `average_precision_score` (AUC-PR, usando `y_score_test`)
    - [ ] **Salvare Metriche Test**: Salvare le metriche calcolate in un file (es. `test_set_metrics.json`) nella directory del modello versionato (Fase 9).
    - [ ] **Visualizzare Risultati**:
        - Matrice di Confusione: Usare `ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)` per una visualizzazione chiara (normalizzata e non). Salvare il grafico.
        - Curva ROC: Usare `RocCurveDisplay.from_predictions(y_test, y_score_test)` (o `from_estimator`). Salvare il grafico.
        - Curva Precision-Recall: Usare `PrecisionRecallDisplay.from_predictions(y_test, y_score_test)` (o `from_estimator`). Salvare il grafico.
    - [ ] **Confrontare con CV Score**: Confrontare le metriche ottenute sul test set con le medie ottenute durante la cross-validation (`best_score_` dal tuning o medie dalla Fase 2 per la stessa metrica).
        - Se le performance sul test set sono significativamente peggiori, potrebbe indicare overfitting durante il tuning (anche se si è usata CV) o che il test set ha caratteristiche diverse dal training set (data drift).
        - Se sono simili o leggermente inferiori, è un buon segno di generalizzazione.
    - [ ] **Documentare Performance Finali**: Riportare chiaramente le metriche di performance finali sul test set nel report finale (Fase 11). Questi sono i numeri "ufficiali" da comunicare riguardo alle capacità del modello.

---

### Fase 7: Error Analysis

- **Obiettivo**: Analizzare gli errori commessi dal modello sul test set (False Positives e False Negatives) per capirne i punti deboli e identificare potenziali aree di miglioramento (es. feature engineering, raccolta dati specifici).
- **Notebook**: `Evaluation.ipynb` o `Modeling.ipynb`
- **Riferimento Teorico**: `detailed_summary.md` - Sezione 6.1 (Analisi degli Errori).
- **Task**:
    - [ ] **Identificare Errori**: Creare un DataFrame contenente `X_test`, `y_test` (verità) e `y_pred_test` (predizioni). Filtrare per identificare le istanze specifiche che sono state classificate erroneamente:
        - False Positives (FP): `(y_test == negative_class) & (y_pred_test == positive_class)`
        - False Negatives (FN): `(y_test == positive_class) & (y_pred_test == negative_class)`
        (Adattare `positive_class` e `negative_class` ai propri label, usando il mapping da `class_labels.json`).
    - [ ] **Analizzare Caratteristiche Errori**: Esaminare le feature delle istanze FP e FN. Ci sono pattern comuni? Hanno valori estremi in alcune feature? Appartengono a sottogruppi specifici dei dati? Confrontare le distribuzioni delle feature per FP/FN rispetto alle istanze classificate correttamente (TP/TN). Visualizzare gli errori (es. proiettandoli su uno spazio 2D con PCA/t-SNE).
    - [ ] **Documentare Osservazioni**: Riportare le principali osservazioni dall'analisi degli errori nel report finale (Fase 11).

---

### Fase 8: Model Interpretation

- **Obiettivo**: Comprendere *perché* il modello prende determinate decisioni. Aumenta la fiducia, aiuta il debugging, può rivelare bias e soddisfare requisiti normativi.
- **Notebook**: `Interpretation.ipynb` o `Modeling.ipynb`
- **Riferimento Teorico**: N/A (specifico per tecniche di interpretabilità)
- **Task**:
    - [ ] **Scegliere Metodo(i) di Interpretazione**: La scelta dipende dal tipo di modello:
        - **Modelli Lineari (Logistic Regression)**: Esaminare i coefficienti (`model.coef_`). Coefficienti più grandi (in valore assoluto) indicano maggiore importanza della feature. Il segno indica la direzione dell'effetto sulla probabilità della classe positiva.
        - **Modelli Basati su Alberi (Decision Tree, RandomForest, Gradient Boosting)**: Esaminare l'importanza delle feature (`model.feature_importances_`). Solitamente basata sulla riduzione media dell'impurità (Gini o entropia) o sul numero di volte che una feature è usata per splittare.
        - **Metodi Agnostici al Modello (applicabili a qualsiasi classificatore)**:
            - **Permutation Importance**: Misura il calo di performance del modello quando i valori di una feature vengono mescolati casualmente nel test set. Un calo maggiore indica maggiore importanza. Implementato in `sklearn.inspection.permutation_importance`.
            - **SHAP (SHapley Additive exPlanations)**: Approccio basato sulla teoria dei giochi che assegna a ogni feature un valore SHAP per ogni predizione. Il valore SHAP rappresenta il contributo di quella feature nello spostare la predizione dal valore base (media delle predizioni) alla predizione finale per quell'istanza. Fornisce spiegazioni sia globali (importanza media delle feature) che locali (per singole predizioni). Libreria `shap`.
            - **LIME (Local Interpretable Model-agnostic Explanations)**: Spiega singole predizioni addestrando un modello interpretabile semplice (es. lineare) localmente intorno all'istanza da spiegare. Libreria `lime`.
    - [ ] **Applicare Metodo(i)**: Calcolare e visualizzare l'importanza delle feature (es. bar plot) o i valori SHAP (es. summary plot, dependence plot, force plot per spiegazioni locali).
    - [ ] **Interpretare Risultati**: Analizzare quali feature sono considerate più importanti dal modello. Questo è coerente con la conoscenza del dominio? Ci sono risultati inaspettati?
    - [ ] **Documentare Interpretazione**: Includere i risultati e le visualizzazioni dell'interpretabilità nel report finale (Fase 11).

---

### Fase 9: Salvare Modello Finale & Artefatti

- **Obiettivo**: Salvare la pipeline addestrata finale e tutti gli artefatti necessari per riprodurre i risultati e utilizzare il modello in futuro (es. per deployment).
- **Notebook**: `Modeling.ipynb` o script dedicato (`SaveScript.py`)
- **Riferimento Teorico**: N/A
- **Task**:
    - [ ] **Creare Directory Versionata**: Creare una directory specifica per questa versione del modello (es. `saved_models/v1.0/`).
    - [ ] **Salvare Pipeline Addestrata**: Usare `joblib.dump` (preferito per oggetti scikit-learn con array NumPy) o `pickle.dump` per salvare l'oggetto `final_model` (la pipeline addestrata, es. `search.best_estimator_`). Esempio: `joblib.dump(final_model, 'saved_models/v1.0/final_pipeline.joblib')`.
    - [ ] **Salvare Nomi Feature**: Salvare la lista dei nomi delle feature usate per addestrare il modello (nell'ordine corretto). Essenziale per assicurarsi che i dati in input per le predizioni future abbiano le stesse colonne. Esempio: salvare `X_train.columns` in `feature_names.json`.
    - [ ] **Salvare Mapping Classi**: Se le etichette delle classi sono state codificate numericamente, salvare il mapping tra i numeri e i nomi originali delle classi (es. `{0: 'No_Crime', 1: 'Crime'}`). Esempio: salvare in `class_labels.json`.
    - [ ] **Salvare Metriche**: Salvare le metriche di performance calcolate sul test set (Fase 6) in un file (es. `test_set_metrics.json`).
    - [ ] **Salvare Grafici**: Salvare i grafici generati (matrice confusione, ROC, PR, feature importance, SHAP plots) nella directory versionata.
    - [ ] **Salvare Codice Sorgente (Opzionale ma raccomandato)**: Salvare una copia dello script/notebook usato per addestrare e salvare questa versione del modello.
    - [ ] **Documentazione (README)**: Aggiungere un file `README.md` nella directory versionata che descriva brevemente il modello, la versione, le performance chiave e come caricarlo/usarlo.
    - [ ] **Controllo Versioni (Git)**: Committare la directory del modello salvato in Git (se non troppo grande, altrimenti usare Git LFS o DVC).

---

### Fase 10: Deployment Artifacts Check

- **Obiettivo**: Verificare che tutti gli artefatti salvati (modello, feature names, ecc.) siano corretti, completi e possano essere caricati e utilizzati correttamente in un ambiente pulito.
- **Notebook**: Script dedicato (`CheckScript.py`) o nuova sessione/notebook.
- **Riferimento Teorico**: N/A
- **Task**:
    - [ ] **Caricare Pipeline**: In un ambiente separato (o dopo aver riavviato il kernel), provare a caricare la pipeline salvata usando `joblib.load('saved_models/v1.0/final_pipeline.joblib')`.
    - [ ] **Caricare Artefatti**: Caricare `feature_names.json`, `class_labels.json`, `test_set_metrics.json`.
    - [ ] **Verificare Nomi Feature**: Assicurarsi che la lista di feature caricate corrisponda a quelle attese.
    - [ ] **Testare Predizione**: Creare un piccolo DataFrame di esempio (o usare alcune righe da `X_test`) assicurandosi che abbia le colonne corrette (usando `feature_names.json`). Eseguire `.predict()` e `.predict_proba()` sulla pipeline caricata con questi dati di esempio. Verificare che l'output sia nel formato atteso e non generi errori.
    - [ ] **Verificare Metriche**: Confrontare le metriche caricate da `test_set_metrics.json` con quelle ottenute originariamente.
    - [ ] **Documentare Checklist**: Confermare che tutti i controlli siano stati superati.

---

### Fase 11: Results Synthesis & Report

- **Obiettivo**: Riassumere l'intero processo, i risultati chiave, le interpretazioni, le limitazioni del modello e le possibili direzioni future in un report comprensibile.
- **Notebook**: Notebook finale (`Final_Report.ipynb`) o documento separato (`Report.md`, `Report.pdf`).
- **Riferimento Teorico**: N/A
- **Task**:
    - [ ] **Struttura Report**: Definire una struttura logica (es. Introduzione/Obiettivo, Dati, Preprocessing, Modellazione, Risultati Valutazione, Analisi Errori, Interpretazione, Conclusioni, Limitazioni, Lavoro Futuro, Appendice).
    - [ ] **Descrivere Dati e Preprocessing**: Riassumere le fonti dati, le caratteristiche principali, i passi di pulizia, feature engineering, scaling e selezione effettuati.
    - [ ] **Descrivere Processo Modellazione**: Spiegare i modelli candidati, la strategia di CV, il processo di tuning, e il modello finale selezionato.
    - [ ] **Presentare Risultati Valutazione**: Riportare le metriche finali sul test set (tabella, grafici matrice confusione, ROC, PR). Confrontare con baseline e risultati CV.
    - [ ] **Riportare Analisi Errori**: Descrivere i pattern identificati negli errori FP/FN.
    - [ ] **Riportare Interpretazione**: Mostrare i risultati dell'interpretabilità (feature importance, SHAP) e discuterne le implicazioni.
    - [ ] **Discutere Limitazioni**: Identificare i limiti del modello (es. performance su sottogruppi, assunzioni fatte, qualità dati).
    - [ ] **Proporre Lavoro Futuro**: Suggerire possibili miglioramenti (es. nuove feature, algoritmi diversi, più dati).
    - [ ] **Conclusioni**: Riassumere i risultati principali e l'efficacia del modello rispetto all'obiettivo iniziale.
    - [ ] **Rendere Replicabile**: Assicurarsi che il report includa riferimenti al codice e agli artefatti salvati per permettere la riproducibilità.

---

### Fase 12: Deployment (Basic Example)

- **Obiettivo**: Creare un semplice servizio (es. API REST) che carichi il modello salvato e possa fornire predizioni su nuovi dati.
- **Notebook/Script**: Script Python in una directory `Deploy/` (es. `app.py`).
- **Riferimento Teorico**: N/A (specifico per framework web)
- **Task**:
    - [ ] **Scegliere Framework**: Selezionare un framework web leggero (es. Flask, FastAPI).
    - [ ] **Creare Script API**: Scrivere uno script Python che:
        - Importi le librerie necessarie (Flask/FastAPI, joblib, pandas).
        - Carichi la pipeline del modello (`joblib.load`) e gli artefatti necessari (nomi feature, mapping classi) all'avvio dell'applicazione.
        - Definisca un endpoint API (es. `/predict`) che accetti dati in input (es. JSON nel body della richiesta POST).
        - Preprocessi i dati ricevuti: Si assicuri che i dati siano nel formato corretto (es. DataFrame Pandas) e abbiano le feature attese nell'ordine corretto (usando i `feature_names.json` salvati).
        - Esegua la predizione usando `model.predict(input_data)` o `model.predict_proba(input_data)`.
        - Formatti la risposta (es. JSON con la predizione e/o le probabilità).
        - Includa gestione base degli errori (es. input malformato).
    - [ ] **Creare `requirements.txt`**: Elencare tutte le dipendenze Python necessarie per eseguire l'API (Flask/FastAPI, scikit-learn, joblib, pandas, gunicorn/uvicorn, ecc.).
    - [ ] **Testare Localmente**: Eseguire l'API localmente e inviare richieste di test (es. usando `curl`, Postman, o uno script Python `requests`) per verificare che funzioni correttamente.
    - [ ] **Containerizzazione (Opzionale)**: Creare un `Dockerfile` per containerizzare l'applicazione API, facilitando il deployment.
    - [ ] **Documentare API**: Fornire istruzioni su come eseguire l'API e come inviare richieste (es. nel `README.md` della directory `Deploy/`).
