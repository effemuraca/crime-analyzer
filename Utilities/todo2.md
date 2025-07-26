# Guida Operativa e Teorica Completa: Workflow Classificazione Rischio Criminalità NYC (DMML)

### 14. Hyperparameter Tuning (Ottimizzazione Iperparametri)
**Teoria:**
- **Iperparametri vs Parametri**: I parametri di un modello vengono appresi dai dati durante il training (es. coefficienti in Logistic Regression, pesi in una rete neurale). Gli iperparametri sono impostazioni dell'algoritmo che vengono scelte *prima* del training e che ne controllano il comportamento (es. `k` in kNN, `C` e `kernel` in SVM, `max_depth` e `criterion` in Decision Tree, `n_estimators` in RandomForest).
- **Obiettivo del Tuning**: Trovare la combinazione di iperparametri che massimizza le performance del modello (stimate tramite cross-validation sul training set) per una data metrica.
- **Metodi**:
    - **Grid Search (`GridSearchCV`)**: Esplora esaustivamente una griglia predefinita di valori per gli iperparametri specificati. Trova l'ottimo *nella griglia*, ma può essere molto lento se la griglia è grande o ci sono molti iperparametri.
    - **Random Search (`RandomizedSearchCV`)**: Campiona un numero fisso (`n_iter`) di combinazioni casuali di iperparametri da distribuzioni specificate. Spesso più efficiente di Grid Search, specialmente quando solo alcuni iperparametri sono importanti, perché esplora una gamma più ampia di valori per ciascuno. Non garantisce di trovare l'ottimo assoluto, ma spesso trova combinazioni molto buone con meno calcoli.
    - **Metodi Avanzati**: Ottimizzazione Bayesiana (es. `scikit-optimize`, `hyperopt`), Successive Halving (`HalvingGridSearchCV`, `HalvingRandomSearchCV`), Hyperband. Questi metodi cercano di allocare il budget computazionale in modo più intelligente, concentrandosi su configurazioni promettenti.
- **Processo con `GridSearchCV`/`RandomizedSearchCV`**:
    1.  Definire il modello (o la **pipeline**!).
    2.  Definire lo spazio di ricerca degli iperparametri (griglia per GridSearch, distribuzioni per RandomSearch).
    3.  Definire lo schema di cross-validation (`cv`) da usare *all'interno* del processo di tuning (sul training set).
    4.  Definire la metrica (`scoring`) da ottimizzare.
    5.  Eseguire `.fit(X_train, y_train)`. L'oggetto Search addestrerà e valuterà il modello con diverse combinazioni di iperparametri usando la CV specificata.
    6.  Accedere ai risultati:
        *   `best_params_`: La combinazione di iperparametri che ha dato il miglior punteggio medio CV.
        *   `best_score_`: Il miglior punteggio medio CV ottenuto.
        *   `best_estimator_`: Il modello (o pipeline) ri-addestrato sull'intero `X_train`, `y_train` usando i `best_params_`. Questo è il modello pronto per la valutazione finale sul test set.
        *   `cv_results_`: Un dizionario con dettagli completi su tutte le combinazioni provate e i relativi punteggi CV.

**Errori comuni:**
- **Data Leakage**: Eseguire il tuning usando dati del test set. Il tuning deve avvenire **esclusivamente sul training set**, usando la cross-validation interna.
- Ottimizzare per una metrica sbagliata (es. accuracy su dati sbilanciati).
- Definire uno spazio di ricerca troppo piccolo (rischio di non trovare buoni iperparametri) o troppo grande (spreco computazionale).
- Non usare una pipeline quando il preprocessing ha iperparametri da ottimizzare (es. `k` in `SelectKBest`, `n_components` in `PCA`). La notazione `<nome_step>__<iperparametro>` permette di ottimizzare anche gli iperparametri degli step della pipeline.
- Interpretare `best_score_` come la performance finale del modello. È una stima ottimistica perché gli iperparametri sono stati scelti per massimizzarla. La vera valutazione richiede il test set separato o Nested CV.

**Best practice:**
- **Usare Pipeline**: Scegliere lo spazio di ricerca per gli iperparametri di *tutti* gli step rilevanti della pipeline.
- **Iniziare con Random Search**: Spesso più efficiente per una prima esplorazione ampia dello spazio. Si può poi usare Grid Search per raffinare la ricerca attorno alle regioni promettenti trovate da Random Search.
- **Scegliere `scoring` appropriato**: Basarsi sugli obiettivi del progetto e sulle caratteristiche dei dati (es. 'f1_weighted', 'roc_auc', 'average_precision' per dati sbilanciati).
- **Usare `n_jobs=-1`**: Per parallelizzare la ricerca su tutti i core CPU disponibili.
- **Analizzare `cv_results_`**: Può dare insight su come gli iperparametri influenzano le performance e la stabilità (deviazione standard dei punteggi CV).

**Nested Cross-Validation (per valutazione robusta del processo di tuning):**
Se l'obiettivo è stimare in modo non distorto le performance che ci si può aspettare dal *processo completo* (tuning + training), si usa Nested CV.
- **Ciclo Esterno (Outer CV)**: Divide il dataset in K fold per la *valutazione*.
- **Ciclo Interno (Inner CV)**: Per ogni fold di training del ciclo esterno, si esegue un'intera procedura di hyperparameter tuning (es. GridSearchCV/RandomizedSearchCV) usando un'ulteriore cross-validation *solo su quel fold di training*.
- Si ottiene un punteggio sul fold di test del ciclo esterno usando il modello ottimizzato nel ciclo interno corrispondente.
- La media dei punteggi ottenuti sui fold di test del ciclo esterno è la stima non distorta delle performance.
È computazionalmente molto costoso.

### 15. Final Model Training
**Teoria:**
- Dopo aver selezionato il modello (o pipeline) migliore e aver trovato la combinazione ottimale di iperparametri tramite cross-validation e hyperparameter tuning (Fasi 12-14) sul **training set**, l'ultimo passo prima della valutazione finale è addestrare questo modello finale.
- **Come**: Si addestra la pipeline scelta, configurata con i migliori iperparametri trovati, sull'**intero training set** (`X_train`, `y_train`).
- **`refit=True`**: Se si usa `GridSearchCV` o `RandomizedSearchCV` con l'impostazione predefinita `refit=True` (consigliato), l'oggetto Search fa questo automaticamente. Dopo `.fit(X_train, y_train)`, l'attributo `best_estimator_` conterrà già la pipeline finale addestrata sull'intero training set con i parametri ottimali. Questo è il modello pronto per la valutazione finale sul test set.
- **Salvataggio del Modello**: Questo è il momento di salvare (serializzare) la pipeline addestrata (`best_estimator_`) su disco per poterla ricaricare in seguito per fare predizioni su nuovi dati o per il deployment, senza dover rieseguire l'intero processo di training e tuning.

**Errori comuni:**
- Dimenticare di ri-addestrare il modello sull'intero training set dopo il tuning (se non si usa `refit=True`).
- Addestrare il modello finale includendo anche il test set.
- Non salvare la pipeline completa (inclusi scaler, PCA, encoder, ecc.), ma solo il classificatore finale. Questo rende impossibile applicare correttamente il modello a nuovi dati grezzi.

**Best practice:**
- Usare `refit=True` (default) in `GridSearchCV`/`RandomizedSearchCV`. Il modello finale sarà `search.best_estimator_`.
- Salvare l'**intera pipeline** usando librerie come `joblib` o `pickle`. `joblib` è spesso preferito per oggetti `scikit-learn` che contengono grandi array `numpy`.
- Versionare il modello salvato insieme al codice che lo ha generato.

### 16. Test Set Evaluation
**Teoria:**
- **Scopo**: Valutare le performance di generalizzazione del modello finale (addestrato sull'intero training set con i parametri ottimali) su dati **mai visti prima**, simulando l'uso del modello in produzione. Questo fornisce la stima più realistica delle performance attese su nuovi dati.
- **Quando**: Questa è l'**ultima** fase di valutazione. Il test set (`X_test`, `y_test`) deve essere usato **solo una volta** a questo punto. Non deve MAI essere usato per tuning, selezione del modello o qualsiasi altra decisione che influenzi il modello stesso.
- **Come**:
    1.  Usare il modello finale addestrato (`final_model_pipeline` dalla Fase 15, che è `search.best_estimator_` se si usa `refit=True`).
    2.  Fare predizioni sul test set: `y_pred = final_model_pipeline.predict(X_test)`.
    3.  Se necessario per alcune metriche (AUC, ROC, Precision-Recall curve), ottenere gli score o le probabilità: `y_score = final_model_pipeline.predict_proba(X_test)[:, 1]` (per la classe positiva in caso binario).
    4.  Calcolare le metriche di valutazione desiderate confrontando `y_pred` (e `y_score`) con le etichette reali `y_test`. Usare le stesse metriche considerate durante la CV e il tuning per coerenza, ma focalizzarsi su quelle più rilevanti per il problema (es. F1, Recall sulla classe minoritaria, AUC-PR).
- **Metriche da Calcolare**:
    - `accuracy_score`
    - `confusion_matrix`
    - `classification_report` (fornisce Precision, Recall, F1 per classe e medie macro/weighted)
    - `roc_auc_score` (se applicabile, richiede `y_score`)
    - `average_precision_score` (AUC-PR, se applicabile, richiede `y_score`)
    - `matthews_corrcoef` (MCC)
    - `cohen_kappa_score`
- **Visualizzazioni**:
    - `ConfusionMatrixDisplay`
    - `RocCurveDisplay`
    - `PrecisionRecallDisplay`

**Errori comuni:**
- Usare il test set più volte (es. per provare diverse soglie di classificazione, per confrontare modelli dopo averli già valutati). Ogni "sbirciatina" al test set invalida la sua funzione di valutazione finale imparziale.
- Riportare solo l'accuracy, specialmente per dati sbilanciati.
- Confrontare i punteggi sul test set con i punteggi della cross-validation (`best_score_` da GridSearchCV/RandomizedSearchCV). È normale che il punteggio sul test set sia leggermente inferiore, poiché `best_score_` è ottimizzato sulla CV.

**Best practice:**
- Isolare il test set fin dall'inizio e non usarlo fino a questo momento.
- Calcolare un set completo di metriche rilevanti per il problema.
- Visualizzare la confusion matrix e le curve ROC/PR per una comprensione più profonda.
- Riportare i risultati sul test set in modo chiaro e trasparente, specificando il modello finale e gli iperparametri usati.
- Confrontare le performance sul test set con quelle del modello baseline.

### 17. Error Analysis
**Teoria:**
- **Scopo**: Andare oltre le metriche aggregate e analizzare *quali* errori commette il modello finale. Capire *perché* il modello sbaglia su certe istanze può fornire insight preziosi per migliorarlo ulteriormente (es. suggerendo nuove feature, modificando il preprocessing, raccogliendo più dati di un certo tipo).
- **Come**:
    - **Analisi della Confusion Matrix**: Esaminare nel dettaglio i numeri di TP, TN, FP, FN per ogni classe. Quali classi vengono confuse più spesso? Ci sono errori asimmetrici (es. più FN che FP per una classe)?
    - **Ispezione degli Errori**: Identificare le istanze specifiche nel test set che sono state misclassificate (FP e FN).
    - **Analisi delle Istanze Misclassificate**: Esaminare le caratteristiche (valori delle feature) delle istanze misclassificate. Hanno qualcosa in comune? Sono vicine ai confini decisionali? Hanno valori insoliti o mancanti per alcune feature? Il modello è poco confidente nelle sue predizioni per queste istanze (se `predict_proba` è disponibile)?
    - **Confronto con Esperti**: Se possibile, discutere gli errori con esperti del dominio per capire se sono "ragionevoli" o se indicano problemi specifici nel modello o nei dati.
    - **Visualizzazione**: Se possibile (es. dati a bassa dimensionalità o usando tecniche come t-SNE/UMAP), visualizzare le istanze misclassificate nello spazio delle feature per vedere se formano cluster o si trovano in regioni specifiche.

**Errori comuni:**
- Fermarsi alle metriche aggregate senza investigare gli errori.
- Analizzare gli errori sul training set invece che sul test set (può dare indicazioni sull'overfitting, ma non sulla generalizzazione).

**Best practice:**
- Creare un DataFrame contenente `X_test`, `y_test`, `y_pred` (e `y_score` se disponibile).
- Filtrare questo DataFrame per identificare le istanze FP e FN.
- Analizzare le statistiche descrittive e le distribuzioni delle feature per i sottoinsiemi di errori rispetto ai sottoinsiemi classificati correttamente.
- Ordinare gli errori per "confidenza" del modello (es. istanze FN con alta probabilità predetta per la classe sbagliata) per focalizzarsi sui casi più problematici.
- Documentare i pattern di errore identificati e le possibili cause/soluzioni.

### 18. Model Interpretation (Interpretabilità del Modello)
**Teoria:**
- **Perché**: Capire *perché* un modello fa certe predizioni è cruciale per:
    - **Fiducia e Affidabilità**: Verificare che il modello abbia imparato pattern sensati e non si basi su artefatti o bias nei dati.
    - **Debugging**: Identificare problemi nel modello o nei dati analizzando le spiegazioni.
    - **Comunicazione**: Spiegare le decisioni del modello agli stakeholder (esperti di dominio, utenti finali, enti regolatori).
    - **Miglioramento**: Ottenere insight per guidare il feature engineering o la raccolta di nuovi dati.
- **Interpretabilità Globale vs Locale**:
    - **Globale**: Spiega il comportamento generale del modello. Quali sono le feature più importanti *in media*? Come influisce una feature sulla predizione *in generale*?
    - **Locale**: Spiega una singola predizione. Perché il modello ha classificato *questa specifica istanza* in questo modo?
- **Metodi (dipendono dal tipo di modello)**:
    - **Modelli Intrinsecamente Interpretabili**:
        - `LogisticRegression`: I coefficienti (`model.coef_`) indicano l'importanza e la direzione dell'influenza di ogni feature (dopo scaling).
        - `Decision Trees`: Il percorso dalla radice alla foglia per una predizione è una spiegazione diretta. L'attributo `feature_importances_` (basato su Gini o Information Gain) dà un'importanza globale. Si può visualizzare l'albero (`plot_tree`) o esportare le regole (`export_text`).
    - **Modelli Complessi (Black Box - es. Ensemble, SVM con kernel, Reti Neurali)**: Richiedono tecniche post-hoc.
        - **Feature Importance (Globale)**:
            - Per modelli basati su alberi (Random Forest, Gradient Boosting): L'attributo `feature_importances_` è disponibile (solitamente basato sulla riduzione media dell'impurità o "mean decrease impurity").
            - **Permutation Importance**: Metodo model-agnostic. Misura l'importanza di una feature rimescolando casualmente i suoi valori nel test set e osservando quanto peggiora la performance del modello. Più affidabile di MDI, specialmente con feature correlate. `sklearn.inspection.permutation_importance`.
        - **SHAP (SHapley Additive exPlanations) (Locale e Globale)**: Metodo model-agnostic basato sulla teoria dei giochi cooperativi (valori di Shapley). Fornisce una misura equa del contributo di ogni feature a una *singola predizione* (spiegazione locale). Aggregando i valori SHAP su tutto il dataset, si ottiene anche un'indicazione dell'importanza globale delle feature e della direzione del loro effetto. Libreria `shap`. Molto potente e popolare.
        - **LIME (Local Interpretable Model-agnostic Explanations) (Locale)**: Metodo model-agnostic che spiega una singola predizione approssimando il modello complesso con un modello interpretabile semplice (es. regressione lineare) in un intorno locale dell'istanza da spiegare. Libreria `lime`.

**Errori comuni:**
- Affidarsi ciecamente all'importanza delle feature basata sull'impurità (MDI) per modelli ad albero, che può essere distorta da feature ad alta cardinalità o correlate. Preferire Permutation Importance o SHAP.
- Interpretare i coefficienti della regressione logistica senza aver prima scalato le feature.
- Non validare le spiegazioni con la conoscenza del dominio.

**Best practice:**
- Usare metodi appropriati per il tipo di modello.
- Per modelli complessi, usare SHAP o Permutation Importance per l'importanza globale.
- Usare SHAP o LIME per spiegare predizioni individuali critiche o sorprendenti.
- Visualizzare i risultati dell'interpretabilità (es. SHAP summary plot, dependence plot, force plot).
- Collegare le scoperte dell'interpretabilità all'analisi degli errori (Fase 17) e alla conoscenza del dominio.

### 19. Results Synthesis (Sintesi dei Risultati)
**Teoria:**
- **Scopo**: Riassumere e comunicare i risultati chiave del progetto in modo chiaro e conciso agli stakeholder. Non si tratta solo di riportare numeri, ma di interpretarli nel contesto del problema.
- **Elementi Chiave**:
    - **Riepilogo del Problema e Obiettivi**: Ricordare brevemente cosa si voleva risolvere.
    - **Modello Finale Scelto**: Specificare l'algoritmo, la pipeline utilizzata e gli iperparametri ottimali.
    - **Performance sul Test Set**: Riportare le metriche di valutazione finali più importanti (non solo accuracy!), idealmente confrontandole con il baseline e con i risultati della CV (per dare un'idea dell'ottimismo della CV).
    - **Interpretabilità**: Riassumere quali sono le feature più importanti secondo il modello e come influenzano la predizione (dai risultati della Fase 18). Collegare queste scoperte alla conoscenza del dominio.
    - **Analisi degli Errori**: Discutere i principali pattern di errore osservati (dalla Fase 17). Dove sbaglia il modello? Ci sono bias evidenti?
    - **Limiti del Modello**: Essere onesti sui limiti. Quali assunzioni sono state fatte? Quali aspetti del problema il modello non cattura bene? Qual è l'incertezza associata alle predizioni?
    - **Conclusioni e Prossimi Passi**: Riassumere il valore pratico del modello. Suggerire possibili miglioramenti futuri (es. raccogliere più dati, provare feature diverse, esplorare modelli più complessi, migliorare la gestione degli errori specifici).

**Errori comuni:**
- Riportare solo le metriche positive e ignorare i limiti o gli errori.
- Non contestualizzare i risultati rispetto agli obiettivi iniziali e al dominio.
- Usare linguaggio troppo tecnico senza spiegazioni per stakeholder non tecnici.

**Best practice:**
- Strutturare la sintesi in modo logico.
- Usare visualizzazioni (grafici delle performance, importanza feature) per supportare la narrazione.
- Essere chiari, concisi e onesti.
- Adattare il livello di dettaglio e il linguaggio al pubblico di destinazione.
- Includere la sintesi come sezione finale del report o del notebook principale del progetto.

### 20. Deployment/Artifacts (Deployment e Artefatti)
**Teoria:**
- **Scopo**: Rendere il modello addestrato utilizzabile per fare predizioni su nuovi dati, al di fuori dell'ambiente di sviluppo. Questo può significare integrarlo in un'applicazione, un'API web, un sistema di reporting, ecc.
- **Artefatti Chiave da Salvare**: Non basta salvare solo il classificatore! Per poter processare nuovi dati grezzi nello stesso modo in cui sono stati processati i dati di training, è necessario salvare l'**intera pipeline** che include:
    - Tutti i passaggi di preprocessing (imputer, encoder, scaler, PCA, selettori di feature, ecc.) fittati sul training set.
    - Il modello di classificazione finale addestrato con i parametri ottimali.
- **Serializzazione**: Processo di conversione dell'oggetto pipeline (in memoria) in un formato salvabile su disco.
    - `joblib`: Spesso raccomandato per oggetti `scikit-learn`, più efficiente con grandi array `numpy`.
    - `pickle`: Modulo standard Python, più generale ma potenzialmente meno efficiente per `scikit-learn`.
- **Altri Artefatti Utili**:
    - **Mapping di Encoding**: Se si usano encoder (es. `OrdinalEncoder`), salvare il mapping tra categorie originali e valori numerici. `OneHotEncoder` lo gestisce internamente se si usa `handle_unknown`.
    - **Nomi delle Feature**: Salvare l'elenco e l'ordine delle feature attese dalla pipeline.
    - **Soglie**: Se si applica una soglia specifica alle probabilità predette per decidere la classe finale, salvare questa soglia.
    - **Codice di Training**: Versionare il codice che ha generato il modello salvato.
    - **Requirements**: Salvare le versioni esatte delle librerie usate (`requirements.txt`).
    - **Metadati**: Salvare informazioni sul modello (data di training, dataset usato, metriche sul test set, versione del codice).

**Errori comuni:**
- Salvare solo il classificatore finale senza i passaggi di preprocessing. Questo rende impossibile usare il modello su nuovi dati grezzi.
- Non salvare le versioni delle librerie, portando a problemi di compatibilità quando si ricarica il modello in un ambiente diverso.
- Non versionare il modello salvato insieme al codice.

**Best practice:**
- **Salvare l'intera pipeline addestrata** (`search.best_estimator_` o `final_model_pipeline`).
- Usare `joblib` per la serializzazione.
- Creare uno script o una funzione per caricare la pipeline e fare predizioni su nuovi dati, assicurandosi che l'input venga formattato correttamente (stesse colonne, stessi tipi).
- Versionare tutti gli artefatti (modello, codice, requirements) insieme.
- Considerare l'uso di piattaforme MLOps (es. MLflow, Kubeflow) per gestire il ciclo di vita del modello (tracking esperimenti, versionamento, deployment).

### 21. Documentation/Reporting (Documentazione e Reporting)
**Teoria:**
- **Scopo**: Documentare in modo completo e chiaro l'intero progetto, dalle fasi iniziali alle conclusioni. Essenziale per la riproducibilità, la manutenibilità, la collaborazione e la comunicazione dei risultati.
- **Contenuti Chiave**:
    - **Introduzione**: Descrizione del problema, obiettivi, dati utilizzati, metriche di successo.
    - **Metodologia**: Descrizione dettagliata di ogni fase del workflow:
        - Data Cleaning (operazioni eseguite, motivazioni).
        - EDA (principali scoperte, visualizzazioni chiave).
        - Feature Engineering (feature create, logica).
        - Preprocessing (encoding, scaling, PCA - metodi e parametri).
        - Gestione Sbilanciamento (tecnica usata).
        - Model Selection (modelli provati, motivazioni).
        - Cross-Validation (schema CV, metrica/he usata/e).
        - Hyperparameter Tuning (metodo, spazio di ricerca, metrica ottimizzata).
        - Modello Finale (algoritmo, iperparametri finali).
    - **Risultati**:
        - Performance della CV sui modelli candidati.
        - Risultati del tuning.
        - Valutazione dettagliata del modello finale sul test set (metriche, confusion matrix, curve).
        - Analisi degli errori.
        - Risultati dell'interpretabilità (importanza feature, spiegazioni locali/globali).
    - **Sintesi e Conclusioni**: (Vedi Fase 19). Discussione dei risultati, limiti, valore pratico, prossimi passi.
    - **Appendice (Opzionale)**: Codice sorgente (o link a repository), dettagli implementativi, configurazioni ambiente.
- **Formato**: Può essere un report scritto (es. PDF, Markdown), un notebook Jupyter ben commentato, una presentazione, o una combinazione di questi.

**Errori comuni:**
- Documentazione incompleta o superficiale.
- Non spiegare le motivazioni dietro le scelte metodologiche.
- Non aggiornare la documentazione dopo modifiche al codice o alla metodologia.
- Documentazione disorganizzata o difficile da seguire.

**Best practice:**
- **Documentare man mano**: Scrivere commenti nel codice e note nei notebook durante lo sviluppo, non solo alla fine.
- **Essere specifici**: Includere parametri, `random_state`, versioni delle librerie.
- **Usare visualizzazioni**: Grafici e tabelle rendono il report più leggibile e impattante.
- **Strutturare bene il report/notebook**: Usare titoli, sezioni, markdown per migliorare la leggibilità.
- **Revisionare**: Rileggere la documentazione per assicurarsi che sia chiara, corretta e completa.
- **Mantenere aggiornato questo file guida (`project_modeling_structure_template.md`)** man mano che le fasi vengono implementate o modificate.

### 22. Chrome Extension (Schema Logico)
**Teoria:**
- **Scopo**: Integrare il modello di classificazione in un'estensione Chrome per fornire predizioni di rischio in tempo reale (o quasi) basate su input utente o contesto di navigazione.
- **Flusso Logico Potenziale**:
    1.  **Input Utente/Contesto**: L'utente fornisce un indirizzo/luogo tramite l'estensione, oppure l'estensione rileva automaticamente la posizione da una pagina web (richiede permessi appropriati).
    2.  **Geocoding (se necessario)**: Convertire l'indirizzo in coordinate geografiche (latitudine, longitudine) usando un servizio esterno (es. OpenStreetMap Nominatim, Google Geocoding API - attenzione ai limiti di utilizzo e costi).
    3.  **Feature Extraction**:
        *   Ottenere le feature necessarie per il modello corrispondenti alle coordinate/luogo. Questo è il passo più complesso e dipende da come è stato addestrato il modello. Potrebbe richiedere:
            *   Interrogare API esterne (meteo, eventi, ecc.).
            *   Accedere a database pre-calcolati (es. densità POI per griglia, dati demografici per zona).
            *   Calcolare feature al volo (es. distanza da punti noti).
            *   Ottenere l'ora corrente per le feature temporali.
    4.  **Preprocessing**: Applicare **esattamente la stessa pipeline di preprocessing** usata durante l'addestramento ai dati estratti. Questo include:
        *   Gestione valori mancanti (se applicabile all'input).
        *   Encoding delle feature categoriche.
        *   Scaling delle feature numeriche.
        *   Applicazione PCA (se usata).
        *   **Importante**: Usare gli oggetti (scaler, encoder, PCA) salvati dalla Fase 15/20, **non fittarli di nuovo!**
    5.  **Predizione**: Caricare la pipeline del modello finale salvata (Fase 20) e chiamare `.predict()` (per ottenere la classe di rischio) e/o `.predict_proba()` (per ottenere la confidenza) sui dati preprocessati.
    6.  **Output Utente**: Visualizzare il risultato (es. livello di rischio 'basso', 'medio', 'alto', magari con un indicatore colorato o un punteggio di confidenza) nell'interfaccia dell'estensione.
- **Considerazioni Tecniche**:
    - **Performance**: L'estrazione delle feature e la predizione devono essere abbastanza veloci per non bloccare l'esperienza utente. Modelli complessi o feature che richiedono chiamate API lente potrebbero essere problematici.
    - **Offline/Online**: Il modello e i trasformatori possono essere inclusi nell'estensione per predizioni offline? O è necessaria una chiamata a un backend API dove risiede il modello (approccio più comune per modelli grandi o che richiedono dati aggiornati)?
    - **Sicurezza**: Gestire chiavi API, dati utente e comunicazioni con backend in modo sicuro.
    - **Aggiornamenti**: Come aggiornare il modello nell'estensione o nel backend quando viene riaddestrato?

**Errori comuni:**
- Non applicare esattamente lo stesso preprocessing dell'addestramento.
- Dimenticare di gestire casi limite o input non validi dall'utente.
- Sottostimare la complessità dell'estrazione delle feature in tempo reale.
- Non considerare i limiti di utilizzo delle API esterne.

**Best practice:**
- Progettare attentamente il flusso dati e l'architettura (offline vs online).
- Incapsulare la logica di preprocessing e predizione in funzioni riutilizzabili.
- Gestire gli errori in modo robusto (es. fallimento API, input non valido).
- Testare l'estensione in varie condizioni.
- Documentare l'architettura e il flusso dati dell'estensione.

---
*Checklist e Guida aggiornata al: 29/04/2025*
```
