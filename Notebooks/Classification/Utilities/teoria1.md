# Sommario Dettagliato: Workflow di Classificazione (DMML)

## 4. Valutazione del Modello
### 4.2 Metriche di Valutazione per Classificazione
- **Necessità**: L'accuracy da sola può essere ingannevole, specialmente con classi sbilanciate.
- **Matrice di Confusione**: Tabella che riassume le performance (predizioni vs. valori reali).
    - Per classificazione binaria (Positivo/Negativo):
        - **True Positives (TP)**: Positivi predetti correttamente.
        - **True Negatives (TN)**: Negativi predetti correttamente.
        - **False Positives (FP)**: Negativi predetti erroneamente come Positivi (Errore Tipo I, Falso Allarme).
        - **False Negatives (FN)**: Positivi predetti erroneamente come Negativi (Errore Tipo II, Mancato Rilevamento).
    - Visualizzazione: `ConfusionMatrixDisplay.from_estimator` o `from_predictions`.
- **Metriche Derivate dalla Matrice di Confusione** (`sklearn.metrics`):
    - **Accuracy**: $(TP+TN) / (TP+TN+FP+FN)$. Frazione di predizioni corrette totali. Ingannatoria con classi sbilanciate.
    - **Precision (Positive Predictive Value)**: $TP / (TP+FP)$. Di tutte le predizioni positive, quante erano corrette? Utile quando il costo dei FP è alto.
    - **Recall (Sensitivity, True Positive Rate - TPR)**: $TP / (TP+FN)$. Di tutti i veri positivi, quanti sono stati identificati? Utile quando il costo dei FN è alto.
    - **Specificity (True Negative Rate - TNR)**: $TN / (TN+FP)$. Di tutti i veri negativi, quanti sono stati identificati?
    - **F1-Score**: $2 * (Precision * Recall) / (Precision + Recall)$. Media armonica di Precision e Recall. Buon bilanciamento tra le due.
    - **False Positive Rate (FPR)**: $FP / (FP+TN) = 1 - Specificity$. Frazione di negativi erroneamente classificati come positivi.
- **ROC Curve (Receiver Operating Characteristic)**:
    - Grafico di **TPR (Recall)** vs **FPR** al variare della soglia di classificazione del modello (per modelli che outputtano probabilità o score).
    - Un buon classificatore ha una curva che si avvicina all'angolo in alto a sinistra (TPR=1, FPR=0).
    - La linea diagonale rappresenta un classificatore casuale.
    - Visualizzazione: `RocCurveDisplay.from_estimator` o `from_predictions`.
- **AUC (Area Under the ROC Curve)**:
    - Area sotto la curva ROC. Misura aggregata della performance del classificatore su tutte le possibili soglie.
    - Valore tra 0 e 1. 1 = classificatore perfetto, 0.5 = classificatore casuale.
    - Interpretazione: Probabilità che un classificatore assegni uno score più alto a un'istanza positiva scelta a caso rispetto a un'istanza negativa scelta a caso.
- **Classification Report (`classification_report`)**: Mostra Precision, Recall, F1-Score per ogni classe, più medie.
- **Gestione Multi-Classe**:
    - Le metriche (Precision, Recall, F1) possono essere calcolate per ogni classe (in modo One-vs-Rest).
    - **Medie**: Per ottenere un singolo score su tutte le classi:
        - **`macro`**: Media aritmetica delle metriche per classe (non pesata). Tratta tutte le classi ugualmente.
        - **`weighted`**: Media delle metriche per classe, pesata per il *support* (numero di istanze vere per classe). Tiene conto dello sbilanciamento.
        - **`micro`**: Calcola le metriche globalmente contando TP, FN, FP totali. Equivale all'accuracy.
- **Riferimento**: Notebook `07.1`.

## 5. Selezione e Tuning del Modello

### 5.1 Tuning degli Iperparametri
- **Iperparametri**: Parametri del modello non appresi dai dati, ma impostati prima del training (es. *k* in KNN, `max_depth` in Albero, `C` e `gamma` in SVM).
- **Obiettivo**: Trovare la combinazione di iperparametri che massimizza la performance di generalizzazione (stimata tramite CV).
- **Metodi (`sklearn.model_selection`)**:
    - **`GridSearchCV`**: Esplora esaustivamente tutte le combinazioni di iperparametri specificate in una griglia.
        - **Processo**: Per ogni combinazione, esegue una CV sul training set e calcola lo score medio. Sceglie la combinazione con lo score migliore.
        - **Vantaggi**: Garantisce di trovare la combinazione migliore nella griglia.
        - **Svantaggi**: Computazionalmente costoso se la griglia è grande.
    - **`RandomizedSearchCV`**: Esplora un numero fisso (`n_iter`) di combinazioni scelte casualmente da distribuzioni specificate per ogni iperparametro.
        - **Vantaggi**: Più efficiente di GridSearchCV, spesso trova performance simili o migliori con meno iterazioni, specialmente se solo pochi iperparametri sono importanti.
        - **Svantaggi**: Non garantisce di trovare il massimo assoluto.
- **Importante**: Il tuning deve essere fatto usando solo il training set (tipicamente con CV *interna*). Il test set finale serve solo per la valutazione *finale* del modello scelto e tunato.
- **Pipeline e Tuning**: GridSearchCV/RandomizedSearchCV funzionano perfettamente con le Pipeline, permettendo di tunare iperparametri di qualsiasi step (es. `feature-selection__k`, `clf__C`).
- **Riferimento**: Notebook `11`.

### 5.2 Confronto tra Modelli
- **Necessità**: Valutare se le differenze di performance tra due o più modelli (o configurazioni) sono statisticamente significative o dovute al caso (variabilità nello split dei dati).
- **Approccio**: Usare test statistici sulle performance ottenute tramite CV (assicurandosi che i modelli siano valutati sugli *stessi* fold).
    1. Eseguire CV (es. `StratifiedKFold`) per ogni modello/configurazione da confrontare, usando lo *stesso* oggetto CV per garantire fold identici.
    2. Estrarre le metriche di interesse (es. F1-score) per ogni fold per ciascun modello.
    3. Si ottengono così campioni *accoppiati* (paired samples) delle performance (un valore per fold per ogni modello).
    4. Applicare un test statistico per dati accoppiati.
- **Test Statistici Comuni**: Confronto tra due modelli (A vs B):
    - **t-test Accoppiato (`scipy.stats.ttest_rel`)**: Assume che le *differenze* tra le performance nei fold seguano una distribuzione normale. Testa l'ipotesi nulla che la differenza media sia zero.
    - **Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`)**: Test non parametrico (non assume normalità). Testa l'ipotesi nulla che la *mediana* delle differenze sia zero (o che le distribuzioni delle performance siano le stesse).
- **Interpretazione del p-value**: Probabilità di osservare una differenza di performance (o una statistica test) estrema come quella ottenuta, *assumendo che l'ipotesi nulla sia vera* (cioè, che non ci sia differenza reale tra i modelli).
    - Se **p-value $\leq \alpha$** (soglia di significatività, es. 0.05): Si rifiuta l'ipotesi nulla. La differenza osservata è statisticamente significativa.
    - Se **p-value $> \alpha$**: Non si può rifiutare l'ipotesi nulla. Non c'è evidenza statistica sufficiente per dire che un modello sia migliore dell'altro.
- **Riferimento**: Notebook `07.2`.