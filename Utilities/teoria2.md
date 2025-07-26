# Riepilogo Dettagliato: Teoria della Classificazione (Basato su 4-Classification.txt)

## 9. Valutazione e Selezione del Modello (Slide 135-166)

*   **Scopo**: Stimare quanto bene un classificatore si comporterà su dati non visti. Selezionare il modello migliore tra le alternative.\
*   **Metriche per la Valutazione delle Prestazioni**: Basate su una Matrice di Confusione per un problema a due classi (Positivo vs. Negativo):\
    *   **Veri Positivi (TP - True Positives)**: Tuple positive classificate correttamente come positive.\
    *   **Veri Negativi (TN - True Negatives)**: Tuple negative classificate correttamente come negative.\
    *   **Falsi Positivi (FP - False Positives)**: Tuple negative classificate erroneamente come positive (Errore di Tipo I).\
    *   **Falsi Negativi (FN - False Negatives)**: Tuple positive classificate erroneamente come negative (Errore di Tipo II).\
    *   **Accuratezza (Accuracy)**: `(TP + TN) / (Totale)` - Correttezza generale. Può essere fuorviante per dataset sbilanciati.\
    *   **Tasso di Errore (Error Rate)**: `(FP + FN) / (Totale) = 1 - Accuratezza`.\
    *   **Sensibilità (Sensitivity) / Recall / Tasso di Veri Positivi (TPR)**: `TP / (TP + FN)` - Frazione di positivi reali identificati correttamente.\
    *   **Specificità (Specificity) / Tasso di Veri Negativi (TNR)**: `TN / (TN + FP)` - Frazione di negativi reali identificati correttamente.\
    *   **Precisione (Precision)**: `TP / (TP + FP)` - Frazione di predizioni positive che erano effettivamente positive.\
    *   **F1-Score (o F-measure)**: Media armonica di Precisione e Recall: `2 * (Precisione * Recall) / (Precisione + Recall)`. Utile per classi sbilanciate.\
*   **Metodi per la Stima dell'Accuratezza**: Come dividere i dati per training e test.\
    *   **Holdout**: Divide i dati in set di training e test (es. 2/3 training, 1/3 test). Semplice, ma la stima può variare a seconda della divisione.\
    *   **Cross-Validation (k-fold)**:\
        *   Divide i dati in `k` sottoinsiemi (fold) di dimensioni uguali.\
        *   Esegue `k` iterazioni: in ogni iterazione, usa `k-1` fold per il training e 1 fold per il test.\
        *   L'accuratezza complessiva è la media delle accuratezze delle `k` iterazioni.\
        *   *Leave-One-Out*: Caso speciale di k-fold dove `k` è il numero di tuple. Costoso.\
        *   *Stratified Cross-Validation*: Assicura che la distribuzione delle classi sia mantenuta in ogni fold.\
    *   **Bootstrap**: Campiona i dati di training *con* reinserimento (`d` volte, dove `d` è la dimensione del dataset). Le tuple non campionate formano il test set (circa il 36.8% dei dati originali). Ripetuto più volte per ottenere una stima stabile.\
*   **Selezione del Modello**: Confronto tra diversi modelli (o configurazioni dello stesso modello).\
    *   **Curve ROC (Receiver Operating Characteristic)**:\
        *   Visualizza le prestazioni di un classificatore binario al variare della soglia di discriminazione.\
        *   Asse Y: Tasso di Veri Positivi (TPR / Recall / Sensibilità).\
        *   Asse X: Tasso di Falsi Positivi (FPR = FP / (FP + TN) = 1 - Specificità).\
        *   Ogni punto sulla curva rappresenta una coppia (FPR, TPR) per una specifica soglia.\
        *   Il classificatore ideale è nell'angolo in alto a sinistra (FPR=0, TPR=1).\
        *   La linea diagonale (TPR=FPR) rappresenta una classificazione casuale.\
        *   **AUC (Area Under the Curve)**: Misura aggregata delle prestazioni su tutte le soglie. Valore tra 0 e 1. Più vicino a 1 è meglio. Un AUC di 0.5 indica prestazioni casuali.\
        *   Utile per confrontare modelli indipendentemente dalla soglia e per classi sbilanciate.\
    *   **Test di Significatività Statistica**: Per determinare se la differenza di accuratezza tra due modelli è statisticamente significativa o dovuta al caso (es. t-test).\
    *   **Costo/Beneficio**: Considera i costi associati a diversi tipi di errori (es. FP vs. FN). Le matrici di costo possono essere utilizzate per selezionare il modello che minimizza il costo atteso.\