## Model card: which clients to call for term deposit

### Purpose  
Bank wants to sell term deposits by phone. Call centre can only reach 10% of the list, so they need to pick the right people. This model scores every client before any call. We just call the top 10% with the highest scores. Goal is simple: among the people we call, as many as possible should say yes.

### Data  
I used the UCI Bank Marketing data (calls from 2008–2010). Training set has 32 940 clients, only 11.3% said yes. Test set has 8 236 clients with no answers. Test file has no `duration` column (how long the call lasted). Makes sense — we only know that after the call. Since the model has to work before calling, I also removed `duration` from training. Otherwise it would cheat.

Some columns have “unknown”. Biggest issue is `default` — about 21% unknown. Education 4.3%, housing and loan 2.4% each, job and marital under 1%. `pdays` uses 999 to mean “never contacted before” (it’s a code, not real days).

### How the model works  
One scikit-learn pipeline, so a raw row from test.csv goes straight in. Numbers: fill with median + scale. Text columns: fill with most common value + one-hot. I kept “unknown” as its own category because in `default` it probably means the bank just doesn’t know, and that can be useful. Everything learns only from the training part of each fold — no leak. Final model is logistic regression with C = 0.1, refitted on full train.csv. Seed 2463.

### How I chose the model  
Accuracy is useless here. A model that always says “no” is right 88.7% of the time but finds zero buyers. So I used ROC-AUC (how well it ranks buyers above non-buyers) with 5-fold stratified CV. Always-no baseline = 0.500. Logistic regression = 0.791 ± 0.012, KNN = 0.781 ± 0.014, random forest = 0.800 ± 0.010. Forest is a bit better, but the gap with logistic (0.009) is smaller than the fold variation, so not a real win. Only forest vs KNN is a bit more meaningful. I picked logistic regression because it’s the simplest and easiest to explain to the call-centre team. Also tried class balancing — AUC changed by only 0.001, basically noise.

### Final results  
I kept 20% of the training data as hold-out and scored it only at the end. ROC-AUC 0.795, average precision 0.467 — close to the CV numbers, good sign. Then applied the 10% rule: out of 6 588 clients, top 659 get a call. Of those, 324 said yes → almost every second call (49.2%) is a success. Random list would give only 11.3%, so the model is about 4.4 times better. It also finds 43.7% of all buyers. Downside: 418 buyers are missed just because capacity is only 10%.

### Limits and how it can fail  
Biggest problem is the split. I split randomly, so the hold-out has calls from the same months and same economy as the training data. In real life the bank trains on the past and calls in the future. 2008–2010 had big changes in interest rates and jobs. The strongest features (euribor, employment numbers) describe exactly this economy. If the economy changes, the scores can become useless and real results will be worse. Second risk: the bank may start recording data differently (more “unknown” in default or education). Also this is a teaching dataset — don’t use it to make conclusions about real banks or real people.

### What to monitor after launch  
1. Share of “yes” among called clients vs the 49% on hold-out (ideally with a small random control group).  
2. Score distribution and main inputs (euribor, share of “unknown”).  
3. ROC-AUC on recent calls every month. If it drops clearly below 0.79 from CV, retrain on newer data.