# Reinforcement Learning for Simplified Incident Response in a Simulated Enterprise Network

---

## 1. Cos'è

Il progetto studia se un agente di Reinforcement Learning impara a gestire un
incidente di sicurezza in una piccola rete aziendale meglio di baseline
procedurali. L'ambiente è un simulatore custom in stile Gymnasium: una rete di 8
nodi (5 *user*, 2 *service*, 1 *critical*) in cui un attaccante stocastico si
propaga tra nodi adiacenti. Il difensore (Blue Team) osserva lo stato in modo
**parziale e rumoroso** — vede alert, non la compromissione reale — e dispone di
quattro contromisure (Analyse, Isolate, Restore, Reconnect) più DoNothing, per un
totale di 33 azioni discrete. La reward penalizza sia il danno da compromissione
sia il costo delle contromisure: l'agente deve quindi bilanciare **sicurezza e
continuità operativa**, non solo "spegnere tutto".

Sull'ambiente vengono confrontati sette approcci: quattro baseline
(`do_nothing`, `panic_isolation`, `heuristic`, `random`), un oracolo con
conoscenza perfetta (`oracle`, tetto superiore di riferimento) e due agenti RL
(`dqn`, `ppo`). Il confronto è accompagnato da test di significatività statistica
(bootstrap appaiato).

## 2. Struttura del progetto

```
.
├── envs/
│   ├── topology.py              # topologia fissa a 8 nodi (NodeSpec / NodeRuntime)
│   ├── config.py                # parametri numerici dell'ambiente (EnvConfig)
│   └── incident_response_env.py # l'ambiente Gymnasium (reset/step/reward)
├── baselines/
│   ├── policy_base.py           # interfaccia BasePolicy, RandomPolicy, decode_obs, helper
│   ├── do_nothing.py            # baseline passiva di riferimento
│   ├── panic_isolation.py       # isola a ogni alert
│   ├── heuristic_defender.py    # difensore procedurale a priorità
│   ├── oracle_defender.py       # conoscenza perfetta (tetto superiore)
│   └── sb3_wrapper.py           # adatta un modello Stable-Baselines3 a BasePolicy
├── tests/
│   ├── conftest.py              # fixture pytest condivise
│   ├── test_env.py              # test sistematici dell'ambiente
│   └── test_baselines.py        # test delle policy non-RL
├── tools/
│   ├── plot_training_curves.py  # curve di training + tabella numerica dai .npz
│   ├── bootstrap_ci.py          # intervallo di confidenza bootstrap appaiato
│   ├── build_summary.py         # sintesi testuale finale del confronto
│   └── extract_action_dist.py   # distribuzione delle azioni per seed
├── train_dqn.py                 # training DQN
├── train_ppo.py                 # training PPO
├── evaluate.py                  # evaluation e tabella di confronto fra policy
├── random_test.py               # rollout casuali per ispezionare l'ambiente
├── utils.py                     # formattazione, logging e sintesi episodica
├── runs/                        # risultati (curve, tabelle, significatività, modelli)
├── requirements.txt
└── README.md
```

## 3. Requisiti e installazione

- **Python 3.10**
- Ambiente sviluppato con **PyTorch 2.6.0 + CUDA 12.4**, **Stable-Baselines3 2.8.0**,
  **Gymnasium 1.2.3** (versioni esatte in `requirements.txt`).

```bash
conda create -n progetto-ai python=3.10
conda activate progetto-ai
pip install -r requirements.txt
```

**Nota su PyTorch.** `requirements.txt` fissa `torch==2.6.0+cu124`, la build CUDA
12.4 usata per l'addestramento. Su una macchina con setup CUDA diverso, o senza
GPU, questa build potrebbe non essere installabile direttamente da PyPI: in tal
caso installare la versione di PyTorch appropriata al proprio sistema (vedi
`https://pytorch.org/get-started/locally/`) — ad esempio la build CPU
`pip install torch==2.6.0`. L'ambiente e le baseline girano anche su CPU; il
training RL beneficia della GPU ma non la richiede.

## 4. Come si esegue

### 4.1 Ispezione rapida dell'ambiente

```bash
python random_test.py                 # 10 episodi con policy casuale, primo verbose
python random_test.py --n-episodes 20 --seed 42
```

### 4.2 Training

DQN e PPO condividono lo stesso budget (500k step per seed in modalità `full`) e
salvano ciascuno, per ogni seed, il best model, le curve di evaluation
(`evaluations.npz`), il modello finale, i `meta.json` con gli iperparametri e i
log Tensorboard.

```bash
# DQN (le run ufficiali usano la normalizzazione della reward)
python train_dqn.py --mode full --seeds 0,1,2,3,4 --normalize-reward

# PPO
python train_ppo.py --mode full --seeds 0,1,2,3,4
```

Modalità più leggere per verifiche rapide: `--mode smoke` (~10k step) e
`--mode preliminary` (~200k step).

### 4.3 Evaluation e confronto

Confronta tutte le policy sugli **stessi 50 scenari** (seed condiviso, confronto
appaiato). Salvare l'output testuale su file consente di conservare la tabella
principale di confronto.

```bash
python evaluate.py \
    --policy heuristic,do_nothing,oracle,panic_isolation,random,dqn,ppo \
    --model-path     runs/dqn/full/seed1/best_model/best_model.zip \
    --ppo-model-path runs/ppo/full/seed3/best_model/best_model.zip \
    --n-episodes 50 --seed 42 --quiet --no-color \
    > runs/final_comparison/comparison_final.txt
```

Con `--save` l'evaluation salva anche, per ogni policy, un JSONL per-step e uno
per-episodio (usati poi dai test di significatività).

### 4.4 Strumenti di analisi

```bash
# Curve di training + tabella numerica (denormalizza DQN per confrontarlo con PPO)
python tools/plot_training_curves.py --algo dqn ppo \
    --runs-root runs --out-dir runs/final_comparison

# Significatività statistica: bootstrap appaiato sulla differenza di reward
python tools/bootstrap_ci.py \
    --a runs/final_comparison/evaluations/eval_ppo_best_model.zip__seed42_summaries.jsonl \
    --b runs/final_comparison/evaluations/eval_heuristic_seed42_summaries.jsonl \
    --label-a "PPO" --label-b "Heuristic" \
    > runs/final_comparison/significance/ppo_vs_heuristic.txt

# Sintesi testuale finale (integra tabella + bootstrap + numeri di training)
python tools/build_summary.py \
    --comparison-table runs/final_comparison/comparison_final.txt \
    --eval-dir         runs/final_comparison/evaluations \
    --dqn-summary      runs/dqn/full/runs_summary.json \
    --ppo-summary      runs/ppo/full/runs_summary.json \
    --out              runs/final_comparison/synthesis.txt
```

### 4.5 Test

La suite (`pytest`) verifica sistematicamente l'ambiente e le baseline:
dinamica del restore, regola del doppio conteggio nella reward, terminazione,
osservabilità parziale, correttezza delle policy procedurali.

```bash
pytest -q
```

## 5. Dove sono i risultati

Tutti gli artefatti stanno sotto `runs/`. In particolare, in
`runs/final_comparison/`:

- `comparison_final.txt` — la tabella di confronto fra le sette policy;
- `synthesis.txt` — la sintesi finale con i confronti di significatività;
- `significance/` — gli intervalli di confidenza bootstrap dei confronti chiave;
- `training_curves_*.png` e `training_curves_summary.txt` — curve di training e
  numeri per-seed;
- `evaluations/` — i JSONL per-policy prodotti dall'evaluation.

I due **best model** (DQN e PPO) sono inclusi sotto `runs/dqn/full/` e
`runs/ppo/full/`, così l'evaluation della sezione 4.3 è riproducibile senza
ri-addestrare.

## 6. Sintesi dei numeri chiave

Evaluation su 50 episodi (seed 42), ordinata per reward media decrescente
(reward alta = meglio; `comp_val` = valore operativo medio compromesso per step,
basso = meglio):

| policy | reward | win% | wipe% | inv% | crit% | comp_val |
|---|---:|---:|---:|---:|---:|---:|
| `oracle` (tetto) | −2.00 | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| **`ppo`** | **−26.82** | **88.0** | 12.0 | 12.4 | 12.0 | 1.30 |
| `heuristic` | −104.28 | 72.0 | 22.0 | 0.0 | 32.0 | 3.35 |
| `dqn` | −141.22 | 0.0 | 36.0 | 49.0 | 42.0 | 4.48 |
| `do_nothing` | −160.64 | 0.0 | 100.0 | 0.0 | 100.0 | 9.99 |
| `random` | −352.12 | 16.0 | 32.0 | 28.4 | 74.0 | 8.03 |
| `panic_isolation` | −364.11 | 0.0 | 0.0 | 0.0 | 4.0 | 2.45 |

Significatività (bootstrap appaiato, CI 95% sulla differenza di reward media):

| confronto (A vs B) | Δ = mean(A) − mean(B) | CI 95% | esito |
|---|---:|---|---|
| PPO vs Heuristic | +77.47 | [+36.85, +120.49] | significativo |
| PPO vs DQN | +114.41 | [+72.19, +159.36] | significativo |
| PPO vs DoNothing | +133.82 | [+107.29, +161.42] | significativo |
| DQN vs Heuristic | −36.94 | [−90.94, +16.62] | non significativo |
| DQN vs DoNothing | +19.42 | [−27.58, +65.10] | non significativo |

**Lettura dei risultati.** PPO **impara** il compito: batte la baseline
euristica in modo statisticamente significativo e si avvicina al tetto
dell'oracolo (−2.00), con win rate 88% e pochi nodi lasciati esposti. DQN, con
questa configurazione, **non** impara: è statisticamente indistinguibile da
`do_nothing`, non vince mai un episodio e spreca circa metà delle azioni in mosse
invalide (49%). Le due baseline "estreme" confermano le attese: `do_nothing`
perde sempre il nodo critico, `panic_isolation` sopravvive ma paga un costo di
isolamento altissimo.

I numeri di questa sezione si rileggono dagli artefatti in `runs/final_comparison/`
(`comparison_final.txt`, `synthesis.txt`) senza rieseguire nulla.

## 7. Note su seed e riproducibilità

- L'evaluation condivide lo stesso `--seed` fra tutte le policy: a parità di
  seed, ogni policy affronta esattamente gli stessi 50 scenari (stesso
  `patient_zero`, stessa propagazione, stesso rumore). Questo rende il confronto
  **appaiato** e i test di significatività più potenti.
- In training, `env_seed` (ambiente) e `algo_seed` (algoritmo) sono separati e
  registrati nei `meta.json`, per distinguere la varianza dovuta allo scenario da
  quella dovuta all'algoritmo.
- Tutte le esecuzioni sono deterministiche dato il seed; i modelli inclusi nella
  consegna permettono di riprodurre la tabella della sezione 6 lanciando
  l'evaluation della sezione 4.3.