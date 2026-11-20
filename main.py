"""SimulacraBench submission -- DarkStack.

DarkStack: a pooled, routing-as-data, in-container, self-tuned mixture.

No LLM, no bundled artifact, no network. Everything is fitted inside predict()
on the visible respondents the harness hands over (TRAIN in development,
TRAIN + DEV in the final phase), so one file adapts itself to both phases.

What is scored is p(answer_j | GIVEN block) for respondents whose answers are
all blank. Several candidates are built for every item and pooled:

  M  crowd marginal            the official zero, kept as a floor
  L  greedy backoff ladder     count-based; picks its own strata and smoothing
                               by closed-form leave-one-out log score
  B  gate recursion            the schema's skip logic taken at its word, as in
                               the starter kit's reference model
  F  multi-output forest       one forest for all items; leaves are groups of
                               respondents the whole questionnaire treats alike
  G  gradient-boosted trees    one small model per item, most signal first, as
                               many as the clock allows
  H  wide + deep student       one shared representation of the respondent
                               feeding every item's head; repeated K-fold bag
  R  residual student          the same net added on top of the ladder's
                               log-probabilities: as sharp as the counts where
                               the counts are sure, learning only their errors
  W  multinomial logit         main effects only, all items at once
  D  persona-distilled student (off by default) H trained on targets a
                               latent-class teacher builds from each TRAIN
                               respondent's *other* answers

Design stances:

  1. ROUTING IS DATA FIRST, A RULE SECOND. The gate sentinel, "Not answered",
     refusals -- all are ordinary levels, so who gets skipped or drops out is
     learned from the rows, including from the missingness pattern of the GIVEN
     block itself. The declared gates can enter as one candidate (B); how far
     they can be trusted is then decided item by item, out of fold.

  2. WHAT THE COUNTS ARE SURE OF STAYS SURE. Much of a real questionnaire is
     decided outright by the GIVEN block (a module one group never sees, an
     answer nobody at a level ever gives). Counts know this exactly; a network
     stopped early does not. So the residual student starts from the counts,
     every item keeps the epoch that was best for it, and after pooling any
     answer no comparable visible respondent ever gave is capped at the
     posterior mean of a never-seen outcome (see _support_caps).

  3. SELF-TUNING INSIDE THE AIRGAP. The real rows are only visible inside the
     container, so every choice that needs them -- ladder strata and smoothing,
     early stopping, per-item pooling weights, caps, sharpening -- is made
     there, on out-of-fold log score, under a deadline.

  4. STACK FOR THE BAG YOU DEPLOY. A student's out-of-fold vector averages R
     models, its held-out vector K * R, so plain stacking under-weights the
     candidates bagging helps most. The pool's objective is extrapolated in
     bag size (score ~ F_inf - a / r) to the deployed size before the weights
     are fitted. See _debias_weights.

  5. WHAT A LOOKALIKE SAID IS EVIDENCE. Every candidate above is a smooth
     function of the GIVEN block, so none of them can use the one thing a
     table of respondents also holds: the answers of the visible respondents
     who match this one on nearly every GIVEN column (a household interviewed
     in two rounds, a re-registered account, a tightly knit stratum). After
     pooling, each row's vector is updated with the answers of its neighbours
     at Hamming distance 0..4, the pooled vector acting as the prior; bandwidth
     and prior strength are chosen per item by leave-one-out log score, and an
     item where the neighbours add nothing is left untouched. See _echo. (Off:
     on the hosted development data it changed nothing.)

  6. NOTHING CAN COST THE RUN -- OR, IN STRICT MODE, NOTHING FAILS QUIETLY.
     With CFG["strict"] off, the crowd marginal is always in the pool, a
     candidate that fails or runs out of time is dropped, and any exception
     falls back to the official baseline's output. With it on, the same events
     raise instead, each with its own exception type, so a hosted run that is
     not doing what the local run does says so. Nothing is ever printed: the
     only things that leave the container are the vectors or an exception type.

Ablation switches are in CFG.
"""

import os as _os
import time as _time

_T0 = _time.time()

# The hosted container has 8 cores. A host that reports more would make every
# thread pool oversubscribe them, so the pools are pinned before they exist.
_CORES = max(1, min(8, _os.cpu_count() or 8))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, str(_CORES))

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAVE_TORCH = True
    try:
        torch.set_num_threads(_CORES)
    except Exception:
        pass
except Exception:                      # the ladder and the pool need no torch
    torch = None
    _HAVE_TORCH = False


CFG = {
    # In strict mode nothing degrades quietly: a stage that fails, is skipped
    # or runs out of time raises, and the platform's sanitized exception type
    # says which one (see _degraded). A failed submission costs no quota.
    "strict": False,
    "strict_gpu": True,          # in strict mode, also require a visible GPU
    # ---- ablation switches -------------------------------------------------
    "use_ladder": True,          # candidate L
    "use_gates": False,          # candidate B (hosted: cost UNHCR a cell)
    "use_forest": True,          # candidate F
    "use_gbdt": True,            # candidate G
    "use_student": True,         # candidates H, R and W
    "use_distill": False,        # candidate D
    "use_item_weights": True,    # per-item pooling weights (else one global set)
    "use_debias": True,          # fit the pool for the bag that is deployed
    "use_sharpen": True,         # per-item power sharpening of the pooled vector
    "use_echo": False,           # update the pooled vector with lookalikes' answers
    # Share of the starter kit's reference model (official_baseline.py, shipped
    # unchanged beside this file) in the final vector, per instrument. 0 leaves
    # it out entirely; 1 returns its vectors as they are.
    "official_weight": {"unicef": 0.0, "world_bank": 0.0, "unhcr": 0.0},
    "use_support": True,         # cap answers no comparable respondent ever gave
    "use_defloor": True,         # invert the grader's epsilon-mixing
    # ---- cross-validation --------------------------------------------------
    "n_folds": 5,
    "max_reps_dev": 3,           # repeated K-fold runs of the students, averaged
    "max_reps_final": 3,         # as in the best hosted development run
    "seed": 20261112,
    # ---- ladder ------------------------------------------------------------
    "ladder_depth": 5,
    "ladder_top": 6,             # columns a ladder may use beyond its first rung
    "ladder_min_gain": 3e-4,     # nats per respondent a new rung must buy
    "ladder_alphas": (0.5, 2.0, 8.0, 32.0, 128.0),
    # ---- gate recursion ----------------------------------------------------
    "gate_alpha": 2.0,
    "gate_strata": 2,
    # ---- trees -------------------------------------------------------------
    "forest_trees": 300,
    "forest_batch": 50,
    "forest_leaf": 25,
    "forest_feats": 0.3,
    "forest_smooth": 1.0,        # pseudo-respondents pulling a leaf to the crowd
    "gbdt_folds": 3,
    "gbdt_prior": 0.002,         # share of the crowd mixed into a tree's vector
    "gbdt_max_iter": 300,
    "gbdt_lr": 0.06,
    "gbdt_leaves": 15,
    "gbdt_min_leaf": 20,
    "gbdt_l2": 1.0,
    # ---- students ----------------------------------------------------------
    "students": (                # the first one is the lead: it always finishes
        {"name": "H", "depth": 2, "lr": 1e-3, "target": "hard", "offset": False},
        {"name": "R", "depth": 2, "lr": 1e-3, "target": "hard", "offset": True},
        {"name": "W", "depth": 0, "lr": 5e-3, "target": "hard", "offset": False},
    ),
    "student_distill": {"name": "D", "depth": 2, "lr": 1e-3, "target": "persona",
                        "offset": False},
    "offset_floor": 1e-4,        # the ladder's vector is clipped here before log
    "emb_dim": 12,
    "hidden": 256,
    "dropout": 0.15,
    "weight_decay": 0.05,
    "batch": 256,
    "max_epochs": 60,
    "patience": 5,
    # ---- persona teacher (only if use_distill) -----------------------------
    "lca_rows_per_class": 120,
    "lca_min_classes": 8,
    "lca_max_classes": 64,
    "lca_iters": 60,
    "lca_prior": 1.0,
    "distill_lambda": 0.75,
    # ---- pooling -----------------------------------------------------------
    "pool_shrink": 100.0,        # pseudo-respondents pulling item weights to global
    "debias_rows": 4000,         # visible rows the debiased objective is fitted on
    "debias_steps": 150,
    "sharpen_grid": (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3),
    "sharpen_min_gain": 5e-4,
    "echo_depth": 4,             # lookalikes differ on at most this many columns
    "echo_lambdas": (0.03, 0.1, 0.3, 0.6),    # weight lost per differing column
    "echo_alphas": (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),  # pseudo-respondents behind the prior
    "echo_min_gain": 5e-4,       # nats per respondent before an item is touched
    "echo_z": 3.0,               # ... and this many standard errors
    "support_min_rows": 30,      # a GIVEN level needs this many rows to veto
    "support_grid": (0.5, 1.0, 2.0),
    "support_min_gain": 1e-4,
    "floor": 1e-3,               # the grader's epsilon
    # ---- time --------------------------------------------------------------
    "budget_dev": 900.0,
    "budget_final": 1080.0,      # plan as for development, plus ~10% more rows;
                                 # the platform itself allows 3600
    "budget_use": 0.70,          # share of the wall clock this code plans to use
    "stage_ladder": 0.12,        # cumulative shares of one instrument's span
    "stage_gates": 0.14,         # only used when candidate B is on
    "stage_forest": 0.20,
    "stage_gbdt": 0.45,
    "stage_neural": 0.90,
    "span_override": None,       # local testing only
    "min_train_rows_ladder": 60,
    "min_train_rows_models": 600,
}

# Share of the run each instrument gets, keyed by its number of PREDICT items.
_KNOWN = {12: ("unicef", 0.20), 76: ("world_bank", 0.32), 213: ("unhcr", 0.48)}
_SEEN = set()

# Filled on every call; never printed. Local tooling reads it, the grader
# cannot. _KEEP additionally stashes the candidate matrices for local ablation.
DIAG = {}
_KEEP = False


# =============================================================== schema, data

def _item_lists(schema):
    items = [name for name, rec in schema["items"].items()
             if rec.get("class") in ("GIVEN", "PREDICT")]
    gated = schema.get("gated_value", "NA_GATED")
    options = {}
    for name in items:
        rec = schema["items"][name]
        opts = list(rec.get("values") or [])
        if rec.get("gate"):
            opts.append(gated)
        options[name] = opts
    return items, options


def _marginal_predict(frame, schema):
    """The official baseline, verbatim in behaviour. The last line of defence."""
    items, options = _item_lists(schema)
    marginals = {}
    for name in items:
        counts = frame[name].value_counts()
        weights = np.array([counts.get(opt, 0) + 0.5 for opt in options[name]],
                           dtype=float)
        marginals[name] = (weights / weights.sum()).tolist()
    blank = pd.isna(frame[items].to_numpy(dtype=object))
    return [marginals[items[c]]
            for r in range(blank.shape[0])
            for c in range(len(items))
            if blank[r, c]]


def _encode(frame, items, options):
    """Answers as integer codes. -1 where blank or not a declared option."""
    grid = frame[items].to_numpy(dtype=object)
    blank = pd.isna(grid)
    codes = np.full(grid.shape, -1, dtype=np.int64)
    for c, name in enumerate(items):
        lookup = {opt: i for i, opt in enumerate(options[name])}
        mapped = np.array(pd.Series(grid[:, c]).map(lookup), dtype=float)
        stray = np.isnan(mapped) & ~blank[:, c]
        if stray.any():                 # e.g. 1 against "1" after a round trip
            as_text = {str(opt): i for i, opt in enumerate(options[name])}
            for r in np.nonzero(stray)[0]:
                mapped[r] = as_text.get(str(grid[r, c]), np.nan)
        codes[:, c] = np.where(np.isnan(mapped), -1, mapped).astype(np.int64)
    return codes, blank


# ===================================================================== ladder

def _loo_counts(comb, y, width):
    """For each row: how many rows share its stratum, and its stratum+answer."""
    _, inv_t, cnt_t = np.unique(comb, return_inverse=True, return_counts=True)
    _, inv_j, cnt_j = np.unique(comb * width + y, return_inverse=True,
                                return_counts=True)
    return cnt_j[inv_j].astype(float), cnt_t[inv_t].astype(float)


def _first_rungs(x_fit, col_fit, y1h_fit, base_true, cards, alphas):
    """Leave-one-out log score of every (item, column, alpha) first rung.

    One pass per GIVEN column covers all items at once: the stratum-by-answer
    counts are a (levels x answer columns) table, and each respondent's own
    answer is taken out of the cell it sits in.
    """
    n, n_feat = x_fit.shape
    n_items = col_fit.shape[1]
    out = np.empty((n_items, n_feat, len(alphas)))
    for f in range(n_feat):
        x = x_fit[:, f]
        table = np.zeros((int(cards[f]), y1h_fit.shape[1]), dtype=np.float32)
        for level in np.unique(x):
            table[level] = y1h_fit[x == level].sum(axis=0)
        c_joint = table[x[:, None], col_fit].astype(np.float64)
        c_tot = np.bincount(x, minlength=int(cards[f])).astype(np.float64)[x][:, None]
        for ai, alpha in enumerate(alphas):
            out[:, f, ai] = np.log((c_joint - 1.0 + alpha * base_true)
                                   / (c_tot - 1.0 + alpha)).mean(axis=0)
    return out


def _ladder_item(x_tr, x_te, y, width, cards, cur_tr, cur_te, first, allowed,
                 cfg):
    """Greedy Dirichlet backoff for one item.

    Each rung refines the strata by one more GIVEN column and smooths the
    stratum counts towards the rung below:  (n_s + alpha * below) / (N_s + alpha).
    The column and alpha of a rung are the pair with the best leave-one-out log
    score, which for count models is closed form -- take the respondent's own
    answer out of the counts -- so the search costs no refits. `first` is the
    already-scored best first rung; later rungs choose among `allowed`.
    """
    n = x_tr.shape[0]
    rows = np.arange(n)
    alphas = cfg["ladder_alphas"]
    key_tr = np.zeros(n, dtype=np.int64)
    key_te = np.zeros(x_te.shape[0], dtype=np.int64)
    cur_true = cur_tr[rows, y]
    best_ll = float(np.log(cur_true).mean())
    used = []

    for depth in range(cfg["ladder_depth"]):
        if depth == 0:
            pick = first
        else:
            pick = None
            for f in allowed:
                if f in used:
                    continue
                comb = key_tr * cards[f] + x_tr[:, f]
                c_joint, c_tot = _loo_counts(comb, y, width)
                for alpha in alphas:
                    ll = float(np.log((c_joint - 1.0 + alpha * cur_true)
                                      / (c_tot - 1.0 + alpha)).mean())
                    if pick is None or ll > pick[0]:
                        pick = (ll, f, alpha)
        if pick is None or pick[0] - best_ll < cfg["ladder_min_gain"]:
            break
        best_ll, f, alpha = pick
        used.append(f)

        comb_tr = key_tr * cards[f] + x_tr[:, f]
        comb_te = key_te * cards[f] + x_te[:, f]
        uniq, inv = np.unique(np.concatenate([comb_tr, comb_te]),
                              return_inverse=True)
        inv_tr, inv_te = inv[:n], inv[n:]
        counts = np.bincount(inv_tr * width + y,
                             minlength=uniq.size * width
                             ).reshape(uniq.size, width).astype(float)
        total = counts.sum(axis=1)

        new_tr = counts[inv_tr] + alpha * cur_tr
        new_tr[rows, y] -= 1.0
        new_tr /= (total[inv_tr] - 1.0 + alpha)[:, None]
        cur_te = (counts[inv_te] + alpha * cur_te) / (total[inv_te] + alpha)[:, None]
        cur_tr = new_tr
        cur_true = cur_tr[rows, y]
        key_tr, key_te = inv_tr, inv_te

    return cur_te, len(used)


def _ladder_fit_apply(x_fit, y_fit, x_app, widths, offs, cards, cfg, deadline):
    """Fit every item's ladder on the fit rows; predict the apply rows.

    Returns None if the clock ran out before every item was done.
    """
    n, n_items = y_fit.shape
    m = x_app.shape[0]
    rows = np.arange(n)
    col_fit = y_fit + offs[:-1][None, :]
    y1h = np.zeros((n, int(offs[-1])), dtype=np.float32)
    y1h[rows[:, None], col_fit] = 1.0
    counts_all = y1h.sum(axis=0).astype(np.float64)
    width_cols = np.repeat(widths, widths).astype(np.float64)
    base_cols = (counts_all - 1.0 + 0.5) / (n - 1.0 + 0.5 * width_cols)
    base_true = base_cols[col_fit]
    scores = _first_rungs(x_fit, col_fit, y1h, base_true, cards,
                          cfg["ladder_alphas"])
    del y1h

    out = np.empty((m, int(offs[-1])), dtype=np.float32)
    depths = []
    for j in range(n_items):
        if _time.time() > deadline:
            return None
        width = int(widths[j])
        a, b = int(offs[j]), int(offs[j + 1])
        y = y_fit[:, j]
        counts = counts_all[a:b]
        base_app = np.tile((counts + 0.5) / (n + 0.5 * width), (m, 1))
        base_loo = np.tile(counts + 0.5, (n, 1))
        base_loo[rows, y] -= 1.0
        base_loo /= (n - 1.0 + 0.5 * width)
        per_feat = scores[j].max(axis=1)
        order = np.argsort(per_feat)[::-1]
        f0 = int(order[0])
        a0 = int(np.argmax(scores[j, f0]))
        first = (float(scores[j, f0, a0]), f0, cfg["ladder_alphas"][a0])
        allowed = [int(f) for f in order[:cfg["ladder_top"]]]
        pred, depth = _ladder_item(x_fit, x_app, y, width, cards, base_loo,
                                   base_app, first, allowed, cfg)
        out[:, a:b] = pred
        depths.append(depth)
    return out, float(np.mean(depths))


def _marginal(y_tr, m, widths, offs):
    """Candidate M: leave-one-out for the visible rows, plain for the held-out."""
    n, n_items = y_tr.shape
    rows = np.arange(n)
    loo = np.empty((n, int(offs[-1])), dtype=np.float32)
    te = np.empty((m, int(offs[-1])), dtype=np.float32)
    for j in range(n_items):
        width = int(widths[j])
        a, b = int(offs[j]), int(offs[j + 1])
        y = y_tr[:, j]
        counts = np.bincount(y, minlength=width).astype(float)
        te[:, a:b] = (counts + 0.5) / (n + 0.5 * width)
        block = np.tile(counts + 0.5, (n, 1))
        block[rows, y] -= 1.0
        loo[:, a:b] = block / (n - 1.0 + 0.5 * width)
    return loo, te


def _ladder_candidate(x_tr, x_te, y_tr, widths, offs, cards, cfg, deadline, rng):
    """Candidate L, honestly out-of-fold.

    The ladder's own leave-one-out score is what *chose* its rungs, so it is
    optimistic about them. Each fold's ladder is therefore chosen without that
    fold, and the held-out rows get the average of the fold ladders.
    """
    n = x_tr.shape[0]
    m = x_te.shape[0]
    k_folds = cfg["n_folds"]
    folds = rng.permutation(n) % k_folds
    oof = np.empty((n, int(offs[-1])), dtype=np.float32)
    held = np.zeros((m, int(offs[-1])), dtype=np.float32)
    depths = []
    for k in range(k_folds):
        tr, va = folds != k, folds == k
        res = _ladder_fit_apply(x_tr[tr], y_tr[tr],
                                np.concatenate([x_tr[va], x_te]), widths, offs,
                                cards, cfg, deadline)
        if res is None:
            return None
        pred, depth = res
        n_va = int(va.sum())
        oof[va] = pred[:n_va]
        held += pred[n_va:] / k_folds
        depths.append(depth)
    return oof, held, float(np.mean(depths))


# ====================================================================== gates

_SIGNALS = {
    "torch": ImportError,          # torch could not be imported
    "cuda": OSError,               # no GPU visible to torch
    "ladder": ArithmeticError,     # the ladder ran out of time
    "gates": AssertionError,       # the gate candidate ran out of time
    "forest": BufferError,         # the forest grew no tree in its slice
    "trees": LookupError,          # boosted trees reached under a quarter of the items
    "lead": EOFError,              # the lead student got fewer than two repeats
    "students": ReferenceError,    # some student never completed a repeat
    "overrun": OverflowError,      # the instrument overran its span
}


def _degraded(kind):
    """Strict mode: turn a quiet degradation into a failure with its own type.

    Every signal is a fact about how this code ran on the hosted machine --
    an import, a device, a clock. None depends on what any respondent answered.
    """
    if CFG["strict"]:
        raise _SIGNALS[kind]()


def _gate_specs(schema, items, options, t_cols):
    """Per PREDICT item: None, or (parent column, parent is GIVEN, allowed codes)."""
    pos = {name: c for c, name in enumerate(items)}
    specs = []
    for c in t_cols:
        gate = schema["items"][items[c]].get("gate")
        if not gate:
            specs.append(None)
            continue
        parent = gate.get("parent")
        if parent not in pos:
            specs.append((-1, False, []))
            continue
        pc = pos[parent]
        lookup = {v: i for i, v in enumerate(schema["items"][parent].get("values") or [])}
        allowed = [lookup[v] for v in gate.get("observed_if", []) if v in lookup]
        specs.append((pc, schema["items"][parent]["class"] == "GIVEN", allowed))
    return specs


def _mutual_information(x, y, nx, ny):
    if x.size < 8:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _gate_fit_apply(x_fit, y_fit, x_app, parent_app, widths, offs, cards, specs,
                    slot_of, cfg):
    """The schema's skip logic taken at its word, fitted on the fit rows.

    P(answer | asked) comes from Dirichlet backoff over the GIVEN columns that
    say most about the answers of those who were asked. P(asked) follows the
    declared gate: an indicator when the parent is GIVEN, and the parent's own
    predicted distribution, multiplied down the chain, when it is not. This is
    the starter kit's reference model; here it is one candidate among several,
    and the pool decides, item by item, how far the declared gates can be
    trusted.
    """
    n_app = x_app.shape[0]
    n_items = y_fit.shape[1]
    alpha = cfg["gate_alpha"]
    answer, asked_p = [None] * n_items, [None] * n_items
    for j in range(n_items):
        k_ans = int(widths[j]) - (1 if specs[j] is not None else 0)
        y = y_fit[:, j]
        asked = y < k_ans
        ya = y[asked]
        xa = x_fit[asked]
        score = [_mutual_information(xa[:, f], ya, int(cards[f]), k_ans)
                 for f in range(x_fit.shape[1])]
        ladder = [int(f) for f in np.argsort(-np.asarray(score), kind="stable")
                  [:cfg["gate_strata"]]]
        running = np.full((n_app, k_ans), 1.0 / k_ans)
        key_fit = np.zeros(ya.size, dtype=np.int64)
        key_app = np.zeros(n_app, dtype=np.int64)
        size = 1
        for column in [None] + ladder:
            if column is not None:
                key_fit = key_fit * int(cards[column]) + xa[:, column]
                key_app = key_app * int(cards[column]) + x_app[:, column]
                size *= int(cards[column])
            table = np.bincount(key_fit * k_ans + ya, minlength=size * k_ans
                                ).reshape(size, k_ans).astype(float)
            counts = table[key_app]
            running = (counts + alpha * running) / (
                counts.sum(axis=1, keepdims=True) + alpha)
        answer[j] = running
        if specs[j] is not None and specs[j][0] < 0:     # parent not shipped
            asked_p[j] = np.full(n_app, (asked.sum() + 0.5) / (y.size + 1.0))

    def depth(j):
        levels, seen = 0, set()
        while specs[j] is not None and specs[j][0] in slot_of and j not in seen:
            seen.add(j)
            j = slot_of[specs[j][0]]
            levels += 1
        return levels

    for j in sorted(range(n_items), key=depth):
        if asked_p[j] is not None:
            continue
        if specs[j] is None:
            asked_p[j] = np.ones(n_app)
            continue
        pc, parent_given, allowed = specs[j]
        if parent_given or pc not in slot_of:
            asked_p[j] = np.isin(parent_app[:, pc], allowed).astype(float)
        else:
            jp = slot_of[pc]
            k_par = answer[jp].shape[1]
            opens = [a for a in allowed if a < k_par]
            asked_p[j] = asked_p[jp] * (answer[jp][:, opens].sum(axis=1)
                                        if opens else 0.0)

    out = np.empty((n_app, int(offs[-1])), dtype=np.float32)
    for j in range(n_items):
        a, b = int(offs[j]), int(offs[j + 1])
        if specs[j] is None:
            out[:, a:b] = answer[j]
        else:
            p = asked_p[j][:, None]
            out[:, a:b - 1] = p * answer[j]
            out[:, b - 1] = 1.0 - p[:, 0]
    return out


def _gate_candidate(x_tr, x_te, y_tr, parent_tr, parent_te, widths, offs, cards,
                    specs, slot_of, cfg, deadline, rng):
    """Candidate B, out-of-fold for the visible rows, fold-bagged for the rest."""
    n = x_tr.shape[0]
    m = x_te.shape[0]
    k_folds = cfg["n_folds"]
    folds = rng.permutation(n) % k_folds
    oof = np.empty((n, int(offs[-1])), dtype=np.float32)
    held = np.zeros((m, int(offs[-1])), dtype=np.float32)
    for k in range(k_folds):
        if _time.time() > deadline:
            return None
        tr, va = folds != k, folds == k
        pred = _gate_fit_apply(
            x_tr[tr], y_tr[tr], np.concatenate([x_tr[va], x_te]),
            np.concatenate([parent_tr[va], parent_te]), widths, offs, cards,
            specs, slot_of, cfg)
        n_va = int(va.sum())
        oof[va] = pred[:n_va]
        held += pred[n_va:] / k_folds
    return oof, held


# ====================================================================== trees

def _forest_candidate(x_tr, x_te, y_tr, widths, offs, cards, marg_row, cfg,
                      deadline):
    """Candidate F: one multi-output forest for every item at once.

    A tree here splits on whatever GIVEN level separates respondents best
    across *all* items, so its leaves are groups of respondents the whole
    questionnaire treats alike -- which is what a routing rule or an export
    convention shared by a hundred items looks like. Each leaf then answers
    every item with its own smoothed counts. Visible rows are scored only by
    the trees whose bootstrap left them out, so the candidate is out-of-bag
    honest without a single refit.
    """
    from sklearn.ensemble import RandomForestClassifier

    n, n_items = y_tr.shape
    m = x_te.shape[0]
    total_width = int(offs[-1])
    feat_off = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
    both = np.concatenate([x_tr, x_te])
    onehot = np.zeros((n + m, int(np.sum(cards))), dtype=np.float32)
    onehot[np.arange(n + m)[:, None], both + feat_off[None, :]] = 1.0
    starts = offs[:-1].astype(np.int64)
    marg = marg_row.astype(np.float64)

    oof = np.zeros((n, total_width), dtype=np.float32)
    seen = np.zeros(n)
    held = np.zeros((m, total_width), dtype=np.float32)
    grown, batch_no = 0, 0
    while grown < cfg["forest_trees"] and _time.time() < deadline:
        began = _time.time()
        forest = RandomForestClassifier(
            n_estimators=cfg["forest_batch"], min_samples_leaf=cfg["forest_leaf"],
            max_features=cfg["forest_feats"], bootstrap=True, n_jobs=_CORES,
            random_state=int(cfg["seed"]) + batch_no)
        forest.fit(onehot[:n], y_tr)
        classes = forest.classes_ if n_items > 1 else [forest.classes_]
        k_max = max(len(c) for c in classes)
        src, dst = [], []
        for j in range(n_items):
            for pos, code in enumerate(classes[j]):
                src.append(j * k_max + pos)
                dst.append(int(offs[j]) + int(code))
        src, dst = np.asarray(src), np.asarray(dst)
        for tree, bag in zip(forest.estimators_, forest.estimators_samples_):
            leaves = tree.apply(onehot)
            uniq, inv = np.unique(leaves, return_inverse=True)
            value = tree.tree_.value[uniq].reshape(uniq.size, n_items, -1)
            value = value[:, :, :k_max].reshape(uniq.size, -1)
            table = np.zeros((uniq.size, total_width))
            table[:, dst] = value[:, src]
            # tree_.value is counts in some scikit-learn versions and shares in
            # others; shares times the leaf size is counts in both.
            table /= np.repeat(np.maximum(np.add.reduceat(table, starts, axis=1),
                                          1e-12), widths, axis=1)
            size = tree.tree_.weighted_n_node_samples[uniq][:, None]
            table = ((table * size + cfg["forest_smooth"] * marg[None, :])
                     / (size + cfg["forest_smooth"])).astype(np.float32)
            out = np.ones(n, dtype=bool)
            out[bag] = False
            oof[out] += table[inv[:n][out]]
            seen[out] += 1
            held += table[inv[n:]]
        grown += len(forest.estimators_)
        batch_no += 1
        if _time.time() + (_time.time() - began) > deadline:
            break
    if grown == 0:
        return None
    never = seen == 0
    oof[~never] /= seen[~never][:, None]
    oof[never] = marg_row
    return oof, held / grown, grown


def _gbdt_fit(x_fit, y, width, prior, parts, cfg, seed):
    """One boosted model for one item; its vectors for each row set in `parts`.

    Options too rare to survive the model's own stratified validation split
    are folded into one bucket for fitting, and the bucket's probability is
    handed back in proportion to the crowd. Returns None when there is nothing
    to fit.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    counts = np.bincount(y, minlength=width)
    common = np.nonzero(counts >= 12)[0]
    rare = np.nonzero(counts < 12)[0]
    bucket = counts[rare].sum() >= 12
    if common.size + int(bucket) < 2:
        return None
    relabel = np.full(width, -1, dtype=np.int64)
    relabel[common] = np.arange(common.size)
    if bucket:
        relabel[rare] = common.size
    y_fit = relabel[y]
    keep = y_fit >= 0
    model = HistGradientBoostingClassifier(
        max_iter=cfg["gbdt_max_iter"], learning_rate=cfg["gbdt_lr"],
        max_leaf_nodes=cfg["gbdt_leaves"], min_samples_leaf=cfg["gbdt_min_leaf"],
        l2_regularization=cfg["gbdt_l2"],
        categorical_features=list(range(x_fit.shape[1])),
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
        random_state=seed)
    model.fit(x_fit[keep], y_fit[keep])
    seen = np.asarray(model.classes_, dtype=np.int64)
    mix = cfg["gbdt_prior"]
    out = []
    for x_part in parts:
        raw = np.zeros((x_part.shape[0], common.size + 1))
        raw[:, seen] = model.predict_proba(x_part)
        full = np.zeros((x_part.shape[0], width))
        full[:, common] = raw[:, :common.size]
        if rare.size:
            share = prior[rare] / max(prior[rare].sum(), 1e-12)
            full[:, rare] = raw[:, common.size:] * share[None, :]
        out.append((1.0 - mix) * full + mix * prior)
    return out


def _gbdt_candidate(x_tr, x_te, y_tr, widths, offs, marg_loo, marg_te, order,
                    cfg, deadline, rng):
    """Candidate G: one small boosted model per item.

    Items are taken in `order` -- most signal first, as the ladder measured it
    -- and the loop stops when the clock says so. An item it did not reach
    keeps the crowd marginal, which the pool then reads as "no better than M
    here". Visible rows are predicted out-of-fold; held-out rows by one more
    model fitted on every visible row, so the out-of-fold score describes the
    kind of model that is actually deployed.
    """
    n = y_tr.shape[0]
    k_folds = cfg["gbdt_folds"]
    folds = rng.permutation(n) % k_folds
    oof = marg_loo.copy()
    held = marg_te.copy()
    done = 0
    for j in order:
        if _time.time() > deadline:
            break
        a, b = int(offs[j]), int(offs[j + 1])
        y = y_tr[:, j]
        prior = marg_te[0, a:b].astype(float)
        item_oof = np.tile(prior, (n, 1))
        item_te = None
        try:
            for k in range(k_folds):
                tr, va = folds != k, folds == k
                got = _gbdt_fit(x_tr[tr], y[tr], b - a, prior, [x_tr[va]], cfg,
                                int(cfg["seed"]) + k)
                if got is not None:
                    item_oof[va] = got[0]
                if _time.time() > deadline:
                    break
            else:
                got = _gbdt_fit(x_tr, y, b - a, prior, [x_te], cfg,
                                int(cfg["seed"]) + k_folds)
                item_te = np.tile(prior, (x_te.shape[0], 1)) if got is None \
                    else got[0]
        except Exception:
            item_te = None
        if item_te is not None:
            oof[:, a:b] = item_oof
            held[:, a:b] = item_te
            done += 1
    return oof, held, done


# ==================================================================== student

if _HAVE_TORCH:

    class _Student(nn.Module):
        """Wide + deep multi-task net: GIVEN block in, every item's logits out.

        The wide path is a multinomial logit on the one-hot GIVEN block (main
        effects). The deep path shares one representation of the respondent
        across all items, which is what lets 213 items regularise each other.
        depth == 0 leaves the wide path alone.
        """

        def __init__(self, n_levels, n_feat, total_width, bias, depth, cfg):
            super().__init__()
            self.wide = nn.EmbeddingBag(n_levels, total_width, mode="sum")
            nn.init.zeros_(self.wide.weight)
            self.bias = nn.Parameter(bias.clone())
            self.deep = depth > 0
            if self.deep:
                self.emb = nn.Embedding(n_levels, cfg["emb_dim"])
                layers, dim = [], n_feat * cfg["emb_dim"]
                for _ in range(depth):
                    layers += [nn.Linear(dim, cfg["hidden"]), nn.SiLU(),
                               nn.Dropout(cfg["dropout"])]
                    dim = cfg["hidden"]
                self.mlp = nn.Sequential(*layers)
                self.out = nn.Linear(dim, total_width)
                nn.init.normal_(self.out.weight, std=0.01)
                nn.init.zeros_(self.out.bias)

        def forward(self, x, offset=None):
            logits = self.bias + self.wide(x)
            if self.deep:
                logits = logits + self.out(self.mlp(self.emb(x).flatten(1)))
            if offset is not None:
                logits = logits + offset
            return logits


    def _seg_log_softmax(logits, groups):
        """log-softmax within each item's block of columns."""
        out = torch.empty_like(logits)
        for cols in groups:                       # (items of one width, width)
            out[:, cols] = F.log_softmax(logits[:, cols], dim=-1)
        return out


    def _predict_probs(model, x, groups, offset=None, chunk=4096):
        parts = []
        with torch.no_grad():
            for s in range(0, x.shape[0], chunk):
                off = None if offset is None else offset[s:s + chunk]
                parts.append(_seg_log_softmax(model(x[s:s + chunk], off), groups
                                              ).exp().float().cpu())
        return torch.cat(parts, 0).numpy()


    def _train_student(x_tr, t_tr, x_va, col_va, x_te, spec, variant, cfg,
                       seed, deadline, o_tr=None, o_va=None, o_te=None):
        """Fit on (x_tr, t_tr); early-stop on the held-out fold's log score.

        `t_tr` is a row-stochastic target per item block -- one-hot for the
        hard students, a persona mixture for the distilled one. The stopping
        criterion is always the real answers of the validation fold.

        With offsets (o_*), the net's logits are added to the ladder's
        log-probabilities: it starts out exactly as sharp as the counts are and
        only has to learn where they are wrong.
        """
        torch.manual_seed(seed)
        device = x_tr.device
        bias = spec["bias"] if o_tr is None else torch.zeros_like(spec["bias"])
        model = _Student(spec["n_levels"], spec["n_feat"], spec["total_width"],
                         bias, variant["depth"], cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=variant["lr"],
                                weight_decay=cfg["weight_decay"])
        groups = spec["groups"]
        item_of_col = spec["item_of_col"]
        n = x_tr.shape[0]
        batch = cfg["batch"] * (2 if n >= 10000 else 1)
        n_items = col_va.shape[1]
        best_val, bad = float("inf"), 0
        best_item = torch.full((n_items,), float("inf"), device=device)
        keep_va = torch.zeros((x_va.shape[0], spec["total_width"]), device=device)
        keep_te = torch.zeros((x_te.shape[0], spec["total_width"]), device=device)

        for _ in range(cfg["max_epochs"]):
            model.train()
            perm = torch.randperm(n, device=device)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                off = None if o_tr is None else o_tr[idx]
                logp = _seg_log_softmax(model(x_tr[idx], off), groups)
                loss = -(t_tr[idx] * logp).sum(dim=1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logp = _seg_log_softmax(model(x_va, o_va), groups)
                per_item = -logp.gather(1, col_va).mean(dim=0)
                val = float(per_item.sum())
                # Items do not peak together: a routed block keeps sharpening
                # long after an attitude item has started to overfit. Each
                # item keeps the vectors of the epoch that was best for it.
                better = per_item < best_item - 1e-5
                if bool(better.any()):
                    cols = better[item_of_col]
                    keep_va[:, cols] = logp.exp()[:, cols]
                    keep_te[:, cols] = _seg_log_softmax(
                        model(x_te, o_te), groups).exp()[:, cols]
                    best_item = torch.where(better, per_item, best_item)
            if val < best_val - 1e-4:
                best_val, bad = val, 0
            else:
                bad += 1
                if bad in (2, 4):
                    for group in opt.param_groups:
                        group["lr"] *= 0.5
                if bad >= cfg["patience"]:
                    break
            if _time.time() > deadline:
                break

        return (keep_va.float().cpu().numpy(), keep_te.float().cpu().numpy(),
                best_val)


    # ============================================================== teacher

    def _lca_fit(x1h, n_classes, cfg, seed):
        """Latent-class model over the GIVEN block and every answer, by EM.

        A class is a persona: a full response profile, routing included.
        """
        n = x1h.shape[0]
        device = x1h.device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        assign = torch.randint(0, n_classes, (n,), generator=gen).to(device)
        resp = torch.full((n, n_classes), 0.1 / max(n_classes - 1, 1),
                          device=device)
        resp[torch.arange(n, device=device), assign] = 0.9
        level_share = x1h.mean(dim=0)
        prior = cfg["lca_prior"]
        last = None
        theta = log_theta = log_pi = None
        for it in range(cfg["lca_iters"]):
            size = resp.sum(dim=0)
            theta = (x1h.T @ resp + prior * level_share[:, None]) / (size + prior)
            log_theta = theta.clamp_min(1e-12).log()
            log_pi = torch.log((size + 1.0) / (n + n_classes))
            joint = x1h @ log_theta + log_pi
            norm = torch.logsumexp(joint, dim=1, keepdim=True)
            resp = (joint - norm).exp()
            ll = float(norm.mean())
            if last is not None and it >= 15 and ll - last < 1e-5 * abs(ll):
                break
            last = ll
        return theta, log_theta, log_pi


    def _persona_targets(x1h, col_all, blocks_all, offs, theta, log_theta,
                         log_pi):
        """Leave-one-item-out persona targets for every (respondent, item).

        For item j the respondent's persona is inferred from the GIVEN block
        and all their answers *except* j, then j's target is that persona
        mixture's answer distribution.
        """
        joint = x1h @ log_theta + log_pi
        n_items = col_all.shape[1]
        out = torch.empty((x1h.shape[0], int(offs[-1])), device=x1h.device)
        for j in range(n_items):
            a, b = blocks_all[j]
            post = torch.softmax(joint - log_theta[col_all[:, j]], dim=1)
            out[:, int(offs[j]):int(offs[j + 1])] = post @ theta[a:b].T
        return out


def _neural_candidates(x_tr, x_te, y_tr, widths, offs, cards, marg_te_row,
                       ladder, cfg, device_name, stage_deadline, hard_deadline,
                       max_reps, rng):
    """The students: out-of-fold for visible rows, fold-bagged for held-out.

    Repeated K-fold. Each repeat reshuffles the folds, so a visible row's
    out-of-fold vector ends up an average over several models that never saw
    it -- closer to the bag that predicts the held-out rows, which is what
    the pool's weights should be judged against.
    """
    device = torch.device(device_name)
    n, n_items = y_tr.shape
    m = x_te.shape[0]
    n_feat = x_tr.shape[1]
    total_width = int(offs[-1])
    feat_off = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
    n_levels = int(np.sum(cards))

    xa_tr = torch.as_tensor(x_tr + feat_off[None, :], device=device)
    xa_te = torch.as_tensor(x_te + feat_off[None, :], device=device)
    col_np = y_tr + offs[:-1][None, :]
    col = torch.as_tensor(col_np, device=device)
    y1h = torch.zeros((n, total_width), device=device)
    y1h.scatter_(1, col, 1.0)

    by_width = {}
    for j in range(n_items):
        by_width.setdefault(int(widths[j]), []).append(
            np.arange(int(offs[j]), int(offs[j + 1])))
    groups = [torch.as_tensor(np.stack(v), device=device)
              for v in by_width.values()]
    spec = {"n_levels": n_levels, "n_feat": n_feat, "total_width": total_width,
            "groups": groups,
            "item_of_col": torch.as_tensor(np.repeat(np.arange(n_items), widths),
                                           device=device),
            "bias": torch.as_tensor(np.log(np.maximum(marg_te_row, 1e-6)),
                                    dtype=torch.float32, device=device)}

    variants = [v for v in cfg["students"]
                if not v.get("offset") or ladder is not None]
    off_tr = off_te = None
    if any(v.get("offset") for v in variants):
        low = cfg["offset_floor"]
        off_tr = torch.as_tensor(np.log(np.maximum(ladder[0], low)),
                                 dtype=torch.float32, device=device)
        off_te = torch.as_tensor(np.log(np.maximum(ladder[1], low)),
                                 dtype=torch.float32, device=device)
    if cfg["use_distill"]:
        variants.append(cfg["student_distill"])
        g1h = torch.zeros((n, n_levels), device=device)
        g1h.scatter_(1, xa_tr, 1.0)
        x1h_all = torch.cat([g1h, y1h], dim=1)
        col_all = col + n_levels
        blocks_all = [(n_levels + int(offs[j]), n_levels + int(offs[j + 1]))
                      for j in range(n_items)]
        del g1h

    k_folds = cfg["n_folds"]
    lam = cfg["distill_lambda"]
    seed = int(cfg["seed"])
    sums = {}                       # name -> [oof sum, held-out sum, [liks]]
    took = {}                       # name -> seconds its last repeat needed

    def run_repeat(vi, rep, patient):
        """One K-fold pass of one student. `patient` lets it run to the end."""
        variant = variants[vi]
        began = _time.time()
        folds = rng.permutation(n) % k_folds
        oof = np.zeros((n, total_width), np.float32)
        held = np.zeros((m, total_width), np.float32)
        for k in range(k_folds):
            va_np = np.nonzero(folds == k)[0]
            tr = torch.as_tensor(np.nonzero(folds != k)[0], device=device)
            va = torch.as_tensor(va_np, device=device)
            now = _time.time()
            if patient:
                limit = hard_deadline
            else:
                limit = now + max(stage_deadline - now, 0.0) / (k_folds - k)
            if variant["target"] == "hard":
                target = y1h[tr]
            else:
                n_classes = int(np.clip(len(tr) // cfg["lca_rows_per_class"],
                                        cfg["lca_min_classes"],
                                        cfg["lca_max_classes"]))
                theta, log_theta, log_pi = _lca_fit(
                    x1h_all[tr], n_classes, cfg, seed + 7919 * rep + k)
                soft = _persona_targets(x1h_all[tr], col_all[tr], blocks_all,
                                        offs, theta, log_theta, log_pi)
                target = (1.0 - lam) * y1h[tr] + lam * soft
                del soft, theta, log_theta
            use_off = bool(variant.get("offset"))
            p_va, p_te, _ = _train_student(
                xa_tr[tr], target, xa_tr[va], col[va], xa_te, spec, variant,
                cfg, seed + 104729 * rep + 31 * k + 17 * vi, limit,
                off_tr[tr] if use_off else None,
                off_tr[va] if use_off else None,
                off_te if use_off else None)
            oof[va_np] = p_va
            held += p_te / k_folds
            del target
        lik = np.take_along_axis(oof, col_np, axis=1)
        name = variant["name"]
        if name not in sums:
            sums[name] = [oof, held, [lik]]
        else:
            sums[name][0] += oof
            sums[name][1] += held
            sums[name][2].append(lik)
        took[name] = _time.time() - began

    def fits(names):
        """Would one more repeat of each of these students end in time?"""
        need = sum(took.get(nm, took[variants[0]["name"]]) for nm in names)
        return _time.time() + 1.15 * need <= stage_deadline

    # The first student is the one the pool leans on, so it goes first and is
    # never cut short; its second repeat comes next, because two repeats are
    # what lets the pool be fitted for the deployed bag. The other students
    # follow, and whole extra rounds only while the clock allows.
    lead = variants[0]["name"]
    rest = [v["name"] for v in variants[1:]]
    run_repeat(0, 0, True)
    if max_reps >= 2 and fits([lead]):
        run_repeat(0, 1, False)
    lead_reps = len(sums[lead][2])
    for vi in range(1, len(variants)):
        for rep in range(lead_reps):
            if fits([variants[vi]["name"]]):
                run_repeat(vi, rep, False)
    if lead_reps >= 2 and all(len(sums.get(nm, [0, 0, []])[2]) == lead_reps
                              for nm in rest):
        for rep in range(lead_reps, max_reps):
            if not fits([lead] + rest):
                break
            for vi in range(len(variants)):
                run_repeat(vi, rep, False)

    return [(v["name"], sums[v["name"]][0] / len(sums[v["name"]][2]),
             sums[v["name"]][1] / len(sums[v["name"]][2]), sums[v["name"]][2])
            for v in variants if v["name"] in sums]


# ==================================================================== pooling

def _pool_weights(liks, shrink, per_item, iters=80):
    """Linear-pool weights by EM on out-of-fold likelihoods of the truth.

    liks[c, i, j] is the probability candidate c gave respondent i's actual
    answer to item j, from a model that had not seen that respondent. The
    mixture-weight likelihood is concave, EM climbs it monotonically, and the
    crowd marginal is always a candidate -- so the pool cannot do worse
    out-of-fold than the official zero. Item weights are shrunk to the global
    ones by `shrink` pseudo-respondents.
    """
    n_cand, n, n_items = liks.shape
    liks = np.maximum(liks, 1e-12)
    w_glob = np.full(n_cand, 1.0 / n_cand)
    flat = liks.reshape(n_cand, -1)
    for _ in range(iters):
        mix = (w_glob[:, None] * flat).sum(axis=0)
        w_glob = (w_glob[:, None] * flat / mix).mean(axis=1)
    w = np.tile(w_glob[:, None], (1, n_items))
    if per_item:
        for _ in range(iters):
            mix = (w[:, None, :] * liks).sum(axis=0)
            resp = (w[:, None, :] * liks / mix).sum(axis=1)
            w = (resp + shrink * w_glob[:, None]) / (n + shrink)
    return w, w_glob


def _debias_weights(liks, rep_liks, cfg, w_init, w_glob, device_name, rng):
    """Re-fit the pool for the bag that is actually deployed.

    A student's out-of-fold vector for a visible row averages R models, one
    per repeat; its held-out vector averages K * R. Out-of-fold scores
    therefore understate bagged candidates, and plain stacking under-weights
    exactly the candidates bagging helps most. To first order the log score of
    an r-model average is  F(r) = F_inf - a / r,  so two points -- the R-model
    average and the single repeats -- give the objective at the deployed size:

        J(w) = F_R(w) + gamma * (F_R(w) - F_1(w)),  gamma = (K - 1) / (K (R - 1)).

    J is not concave, so it is climbed by gradient: first one weight vector
    for the whole instrument, then per item, shrunk towards that vector and
    keeping each item's best iterate.
    """
    bagged = {c: v for c, v in rep_liks.items() if len(v) >= 2}
    if not bagged:
        return w_init, w_glob, 0.0
    n_rep = min(len(v) for v in bagged.values())
    k_folds = cfg["n_folds"]
    gamma = (k_folds - 1.0) / (k_folds * (n_rep - 1.0))
    n_cand, n, n_items = liks.shape
    rows = np.arange(n)
    if n > cfg["debias_rows"]:
        rows = np.sort(rng.choice(n, size=cfg["debias_rows"], replace=False))
    device = torch.device(device_name)
    bag = torch.as_tensor(np.maximum(liks[:, rows], 1e-12), dtype=torch.float32,
                          device=device)
    singles = bag.unsqueeze(0).repeat(n_rep, 1, 1, 1)
    for c, reps in bagged.items():
        for r in range(n_rep):
            singles[r, c] = torch.as_tensor(np.maximum(reps[r][rows], 1e-12),
                                            dtype=torch.float32, device=device)

    def objective(log_w):                     # log_w: (cands, items or 1)
        w = log_w.exp()
        f_bag = torch.log((w[:, None, :] * bag).sum(dim=0)).mean(dim=0)
        f_one = torch.log((w[None, :, None, :] * singles).sum(dim=1)
                          ).mean(dim=(0, 1))
        return f_bag + gamma * (f_bag - f_one)

    # ---- one vector for the instrument ------------------------------------
    theta = torch.as_tensor(np.log(np.maximum(w_glob, 1e-6))[:, None],
                            dtype=torch.float32, device=device).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    best, best_glob = -float("inf"), None
    for _ in range(cfg["debias_steps"]):
        log_w = torch.log_softmax(theta, dim=0)
        total = objective(log_w).mean()
        if total.item() > best:
            best, best_glob = total.item(), log_w.detach().exp()[:, 0].clone()
        opt.zero_grad(set_to_none=True)
        (-total).backward()
        opt.step()
    glob = best_glob / best_glob.sum()
    if not cfg["use_item_weights"]:
        out = glob.double().cpu().numpy()
        return np.tile(out[:, None], (1, n_items)), out, gamma

    # ---- per item, shrunk to it -------------------------------------------
    strength = cfg["pool_shrink"] / float(n)
    start = glob[:, None].repeat(1, n_items)
    theta = start.clamp_min(1e-6).log().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    best_j = torch.full((n_items,), -float("inf"), device=device)
    best_w = start.clone()
    for _ in range(cfg["debias_steps"]):
        log_w = torch.log_softmax(theta, dim=0)
        obj = objective(log_w) + strength * (glob[:, None] * log_w).sum(dim=0)
        with torch.no_grad():
            better = obj > best_j
            best_j = torch.where(better, obj, best_j)
            best_w[:, better] = log_w.exp()[:, better]
        opt.zero_grad(set_to_none=True)
        (-obj.sum()).backward()
        opt.step()
    out = best_w.double().cpu().numpy()
    return (out / out.sum(axis=0, keepdims=True),
            glob.double().cpu().numpy(), gamma)


def _echo_counts(x_a, x_b, y1h_b, cards, depth, same, device_name):
    """For each row of a: the answers of the rows of b that differ from it on
    exactly d GIVEN columns, d = 0..depth, and how many such rows there are.

    `same` says a and b are the same rows, so a row is not its own neighbour.
    """
    n_a, n_b, n_f = x_a.shape[0], x_b.shape[0], x_a.shape[1]
    base = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
    wide = int(np.sum(cards))

    def one_hot(x):
        out = np.zeros((x.shape[0], wide), dtype=np.float32)
        out[np.arange(x.shape[0])[:, None], x + base[None, :]] = 1.0
        return out

    a1h, b1h = one_hot(x_a), one_hot(x_b)
    counts = np.zeros((depth + 1, n_a, y1h_b.shape[1]), dtype=np.float32)
    sizes = np.zeros((depth + 1, n_a), dtype=np.float32)
    chunk = int(max(64, 2.5e7 // max(n_b, 1)))
    on_gpu = _HAVE_TORCH and device_name == "cuda"
    if on_gpu:
        dev = torch.device("cuda")
        b_t = torch.as_tensor(b1h, device=dev)
        y_t = torch.as_tensor(y1h_b, device=dev)
    for lo in range(0, n_a, chunk):
        hi = min(lo + chunk, n_a)
        if on_gpu:
            dist = n_f - torch.as_tensor(a1h[lo:hi], device=dev) @ b_t.T
            for d in range(depth + 1):
                near = (torch.abs(dist - d) < 0.5).float()
                if same:
                    near[torch.arange(hi - lo, device=dev),
                         torch.arange(lo, hi, device=dev)] = 0.0
                counts[d, lo:hi] = (near @ y_t).cpu().numpy()
                sizes[d, lo:hi] = near.sum(dim=1).cpu().numpy()
            del dist, near
        else:
            dist = n_f - a1h[lo:hi] @ b1h.T
            for d in range(depth + 1):
                near = (np.abs(dist - d) < 0.5).astype(np.float32)
                if same:
                    near[np.arange(hi - lo), np.arange(lo, hi)] = 0.0
                counts[d, lo:hi] = near @ y1h_b
                sizes[d, lo:hi] = near.sum(axis=1)
            del dist, near
    return counts, sizes


def _echo(pooled_oof, pooled_te, x_tr, x_te, col_np, widths, offs, cards, cfg,
          device_name, deadline):
    """Update each row's pooled vector with what its lookalikes answered.

    A row's lookalikes are the visible respondents within `echo_depth` GIVEN
    columns of it; one that differs on d columns counts as lambda**d of a
    respondent. With S(y) their weighted count of answer y, Z their weighted
    number and p the pooled vector,

        q(y) = (S(y) + alpha * p(y)) / (Z + alpha),

    which is the posterior mean when p is a Dirichlet prior worth alpha
    respondents. A row nobody resembles keeps p. lambda and alpha are chosen
    per item on the visible rows, each scored against lookalikes that exclude
    itself, on the probability the grader scores; an item is only touched if
    that buys more than `echo_min_gain` nats per respondent and more than
    `echo_z` standard errors -- two dozen settings are tried per item, and the
    best of them looks like a gain even on answers that are pure noise.
    """
    n, n_items = col_np.shape
    depth = int(cfg["echo_depth"])
    y1h = np.zeros((n, int(offs[-1])), dtype=np.float32)
    y1h[np.arange(n)[:, None], col_np] = 1.0
    c_tr, z_tr = _echo_counts(x_tr, x_tr, y1h, cards, depth, True, device_name)
    if _time.time() > deadline:
        return pooled_oof, pooled_te, 0, 0.0
    eps = cfg["floor"]
    k_item = widths.astype(np.float64)[None, :]

    def score(p_true):
        return np.log(eps + (1.0 - k_item * eps) * p_true)

    p_true = np.take_along_axis(pooled_oof, col_np, axis=1)
    c_true = np.stack([np.take_along_axis(c_tr[d], col_np, axis=1)
                       for d in range(depth + 1)]).astype(np.float64)
    before = score(p_true)
    combos = [(lam, alpha) for lam in cfg["echo_lambdas"]
              for alpha in cfg["echo_alphas"]]
    best_gain = np.zeros(n_items)
    best_err = np.zeros(n_items)
    best = np.full(n_items, -1, dtype=np.int64)
    for idx, (lam, alpha) in enumerate(combos):
        wts = float(lam) ** np.arange(depth + 1)
        s_true = np.tensordot(wts, c_true, axes=1)
        z = (wts[:, None] * z_tr).sum(axis=0)[:, None]
        delta = score((s_true + alpha * p_true) / (z + alpha)) - before
        gain = delta.mean(axis=0)
        better = gain > best_gain
        best_gain[better] = gain[better]
        best_err[better] = delta.std(axis=0)[better] / np.sqrt(n)
        best[better] = idx
    use = (best_gain > cfg["echo_min_gain"]) & (best_gain > cfg["echo_z"] * best_err)
    if not use.any() or _time.time() > deadline:
        return pooled_oof, pooled_te, 0, 0.0
    c_te, z_te = _echo_counts(x_te, x_tr, y1h, cards, depth, False, device_name)
    out_oof, out_te = pooled_oof.copy(), pooled_te.copy()
    for idx, (lam, alpha) in enumerate(combos):
        chosen = use & (best == idx)
        if not chosen.any():
            continue
        cols = np.repeat(chosen, widths)
        wts = float(lam) ** np.arange(depth + 1)
        for c, z, src, dst in ((c_tr, z_tr, pooled_oof, out_oof),
                               (c_te, z_te, pooled_te, out_te)):
            s = np.tensordot(wts, c[:, :, cols].astype(np.float64), axes=1)
            zz = (wts[:, None] * z).sum(axis=0)[:, None]
            dst[:, cols] = (s + alpha * src[:, cols]) / (zz + alpha)
    return out_oof, out_te, int(use.sum()), float(best_gain[use].sum() / n_items)


def _support_tables(x_tr, col_np, total_width, cards):
    """Per GIVEN column: answers counted by level, and the size of each level."""
    n = x_tr.shape[0]
    y1h = np.zeros((n, total_width), dtype=np.float32)
    y1h[np.arange(n)[:, None], col_np] = 1.0
    tables = []
    for f in range(x_tr.shape[1]):
        x = x_tr[:, f]
        table = np.zeros((int(cards[f]), total_width), dtype=np.float32)
        for level in np.unique(x):
            table[level] = y1h[x == level].sum(axis=0)
        tables.append((table, np.bincount(x, minlength=int(cards[f])
                                          ).astype(np.float32)))
    return tables


def _support_caps(x, tables, min_rows, strength, own_col=None):
    """An upper bound on every answer's probability, from structural zeros.

    If none of the n visible respondents who share a GIVEN level with this
    row ever gave an answer, the row's probability of giving it is capped at
    strength / (n + 2 * strength) -- the posterior mean of a never-seen
    outcome. It is the one inference from counts that survives refinement:
    whatever nobody at a level does, nobody in any subgroup of it does either.
    A network that stopped early, or a pool that averaged a sharp candidate
    with a blunt one, leaks mass onto such answers; the cap takes it back.
    `own_col` takes each visible row's own answers out of the counts, so the
    caps can be validated leave-one-out on the rows they were built from.
    """
    caps = np.ones((x.shape[0], tables[0][0].shape[1]), dtype=np.float32)
    for f, (table, size) in enumerate(tables):
        level = x[:, f]
        count = table[level]
        rows = size[level][:, None]
        if own_col is not None:
            count = count.copy()
            np.put_along_axis(
                count, own_col,
                np.take_along_axis(count, own_col, axis=1) - 1.0, axis=1)
            rows = rows - 1.0
        bound = np.where((count <= 0) & (rows >= min_rows),
                         strength / (rows + 2.0 * strength), 1.0)
        np.minimum(caps, bound.astype(np.float32), out=caps)
    return caps


def _apply_caps(probs, caps, starts, widths):
    out = np.minimum(probs, caps)
    total = np.add.reduceat(out, starts, axis=1)
    return out / np.repeat(np.maximum(total, 1e-300), widths, axis=1)


def _support_projection(pooled_oof, pooled_te, x_tr, x_te, col_np, widths, offs,
                        cards, cfg):
    """Cap structural zeros, item by item, where leave-one-out says it pays."""
    starts = offs[:-1].astype(np.int64)
    n_items = len(widths)
    tables = _support_tables(x_tr, col_np, int(offs[-1]), cards)
    eps = cfg["floor"]
    k_item = widths.astype(np.float64)[None, :]

    def score(mat):
        return np.log(eps + (1.0 - k_item * eps)
                      * np.take_along_axis(mat, col_np, axis=1)).mean(axis=0)

    before = score(pooled_oof)
    best_gain = np.zeros(n_items)
    best_strength = np.zeros(n_items)
    best_oof = pooled_oof
    for strength in cfg["support_grid"]:
        caps = _support_caps(x_tr, tables, cfg["support_min_rows"], strength,
                             own_col=col_np)
        capped = _apply_caps(pooled_oof, caps, starts, widths)
        gain = score(capped) - before
        better = gain > best_gain
        if better.any():
            if best_oof is pooled_oof:
                best_oof = pooled_oof.copy()
            cols = np.repeat(better, widths)
            best_oof[:, cols] = capped[:, cols]
            best_gain[better] = gain[better]
            best_strength[better] = strength
        del caps, capped
    use = best_gain > cfg["support_min_gain"]
    if not use.any():
        return pooled_oof, pooled_te, 0, 0.0
    cols_off = np.repeat(~use, widths)
    if best_oof is not pooled_oof:
        best_oof[:, cols_off] = pooled_oof[:, cols_off]
    out_te = pooled_te.copy()
    for strength in cfg["support_grid"]:
        chosen = use & (best_strength == strength)
        if not chosen.any():
            continue
        caps = _support_caps(x_te, tables, cfg["support_min_rows"], strength)
        capped = _apply_caps(pooled_te, caps, starts, widths)
        cols = np.repeat(chosen, widths)
        out_te[:, cols] = capped[:, cols]
    return best_oof, out_te, int(use.sum()), float(best_gain[use].sum() / n_items)


def _sharpen_exponents(pool_oof, col_np, offs, cfg):
    """Per-item exponent tau for p -> p**tau / sum, chosen out-of-fold.

    A linear pool of calibrated forecasts is under-confident; a power
    transform is the one-parameter repair. tau stays 1 unless it buys more
    than `sharpen_min_gain` nats per respondent, measured on the probability
    the grader scores.
    """
    n, n_items = col_np.shape
    eps = cfg["floor"]
    k_item = np.diff(offs).astype(np.float64)[None, :]
    logp = np.log(np.maximum(pool_oof, 1e-12)).astype(np.float64)
    true_lp = np.take_along_axis(logp, col_np, axis=1)
    starts = offs[:-1].astype(np.int64)
    base, best_ll, best_tau = None, None, np.ones(n_items)
    for tau in cfg["sharpen_grid"]:
        norm = np.add.reduceat(np.exp(tau * logp), starts, axis=1)
        ll = np.log(eps + (1.0 - k_item * eps) * np.exp(tau * true_lp) / norm
                    ).mean(axis=0)
        if abs(tau - 1.0) < 1e-9:
            base = ll
        if best_ll is None:
            best_ll, best_tau = ll.copy(), np.full(n_items, tau)
        else:
            better = ll > best_ll
            best_ll[better] = ll[better]
            best_tau[better] = tau
    if base is None:
        return np.ones(n_items)
    keep = (best_ll - base) > cfg["sharpen_min_gain"]
    return np.where(keep, best_tau, 1.0)


def _finish(probs, widths, offs, taus, cfg):
    """Sharpen, renormalise, and undo the grader's epsilon-mixing."""
    probs = np.maximum(probs.astype(np.float64), 1e-12)
    eps = cfg["floor"]
    for j in range(len(widths)):
        a, b = int(offs[j]), int(offs[j + 1])
        block = probs[:, a:b]
        if taus is not None and abs(taus[j] - 1.0) > 1e-9:
            block = block ** taus[j]
        block = block / block.sum(axis=1, keepdims=True)
        if cfg["use_defloor"]:
            # The grader scores eps + (1 - K eps) q. Returning (p - eps) /
            # (1 - K eps) makes the scored vector equal p wherever p >= eps.
            block = np.maximum(block - eps, 0.0)
            total = block.sum(axis=1, keepdims=True)
            block = np.where(total > 0, block / np.maximum(total, 1e-300),
                             1.0 / (b - a))
        probs[:, a:b] = block
    return probs


# ===================================================================== budget

def _call_deadline(n_targets, n_visible, schema, cfg):
    """When this instrument has to be done, on the module's own clock."""
    split = schema.get("split") or {}
    n_train, n_dev = split.get("n_train"), split.get("n_dev")
    final = (isinstance(n_train, int) and isinstance(n_dev, int)
             and n_dev > 0 and n_visible >= n_train + n_dev)
    budget = cfg["budget_final"] if final else cfg["budget_dev"]
    now = _time.time()
    usable = budget * cfg["budget_use"] - (now - _T0)
    name, share = _KNOWN.get(n_targets, (None, 0.15))
    ahead = sum(w for k, (nm, w) in _KNOWN.items()
                if nm not in _SEEN and nm != name)
    if name is not None:
        _SEEN.add(name)
    span = max(usable * share / (share + ahead), 8.0)
    if cfg["span_override"]:
        span = float(cfg["span_override"])
    return now + span, final


# ======================================================================= main

def _predict(frame, schema):
    cfg = CFG
    started = _time.time()
    diag = {}
    items, options = _item_lists(schema)
    classes = {name: schema["items"][name]["class"] for name in items}
    codes, blank = _encode(frame, items, options)

    t_cols = [c for c, name in enumerate(items) if classes[name] == "PREDICT"]
    g_cols = [c for c, name in enumerate(items) if classes[name] == "GIVEN"]
    if not t_cols:
        return _marginal_predict(frame, schema)

    y_all = codes[:, t_cols]
    held_rows = blank[:, t_cols].any(axis=1)
    train_rows = ~held_rows & (y_all >= 0).all(axis=1)
    n, m = int(train_rows.sum()), int(held_rows.sum())
    if m == 0:
        return _marginal_predict(frame, schema)

    # ---- the GIVEN block: unknown / blank becomes one more level -----------
    feats, cards = [], []
    for c in g_cols:
        x = codes[:, c].copy()
        card = len(options[items[c]])
        if (x < 0).any():
            x[x < 0] = card
            card += 1
        if np.unique(x).size > 1:
            feats.append(x)
            cards.append(card)
    if not feats or n < cfg["min_train_rows_ladder"]:
        return _marginal_predict(frame, schema)

    x_all = np.stack(feats, axis=1)
    cards = np.asarray(cards, dtype=np.int64)
    x_tr, x_te = x_all[train_rows], x_all[held_rows]
    y_tr = y_all[train_rows]
    widths = np.asarray([len(options[items[c]]) for c in t_cols], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    n_items = len(t_cols)
    col_np = y_tr + offs[:-1][None, :]

    deadline, final = _call_deadline(n_items, n, schema, cfg)
    span = deadline - started
    rng = np.random.default_rng(cfg["seed"] + n_items)
    diag.update({"n_train": n, "n_held": m, "n_items": n_items, "final": final,
                 "span": span})

    # ---- M and L -----------------------------------------------------------
    big_enough = n >= cfg["min_train_rows_models"]
    strict = bool(cfg["strict"]) and big_enough
    eps = cfg["floor"]
    k_item = widths.astype(np.float64)[None, :]

    def scored(p_true):
        """The probability the grader actually scores, after its own mixing."""
        return eps + (1.0 - k_item * eps) * p_true

    marg_loo, marg_te = _marginal(y_tr, m, widths, offs)
    cand = [("M", marg_loo, marg_te)]
    ladder = None
    if cfg["use_ladder"]:
        try:
            ladder = _ladder_candidate(x_tr, x_te, y_tr, widths, offs, cards, cfg,
                                       started + cfg["stage_ladder"] * span, rng)
            if ladder is not None:
                cand.append(("L", ladder[0], ladder[1]))
                diag["ladder_depth_mean"] = ladder[2]
        except Exception as exc:
            if strict:
                raise
            ladder = None
            diag["ladder_error"] = type(exc).__name__
        if ladder is None and strict:
            _degraded("ladder")
    diag["t_ladder"] = _time.time() - started

    # ---- B -----------------------------------------------------------------
    if cfg["use_gates"] and big_enough:
        try:
            slot_of = {c: j for j, c in enumerate(t_cols)}
            gates = _gate_candidate(
                x_tr, x_te, y_tr, codes[train_rows], codes[held_rows], widths,
                offs, cards, _gate_specs(schema, items, options, t_cols),
                slot_of, cfg, started + cfg["stage_gates"] * span, rng)
            if gates is not None:
                cand.append(("B", gates[0], gates[1]))
            elif strict:
                _degraded("gates")
        except Exception as exc:
            if strict:
                raise
            diag["gates_error"] = type(exc).__name__
    diag["t_gates"] = _time.time() - started

    # ---- F -----------------------------------------------------------------
    if cfg["use_forest"] and big_enough:
        try:
            forest = _forest_candidate(x_tr, x_te, y_tr, widths, offs, cards,
                                       marg_te[0], cfg,
                                       started + cfg["stage_forest"] * span)
            if forest is not None:
                cand.append(("F", forest[0], forest[1]))
                diag["forest_trees"] = forest[2]
            elif strict:
                _degraded("forest")
        except Exception as exc:
            if strict:
                raise
            diag["forest_error"] = type(exc).__name__
    diag["t_forest"] = _time.time() - started

    # ---- G -----------------------------------------------------------------
    if cfg["use_gbdt"] and big_enough:
        try:
            # Most signal first: what the ladder gained over the crowd out of
            # fold, item by item. Without a ladder, the crowd's own loss.
            crowd = np.log(np.take_along_axis(marg_loo, col_np, axis=1)).mean(axis=0)
            if ladder is not None:
                signal = np.log(np.maximum(np.take_along_axis(
                    ladder[0], col_np, axis=1), 1e-12)).mean(axis=0) - crowd
            else:
                signal = -crowd
            g_oof, g_te, g_done = _gbdt_candidate(
                x_tr, x_te, y_tr, widths, offs, marg_loo, marg_te,
                np.argsort(signal)[::-1], cfg,
                started + cfg["stage_gbdt"] * span, rng)
            if g_done > 0:
                cand.append(("G", g_oof, g_te))
            diag["gbdt_items"] = g_done
        except Exception as exc:
            if strict:
                raise
            g_done = 0
            diag["gbdt_error"] = type(exc).__name__
        if strict and g_done < 0.25 * n_items:
            _degraded("trees")
    diag["t_gbdt"] = _time.time() - started

    # ---- H, R, W (and D) ---------------------------------------------------
    rep_liks = {}
    if cfg["use_student"] and big_enough:
        if not _HAVE_TORCH:
            if strict:
                _degraded("torch")
            devices = []
        else:
            devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
            if strict and cfg["strict_gpu"] and devices[0] != "cuda":
                _degraded("cuda")
        for device_name in devices:
            try:
                neural = _neural_candidates(
                    x_tr, x_te, y_tr, widths, offs, cards, marg_te[0], ladder,
                    cfg, device_name, started + cfg["stage_neural"] * span,
                    started + 1.25 * span,
                    cfg["max_reps_final"] if final else cfg["max_reps_dev"],
                    rng)
                for name, s_oof, s_te, rep_lik in neural:
                    rep_liks[len(cand)] = [scored(lik.astype(np.float64))
                                           for lik in rep_lik]
                    cand.append((name, s_oof, s_te))
                    diag["reps_" + name] = len(rep_lik)
                diag["device"] = device_name
                break
            except Exception as exc:          # try the next device, then go on
                if strict:
                    raise
                diag["neural_error_" + device_name] = type(exc).__name__
                if _time.time() > started + cfg["stage_neural"] * span:
                    break
        if strict:
            # W runs last and only when time is left over; it is not required.
            wanted = [v["name"] for v in cfg["students"][:2]
                      if not v.get("offset") or ladder is not None]
            if any(diag.get("reps_" + nm, 0) < 1 for nm in wanted):
                _degraded("students")
            if diag.get("reps_" + wanted[0], 0) < 2:
                _degraded("lead")
    diag["t_neural"] = _time.time() - started

    # ---- pool --------------------------------------------------------------
    # Everything from here on is judged on the probability the grader scores,
    # eps + (1 - K eps) p, not on p itself: a candidate that puts an exact zero
    # on the truth costs log(eps), as it will on the leaderboard, and no more.
    names = [c[0] for c in cand]
    w_em = None
    if len(cand) == 1:
        pooled_te, taus = marg_te.astype(np.float64), None
        pooled_raw = pooled_te
    else:
        liks = np.stack([scored(np.take_along_axis(c[1], col_np, axis=1
                                                   ).astype(np.float64))
                         for c in cand])
        weights, w_glob = _pool_weights(liks, cfg["pool_shrink"],
                                        cfg["use_item_weights"])
        w_em = weights
        if cfg["use_debias"] and rep_liks and _HAVE_TORCH:
            try:
                weights, w_glob, gamma = _debias_weights(
                    liks, rep_liks, cfg, weights, w_glob,
                    diag.get("device", "cpu"), rng)
                diag["debias_gamma"] = gamma
                diag["w_em_mean"] = dict(zip(
                    names, np.round(w_em.mean(axis=1), 4).tolist()))
            except Exception as exc:
                if strict:
                    raise
                weights = w_em
                diag["debias_error"] = type(exc).__name__
        w_cols = np.repeat(weights, widths, axis=1)          # (cands, columns)
        pooled_te = np.zeros((m, int(offs[-1])), dtype=np.float64)
        pooled_oof = np.zeros((n, int(offs[-1])), dtype=np.float64)
        for i, c in enumerate(cand):
            pooled_te += w_cols[i][None, :] * c[2]
            pooled_oof += w_cols[i][None, :] * c[1]
        pooled_raw = pooled_te
        if cfg["use_echo"] and big_enough:
            try:
                # Counted on the CPU: the arrays are small, and the numpy path
                # is the one that has been run end to end.
                pooled_oof, pooled_te, echoed, echo_gain = _echo(
                    pooled_oof, pooled_te, x_tr, x_te, col_np, widths, offs,
                    cards, cfg, "cpu", started + 1.15 * span)
                diag["echo_items"] = echoed
                diag["echo_gain"] = echo_gain
            except Exception as exc:
                if strict:
                    raise
                diag["echo_error"] = type(exc).__name__
        pooled_echo = pooled_te
        if cfg["use_support"]:
            try:
                pooled_oof, pooled_te, capped, cap_gain = _support_projection(
                    pooled_oof, pooled_te, x_tr, x_te, col_np, widths, offs,
                    cards, cfg)
                diag["support_items"] = capped
                diag["support_gain"] = cap_gain
            except Exception as exc:
                if strict:
                    raise
                pooled_te = pooled_echo
                diag["support_error"] = type(exc).__name__
        taus = (_sharpen_exponents(pooled_oof, col_np, offs, cfg)
                if cfg["use_sharpen"] else None)
        marg_ll = float(np.log(liks[0]).mean())
        diag["oof_ll"] = {nm: float(np.log(liks[i]).mean())
                          for i, nm in enumerate(names)}
        diag["oof_ll"]["pool"] = float(np.log(scored(np.take_along_axis(
            pooled_oof, col_np, axis=1))).mean())
        diag["oof_skill"] = {nm: 1.0 - v / marg_ll
                             for nm, v in diag["oof_ll"].items()}
        diag["w_glob"] = dict(zip(names, np.round(w_glob, 4).tolist()))
        diag["w_item_mean"] = dict(zip(names,
                                       np.round(weights.mean(axis=1), 4).tolist()))
        diag["taus_changed"] = int((np.abs(taus - 1) > 1e-9).sum()) \
            if taus is not None else 0
        del liks, pooled_oof

    final_te = _finish(pooled_te, widths, offs, taus, cfg)

    # ---- canonical order: rows top to bottom, items in schema key order ----
    marg_any = {}
    slot = {c: j for j, c in enumerate(t_cols)}
    held_index = {int(r): i for i, r in enumerate(np.nonzero(held_rows)[0])}
    blocks = [final_te[:, int(offs[j]):int(offs[j + 1])].tolist()
              for j in range(n_items)]
    out = []
    for r in np.nonzero(blank.any(axis=1))[0]:
        r = int(r)
        i = held_index.get(r)
        for c in np.nonzero(blank[r])[0]:
            c = int(c)
            if c in slot and i is not None:
                out.append(blocks[slot[c]][i])
            else:                               # a blank the model has no row for
                if c not in marg_any:
                    seen = codes[:, c][codes[:, c] >= 0]
                    w = np.bincount(seen, minlength=len(options[items[c]])) + 0.5
                    marg_any[c] = (w / w.sum()).tolist()
                out.append(marg_any[c])
    diag["t_total"] = _time.time() - started
    diag["candidates"] = names
    if _KEEP:
        diag["_cand"] = {c[0]: c[2] for c in cand}
        diag["_oof"] = {c[0]: c[1] for c in cand}
        diag["_col"] = col_np
        if w_em is not None:
            em_cols = np.repeat(w_em, widths, axis=1)
            diag["_cand"]["pool_em"] = sum(em_cols[i][None, :] * c[2]
                                           for i, c in enumerate(cand))
            diag["_cand"]["pool_raw"] = pooled_raw
            diag["_cand"]["pool_echo"] = pooled_echo
        diag["_pooled"] = pooled_te
        diag["_final"] = final_te
        diag["_layout"] = (widths, offs, [items[c] for c in t_cols])
    DIAG.clear()
    DIAG.update(diag)
    if strict and _time.time() > started + 1.3 * span:
        _degraded("overrun")
    return out


def _official_vectors(frame, schema):
    """The starter kit's reference model, run as it ships."""
    import importlib.util
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "official_baseline.py")
    spec = importlib.util.spec_from_file_location("_official_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [np.asarray(v, dtype=float) for v in module.predict(frame.copy(), schema)]


def _own_vectors(frame, schema):
    if CFG["strict"]:
        return _predict(frame, schema)
    try:
        return _predict(frame, schema)
    except Exception as exc:
        DIAG.clear()
        DIAG["fallback"] = type(exc).__name__
        return _marginal_predict(frame, schema)


def predict(frame, schema):
    """One probability vector per blank cell, in canonical order."""
    DIAG.clear()
    n_targets = sum(1 for r in schema["items"].values() if r["class"] == "PREDICT")
    name = _KNOWN.get(n_targets, (None, 0.0))[0]
    weight = float(CFG["official_weight"].get(name, 0.0)) if name else 0.0
    official = None
    if weight > 0.0:
        try:
            official = _official_vectors(frame, schema)
        except Exception as exc:
            if CFG["strict"]:
                raise
            DIAG["official_error"] = type(exc).__name__
    if official is not None and weight >= 1.0:
        _SEEN.add(name)
        return official
    own = _own_vectors(frame, schema)
    if official is None:
        return own
    if len(official) != len(own) or any(a.shape != np.shape(b)
                                        for a, b in zip(official, own)):
        DIAG["official_error"] = "layout"
        return own
    return [(1.0 - weight) * np.asarray(b, dtype=float) + weight * a
            for a, b in zip(official, own)]
