import random
import time

# ---------------------------------------------------------------------------
# Constants & Evaluation Parameters
# ---------------------------------------------------------------------------
SUITS = ("H", "D", "C", "S")
DECK = frozenset((s, r) for s in SUITS for r in range(2, 15))

NONTRUMP_V = {
    2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0, 7: 5.0, 8: 7.0, 9: 9.0,
    10: 13.0, 11: 19.0, 12: 30.0, 13: 47.0, 14: 72.0
}
TRUMP_V = {
    2: 40.0, 3: 42.0, 4: 44.0, 5: 46.0, 6: 49.0, 7: 52.0, 8: 56.0,
    9: 61.0, 10: 67.0, 11: 74.0, 12: 84.0, 13: 95.0, 14: 106.0
}

TR_PREMIUM = 10.0
VOID_REFILL = 10.0
NEAR_VOID_PEN = 5.0
LEN_BONUS = 4.0
DUMP_LEN_W = 3.0
GUARD_K = 15.0
GUARD_Q = 5.0
TRUMP_CLING = 65.0
MASTER_KEEP = 55.0
EXIT_VOID_W = 7.0
RISK_MAX = 0.15
PT_DEAD = 0.08

P1_LEAD_MARGIN = 0.8
P1_FOLLOW_MARGIN = 0.4

N_PARTICLES = 200
MIN_PARTICLES = 50
BEHAVE_EPS = 0.20

EG_HAND = 10        # Deep search into Phase 2
EG_SAMPLES = 12
EG_TIME = 0.035

_S = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _v(card, trump):
    return TRUMP_V[card[1]] if card[0] == trump else NONTRUMP_V[card[1]]

def _beats(lead_card, follow_card, trump):
    ls, lr = lead_card
    fs, fr = follow_card
    if fs == ls:
        return fr > lr
    return fs == trump

def _trick_id(view):
    if view.phase == 1:
        return (25 - view.stock_remaining) // 2 + 1
    return 14 + sum(view.tricks_won.values())

def _opp_size(view):
    if view.phase == 1:
        return 13 if not view.current_trick else 12
    n = len(view.your_hand)
    return n if not view.current_trick else n - 1

def _unseen():
    return DECK - _S["ever_mine"] - _S["faceups"] - _S["opp_led"] - _S["opp_known"]

# ---------------------------------------------------------------------------
# Particle Filter with Hard Constraints on Known Opponent Cards
# ---------------------------------------------------------------------------
def _reset(view):
    _S.clear()
    _S["me"] = view.your_name
    _S["ever_mine"] = set(view.your_hand)
    _S["faceups"] = set()
    _S["opp_led"] = set()
    _S["opp_known"] = set()  # Hard-locked cards opponent is known to hold
    _S["pending"] = None
    if view.face_up_card is not None:
        _S["faceups"].add(view.face_up_card)

    pool = list(_unseen())
    parts = []
    need = 12 if view.current_trick else 13
    
    if view.current_trick:
        lead = view.current_trick[0][1]
        _S["opp_led"].add(lead)

    for _ in range(N_PARTICLES):
        p = set(_S["opp_known"])
        rem_size = need - len(p)
        if rem_size > 0 and len(pool) >= rem_size:
            p.update(random.sample(pool, rem_size))
        parts.append(p)
    _S["particles"] = parts

def _enforce_known(p, target_size):
    """Ensure locked known cards remain in particle p."""
    p.update(_S["opp_known"])
    while len(p) > target_size:
        removable = [c for c in p if c not in _S["opp_known"]]
        if not removable:
            break
        p.discard(random.choice(removable))

def _cull(card):
    _S["opp_known"].discard(card)
    unseen_pool = list(_unseen())
    for p in _S["particles"]:
        if card in p:
            p.discard(card)
            cand = [c for c in unseen_pool if c not in p]
            if cand:
                p.add(random.choice(cand))

def _opp_lead_event(card, view=None):
    _S["opp_led"].add(card)
    _S["opp_known"].discard(card)
    for p in _S["particles"]:
        p.discard(card)

def _opp_follow_event(our_lead, they_won, trump):
    ls, lr = our_lead
    unseen_pool = list(_unseen())

    out = []
    for p in _S["particles"]:
        same = [c for c in p if c[0] == ls]
        legal = same if same else list(p)
        cons = [c for c in legal if _beats(our_lead, c, trump) == they_won]

        if cons:
            played = min(cons, key=lambda c: _v(c, trump)) if random.random() >= BEHAVE_EPS else random.choice(cons)
            p.discard(played)
            _S["opp_known"].discard(played)
        else:
            # Repair particle while retaining opp_known
            size_after = len(p) - 1
            removable = [c for c in p if c not in _S["opp_known"]]
            if removable:
                p.discard(random.choice(removable))
            p.update(_S["opp_known"])
            cand = [c for c in unseen_pool if c not in p]
            while len(p) < size_after and cand:
                c = random.choice(cand)
                p.add(c)
                cand.remove(c)
        out.append(p)
    _S["particles"] = out

def _opp_gain_faceup(card):
    _S["opp_known"].add(card)
    for p in _S["particles"]:
        p.add(card)

def _opp_gain_hidden():
    pool = list(_unseen())
    if not pool:
        return
    for p in _S["particles"]:
        cand = [c for c in pool if c not in p]
        if cand:
            p.add(random.choice(cand))

def _repopulate(size):
    parts = _S["particles"]
    parts = [p for p in parts if len(p) == size]
    unseen_pool = list(_unseen())

    while len(parts) < N_PARTICLES:
        p = set(_S["opp_known"])
        rem_size = size - len(p)
        if rem_size > 0 and len(unseen_pool) >= rem_size:
            p.update(random.sample(unseen_pool, rem_size))
        parts.append(p)
    _S["particles"] = parts

def _update(view):
    if view.phase == 1 and view.stock_remaining == 25 and len(view.your_hand) == 13:
        _reset(view)
        return _trick_id(view)
    if not _S or "particles" not in _S:
        _reset(view)

    new_mine = set(view.your_hand) - _S["ever_mine"]
    _S["ever_mine"].update(new_mine)
    for c in new_mine:
        _cull(c)

    fu = view.face_up_card
    if fu is not None and fu not in _S["faceups"]:
        _S["faceups"].add(fu)
        _cull(fu)

    tid = _trick_id(view)

    pend = _S["pending"]
    if pend is not None and pend[0] < tid:
        ptid, pcard, pfu, pphase = pend
        we_won = (view.lead == view.your_name)
        _opp_follow_event(pcard, not we_won, view.trump_suit)
        if pphase == 1:
            if we_won:
                _opp_gain_hidden()
            elif pfu is not None:
                _opp_gain_faceup(pfu)
        _S["pending"] = None

    if view.current_trick:
        _opp_lead_event(view.current_trick[0][1], view)

    size = _opp_size(view)
    _repopulate(size)
    return tid

def _after_move(view, card, tid):
    if view.current_trick:
        lead_card = view.current_trick[0][1]
        we_win = _beats(lead_card, card, view.trump_suit)
        if view.phase == 1:
            if we_win:
                _opp_gain_hidden()
            elif view.face_up_card is not None:
                _opp_gain_faceup(view.face_up_card)
            _S["pending"] = None
    else:
        _S["pending"] = (tid, card, view.face_up_card, view.phase)

# ---------------------------------------------------------------------------
# Heuristics & Evaluations
# ---------------------------------------------------------------------------
def _suit_summaries():
    out = []
    for p in _S["particles"]:
        d = {}
        for s, r in p:
            if s in d:
                cnt, mx, mn = d[s]
                d[s] = (cnt + 1, max(mx, r), min(mn, r))
            else:
                d[s] = (1, r, r)
        out.append(d)
    return out

def _discard_value(card, hand, trump, opp_max=None):
    s, r = card
    val = _v(card, trump)
    if s == trump:
        return val + TRUMP_CLING
    ranks = [x[1] for x in hand if x[0] == s]
    hi = max(ranks)
    sc = val + DUMP_LEN_W * min(len(ranks), 5)
    if hi == 13 and len(ranks) == 2 and r < hi:
        sc += GUARD_K
    if hi == 12 and len(ranks) <= 3 and r < hi:
        sc += GUARD_Q
    if opp_max is not None and r > opp_max.get(s, 0):
        sc += MASTER_KEEP
    return sc

def _best_discard(cards, hand, trump, opp_max=None):
    return min(cards, key=lambda c: _discard_value(c, hand, trump, opp_max))

def _phase1_follow(view):
    hand = view.your_hand
    trump = view.trump_suit
    lead_card = view.current_trick[0][1]
    ls, lr = lead_card

    fu = view.face_up_card
    delta = _v(fu, trump) - 25.0 if fu else 0.0

    same = [c for c in hand if c[0] == ls]
    if same:
        duck = min(same, key=lambda c: c[1])
        winners = [c for c in same if c[1] > lr]
        if winners:
            w = min(winners, key=lambda c: c[1])
            if 2.0 * delta > _v(w, trump) - _v(duck, trump) + P1_FOLLOW_MARGIN:
                return w
        return duck

    dump = _best_discard(list(hand), hand, trump)
    if ls != trump:
        trumps = [c for c in hand if c[0] == trump]
        if trumps:
            w = min(trumps, key=lambda c: c[1])
            if 2.0 * delta > _v(w, trump) - _v(dump, trump) + P1_FOLLOW_MARGIN:
                return w
    return dump

def _phase1_lead(view):
    hand = view.your_hand
    trump = view.trump_suit
    dump = _best_discard(list(hand), hand, trump)
    fu = view.face_up_card
    if not fu:
        return dump

    val = _v(fu, trump)
    if val > 20.0:
        trumps = [c for c in hand if c[0] == trump]
        if trumps:
            return max(trumps, key=lambda c: c[1])
        masters = [c for c in hand if c[1] >= 13]
        if masters:
            return max(masters, key=lambda c: c[1])
    return dump

def _phase2_lead(view):
    hand = view.your_hand
    trump = view.trump_suit
    summaries = _suit_summaries()

    # Cash high masters
    for c in sorted(hand, key=lambda x: x[1], reverse=True):
        if c[0] != trump and c[1] == 14:
            return c

    exits = [c for c in hand if c[0] != trump] or list(hand)
    return min(exits, key=lambda c: _discard_value(c, hand, trump))

def _phase2_follow(view):
    hand = view.your_hand
    trump = view.trump_suit
    lead_card = view.current_trick[0][1]
    ls, lr = lead_card

    same = [c for c in hand if c[0] == ls]
    if same:
        winners = [c for c in same if c[1] > lr]
        return min(winners, key=lambda c: c[1]) if winners else min(same, key=lambda c: c[1])

    if ls != trump:
        trumps = [c for c in hand if c[0] == trump]
        if trumps:
            return min(trumps, key=lambda c: c[1])

    return _best_discard(list(hand), hand, trump)

# ---------------------------------------------------------------------------
# Alpha-Beta Accelerated Endgame Solver
# ---------------------------------------------------------------------------
def _solve_ab(my, opp, i_lead, trump, memo, alpha, beta):
    if not my:
        return 0
    key = (my, opp, i_lead)
    if key in memo:
        return memo[key]

    if i_lead:
        value = -1
        for c in sorted(my, key=lambda x: x[1], reverse=True):
            rest_my = tuple(x for x in my if x != c)
            same = [o for o in opp if o[0] == c[0]]
            replies = same if same else list(opp)

            worst = 99
            for o in sorted(replies, key=lambda x: x[1]):
                rest_opp = tuple(x for x in opp if x != o)
                i_win = not _beats(c, o, trump)
                val = (1 if i_win else 0) + _solve_ab(rest_my, rest_opp, i_win, trump, memo, alpha, beta)
                if val < worst:
                    worst = val
                if worst <= alpha:
                    break
            if worst > value:
                value = worst
            if value >= beta:
                break
            alpha = max(alpha, value)
    else:
        value = 99
        for o in sorted(opp, key=lambda x: x[1], reverse=True):
            rest_opp = tuple(x for x in opp if x != o)
            same = [c for c in my if c[0] == o[0]]
            replies = same if same else list(my)

            best = -1
            for c in sorted(replies, key=lambda x: x[1]):
                rest_my = tuple(x for x in my if x != c)
                i_win = _beats(o, c, trump)
                val = (1 if i_win else 0) + _solve_ab(rest_my, rest_opp, i_win, trump, memo, alpha, beta)
                if val > best:
                    best = val
                if best >= beta:
                    break
            if best < value:
                value = best
            if value <= alpha:
                break
            beta = min(beta, value)

    memo[key] = value
    return value

def _endgame_move(view):
    hand = view.your_hand
    trump = view.trump_suit
    t0 = time.time()

    if view.current_trick:
        lead_card = view.current_trick[0][1]
        same = [c for c in hand if c[0] == lead_card[0]]
        legal = same if same else list(hand)
    else:
        lead_card = None
        legal = list(hand)

    if len(legal) == 1:
        return legal[0]

    size = _opp_size(view)
    parts = [p for p in _S["particles"] if len(p) == size]
    if not parts:
        return None

    samples = [tuple(sorted(p)) for p in random.sample(parts, min(len(parts), EG_SAMPLES))]
    scores = {c: 0 for c in legal}

    for opp in samples:
        memo = {}
        for c in legal:
            rest_my = tuple(sorted(x for x in hand if x != c))
            if lead_card is None:
                same_o = [o for o in opp if o[0] == c[0]]
                replies = same_o if same_o else list(opp)
                worst = 99
                for o in replies:
                    rest_opp = tuple(x for x in opp if x != o)
                    i_win = not _beats(c, o, trump)
                    val = (1 if i_win else 0) + _solve_ab(rest_my, rest_opp, i_win, trump, memo, -1, 99)
                    worst = min(worst, val)
                scores[c] += worst
            else:
                i_win = _beats(lead_card, c, trump)
                val = (1 if i_win else 0) + _solve_ab(rest_my, tuple(sorted(opp)), i_win, trump, memo, -1, 99)
                scores[c] += val

        if time.time() - t0 > EG_TIME:
            break

    return max(legal, key=lambda c: scores[c])

# ---------------------------------------------------------------------------
# Main Hook & Safety Fallbacks
# ---------------------------------------------------------------------------
def _safe_move(view):
    try:
        hand = view.your_hand
        if view.current_trick:
            ls = view.current_trick[0][1][0]
            same = [c for c in hand if c[0] == ls]
            if same:
                return min(same, key=lambda c: c[1])
            return min(hand, key=lambda c: c[1])
    except Exception:
        pass
    return view.your_hand[0]

def _legal_ok(view, card):
    if card is None or card not in view.your_hand:
        return False
    if view.current_trick:
        ls = view.current_trick[0][1][0]
        if card[0] != ls and any(c[0] == ls for c in view.your_hand):
            return False
    return True

def _move(view):
    tid = _update(view)

    if view.phase == 1:
        card = _phase1_follow(view) if view.current_trick else _phase1_lead(view)
    else:
        card = None
        if len(view.your_hand) <= EG_HAND:
            card = _endgame_move(view)
        if card is None or not _legal_ok(view, card):
            card = _phase2_follow(view) if view.current_trick else _phase2_lead(view)

    if not _legal_ok(view, card):
        card = _safe_move(view)
    _after_move(view, card, tid)
    return card

def nextMove(gameState):
    try:
        return _move(gameState)
    except Exception:
        return _safe_move(gameState)
