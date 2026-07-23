
"""
German Whist Master Engine (PIMC Solver + Dynamic Heuristics)

RENAME THIS FILE to your assigned team name before dropping it into ./germanwhist/players/.
"""

import random

SUITS = ["H", "D", "C", "S"]
FULL_DECK = [(s, r) for s in SUITS for r in range(2, 15)]

try:
    from engine import resolve_trick as _engine_resolve_trick
except Exception:
    _engine_resolve_trick = None


def resolve_trick(lead_card, follow_card, trump_suit):
    """True if lead_card beats follow_card."""
    if _engine_resolve_trick is not None:
        return _engine_resolve_trick(lead_card, follow_card, trump_suit)
    lead_suit, lead_rank = lead_card
    follow_suit, follow_rank = follow_card
    if follow_suit == lead_suit:
        return lead_rank >= follow_rank
    elif follow_suit == trump_suit:
        return False
    else:
        return True


def legal_cards(hand, lead_card):
    if lead_card is None:
        return list(hand)
    lead_suit = lead_card[0]
    same_suit = [c for c in hand if c[0] == lead_suit]
    return same_suit if same_suit else list(hand)


def _legal_set(hand_fs, lead_card):
    if lead_card is None:
        return set(hand_fs)
    suit = lead_card[0]
    same = {c for c in hand_fs if c[0] == suit}
    return same if same else set(hand_fs)


# ---------------------------------------------------------------------------
# Persistent Memory Management
# ---------------------------------------------------------------------------
_mem = {"played": set(), "ready": False}


def _new_game_reset():
    _mem["played"] = set()
    _mem["ready"] = True


def _sync_memory(view):
    if view.phase == 1 and view.stock_remaining == 25 and len(view.your_hand) == 13:
        _new_game_reset()
    elif not _mem["ready"]:
        _new_game_reset()

    for _player, card in view.current_trick:
        _mem["played"].add(card)


def _record_play(card):
    _mem["played"].add(card)


def _unseen_pool(view):
    known = set(view.your_hand)
    known |= _mem["played"]
    known |= {c for _, c in view.current_trick}
    if view.face_up_card is not None:
        known.add(view.face_up_card)
    return [c for c in FULL_DECK if c not in known]


def _card_value(card, trump_suit):
    suit, rank = card
    return rank + (7 if suit == trump_suit else 0)


def _best_of_suit_unseen(view, suit, trump_suit):
    pool = _unseen_pool(view)
    ranks = [r for (s, r) in pool if s == suit]
    return max(ranks) if ranks else 0


# ---------------------------------------------------------------------------
# Phase 1 Logic
# ---------------------------------------------------------------------------
def _want_win_faceup(view):
    trump = view.trump_suit
    face = view.face_up_card
    if face is None:
        return False

    pool = _unseen_pool(view)
    avg_pool_value = (
        sum(_card_value(c, trump) for c in pool) / len(pool) if pool else 8.0
    )
    face_value = _card_value(face, trump)
    my_trumps = sum(1 for c in view.your_hand if c[0] == trump)

    if face[0] == trump and my_trumps < 6:
        return True
    return face_value >= avg_pool_value + 1.5


def _phase1_lead(view):
    trump = view.trump_suit
    hand = view.your_hand
    want_win = _want_win_faceup(view)

    if want_win:
        trumps = sorted([c for c in hand if c[0] == trump], key=lambda c: c[1])
        if trumps:
            return trumps[0]
        return max(hand, key=lambda c: c[1])

    non_trumps = [c for c in hand if c[0] != trump]
    if non_trumps:
        suit_counts = {}
        for c in non_trumps:
            suit_counts[c[0]] = suit_counts.get(c[0], 0) + 1
        best_suit = max(suit_counts, key=lambda s: suit_counts[s])
        candidates = [c for c in non_trumps if c[0] == best_suit]
        return min(candidates, key=lambda c: c[1])
    return min(hand, key=lambda c: c[1])


def _phase1_follow(view):
    trump = view.trump_suit
    lead_card = view.current_trick[0][1]
    legal = legal_cards(view.your_hand, lead_card)
    want_win = _want_win_faceup(view)

    winners = [c for c in legal if not resolve_trick(lead_card, c, trump)]

    if want_win and winners:
        return min(winners, key=lambda c: _card_value(c, trump))

    if not want_win:
        losers = [c for c in legal if resolve_trick(lead_card, c, trump)]
        non_trump_losers = [c for c in losers if c[0] != trump]
        if non_trump_losers:
            return min(non_trump_losers, key=lambda c: c[1])
        if losers:
            return min(losers, key=lambda c: c[1])

    return min(legal, key=lambda c: _card_value(c, trump))


# ---------------------------------------------------------------------------
# Phase 2 Heuristics (Fallback)
# ---------------------------------------------------------------------------
def _phase2_lead_heuristic(view):
    trump = view.trump_suit
    hand = view.your_hand
    non_trump = [c for c in hand if c[0] != trump]

    sure_things = [
        c for c in non_trump if c[1] > _best_of_suit_unseen(view, c[0], trump)
    ]
    if sure_things:
        return min(sure_things, key=lambda c: c[1])

    if non_trump:
        suit_counts = {}
        for c in non_trump:
            suit_counts[c[0]] = suit_counts.get(c[0], 0) + 1
        shortest_suit = min(suit_counts, key=lambda s: suit_counts[s])
        candidates = [c for c in non_trump if c[0] == shortest_suit]
        return min(candidates, key=lambda c: c[1])

    return min(hand, key=lambda c: c[1])


def _phase2_follow_heuristic(view):
    trump = view.trump_suit
    lead_card = view.current_trick[0][1]
    legal = legal_cards(view.your_hand, lead_card)

    winners = [c for c in legal if not resolve_trick(lead_card, c, trump)]

    if winners:
        non_trump_winners = [c for c in winners if c[0] != trump]
        if non_trump_winners:
            return min(non_trump_winners, key=lambda c: c[1])

        my_trumps = [c for c in view.your_hand if c[0] == trump]
        tricks_left = len(view.your_hand)
        if len(my_trumps) >= 2 or tricks_left <= 4:
            return min(winners, key=lambda c: c[1])
        non_winners = [c for c in legal if c not in winners]
        if non_winners:
            return min(non_winners, key=lambda c: _card_value(c, trump))
        return min(winners, key=lambda c: c[1])

    return min(legal, key=lambda c: _card_value(c, trump))


# ---------------------------------------------------------------------------
# Phase 2: Fixed Exact Search (PIMC Zero-Sum Solver)
# ---------------------------------------------------------------------------
EXACT_SEARCH_MAX_CARDS = 6  # Kept light to ensure immediate speed
PIMC_SAMPLES = 8            # Fast sampling per move decision


def _solve(my_hand, opp_hand, leader, trump, memo):
    """Corrected Minimax solver with guaranteed hand convergence."""
    if not my_hand or not opp_hand:
        return 0

    key = (my_hand, opp_hand, leader)
    if key in memo:
        return memo[key]

    if leader == "me":
        best = -1
        for c in _legal_set(my_hand, None):
            worst = 999
            opp_legal = _legal_set(opp_hand, c)
            for oc in opp_legal:
                i_win = resolve_trick(c, oc, trump)
                next_leader = "me" if i_win else "opp"
                sub = _solve(my_hand - {c}, opp_hand - {oc}, next_leader, trump, memo)
                total = (1 if i_win else 0) + sub
                if total < worst:
                    worst = total
            if worst > best and worst != 999:
                best = worst
        memo[key] = max(0, best)
        return memo[key]
    else:
        best = 999
        for oc in _legal_set(opp_hand, None):
            best_resp = -1
            my_legal = _legal_set(my_hand, oc)
            for c in my_legal:
                i_win = not resolve_trick(oc, c, trump)
                next_leader = "me" if i_win else "opp"
                sub = _solve(my_hand - {c}, opp_hand - {oc}, next_leader, trump, memo)
                total = (1 if i_win else 0) + sub
                if total > best_resp:
                    best_resp = total
            if best_resp < best and best_resp != -1:
                best = best_resp
        memo[key] = max(0, best)
        return memo[key]


def _search_lead(my_hand, unseen_pool, trump, samples):
    my_fs = frozenset(my_hand)
    candidates = sorted(my_fs)
    opp_size = len(my_hand)
    if len(unseen_pool) < opp_size:
        return None

    totals = {c: 0.0 for c in candidates}
    for _ in range(samples):
        opp_hand = frozenset(random.sample(unseen_pool, opp_size))
        memo = {}
        for c in candidates:
            worst = 999
            for oc in _legal_set(opp_hand, c):
                i_win = resolve_trick(c, oc, trump)
                next_leader = "me" if i_win else "opp"
                sub = _solve(my_fs - {c}, opp_hand - {oc}, next_leader, trump, memo)
                total = (1 if i_win else 0) + sub
                if total < worst:
                    worst = total
            totals[c] += worst if worst != 999 else 0

    return max(candidates, key=lambda c: totals[c])


def _search_follow(my_hand, lead_card, unseen_pool, trump, samples):
    my_fs = frozenset(my_hand)
    legal = sorted(_legal_set(my_fs, lead_card))
    opp_size = len(my_hand)
    if len(unseen_pool) < opp_size:
        return None

    totals = {c: 0.0 for c in legal}
    for _ in range(samples):
        opp_hand = frozenset(random.sample(unseen_pool, opp_size))
        memo = {}
        for c in legal:
            i_win = not resolve_trick(lead_card, c, trump)
            next_leader = "me" if i_win else "opp"
            # Deduct opponent hand matching lead card
            possible_opp_leads = [oc for oc in opp_hand if oc == lead_card]
            opp_rem = opp_hand - {possible_opp_leads[0]} if possible_opp_leads else opp_hand - {list(opp_hand)[0]}
            
            sub = _solve(my_fs - {c}, opp_rem, next_leader, trump, memo)
            totals[c] += (1 if i_win else 0) + sub

    return max(legal, key=lambda c: totals[c])


def _phase2_lead(view):
    hand = view.your_hand
    if len(hand) <= EXACT_SEARCH_MAX_CARDS:
        pool = _unseen_pool(view)
        move = _search_lead(hand, pool, view.trump_suit, PIMC_SAMPLES)
        if move is not None:
            return move
    return _phase2_lead_heuristic(view)


def _phase2_follow(view):
    hand = view.your_hand
    if len(hand) <= EXACT_SEARCH_MAX_CARDS:
        lead_card = view.current_trick[0][1]
        pool = _unseen_pool(view)
        move = _search_follow(hand, lead_card, pool, view.trump_suit, PIMC_SAMPLES)
        if move is not None:
            return move
    return _phase2_follow_heuristic(view)


# ---------------------------------------------------------------------------
# Main Public Entrypoint
# ---------------------------------------------------------------------------
def nextMove(gameState):
    view = gameState
    _sync_memory(view)

    if view.phase == 1:
        card = _phase1_follow(view) if view.current_trick else _phase1_lead(view)
    else:
        card = _phase2_follow(view) if view.current_trick else _phase2_lead(view)

    _record_play(card)
    return card
