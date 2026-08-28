# Fixed textbook Kimura-2-parameter clock (separate fix for build_clean_k2p.k2p).
# Original bugs: transitions inflated 1.5x, transversions 3x, -0.5 instead of -0.25 on 2nd log.
# Textbook: P=n_trans/S, Q=n_transv/S; d=-0.5*ln(1-2P-Q)-0.25*ln(1-2Q); sat when 2P+Q>=1 or 2Q>=1.
import numpy as np

_BASE_MAP = np.full(256, 4, dtype=np.int8)
_BASE_MAP[ord("A")] = 0
_BASE_MAP[ord("C")] = 1
_BASE_MAP[ord("G")] = 2
_BASE_MAP[ord("T")] = 3

def seq_codes(seq):
    return _BASE_MAP[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]

def k2p_textbook(ref_codes, query_codes, cap=2.5):
    valid = (ref_codes != 4)
    n_valid = int(valid.sum())
    if n_valid < 3:
        return 0.0
    diff = valid & (ref_codes != query_codes)
    is_trans = diff & (
        ((ref_codes == 0) & (query_codes == 2))
        | ((ref_codes == 2) & (query_codes == 0))
        | ((ref_codes == 1) & (query_codes == 3))
        | ((ref_codes == 3) & (query_codes == 1))
    )
    n_trans = int(is_trans.sum())
    n_transv = int(diff.sum()) - n_trans
    p_trans = n_trans / n_valid
    p_transv = n_transv / n_valid
    if 2 * p_trans + p_transv >= 1.0 or 2 * p_transv >= 1.0:
        return cap
    dist = -0.5 * np.log(1 - 2 * p_trans - p_transv) - 0.25 * np.log(1 - 2 * p_transv)
    return float(min(dist, cap))

def _make_seqs(n_sites, n_transition, n_transversion, seed=0):
    rng = np.random.RandomState(seed)
    ref = rng.randint(0, 4, size=n_sites)
    query = ref.copy()
    positions = rng.choice(n_sites, size=n_transition + n_transversion, replace=False)
    for idx in range(n_transition):
        base = ref[positions[idx]]
        query[positions[idx]] = (base + 2) % 4
    for idx in range(n_transversion):
        base = ref[positions[n_transition + idx]]
        query[positions[n_transition + idx]] = (base + 1) % 4
    return ref, query

def _expected(n_sites, n_trans, n_transv):
    p = n_trans / n_sites
    q = n_transv / n_sites
    if 2*p + q >= 1 or 2*q >= 1:
        return 2.5
    return float(-0.5*np.log(1-2*p-q) - 0.25*np.log(1-2*q))

def run_unit_test():
    print("=" * 70)
    print("  UNIT TEST: k2p_textbook vs textbook ground truth")
    print("=" * 70)
    all_ok = True
    for (ns, nt, nv) in [(1000,10,5),(1000,30,15),(1000,60,30),(1000,80,40),(1000,120,60),(2000,50,25),(500,15,7)]:
        ref, q = _make_seqs(ns, nt, nv)
        got = k2p_textbook(ref, q)
        exp = _expected(ns, nt, nv)
        ok = abs(got - exp) < 1e-9
        all_ok = all_ok and ok
        print("  sites=%5d n_trans=%4d n_transv=%4d | fixed=%.5f expected=%.5f [%s]" % (ns, nt, nv, got, exp, "OK" if ok else "FAIL"))
    print("=" * 70)
    print("  RESULT: ALL PASS" if all_ok else "  RESULT: SOME FAILED")
    print("=" * 70)
    return all_ok



if __name__ == '__main__':
    run_unit_test()
