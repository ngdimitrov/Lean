import numpy as np
from sizer import VolatilityTargetSizer


def _approx(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_empty_active_returns_empty():
    s = VolatilityTargetSizer()
    assert s.compute_weights({}, []) == {}


def test_below_min_observations_excluded():
    s = VolatilityTargetSizer(min_observations=30)
    short = np.full(10, 0.01)
    assert s.compute_weights({"A": short}, ["A"]) == {}


def test_single_asset_scales_to_target_vol():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.02, 500)
    s = VolatilityTargetSizer(target_annual_vol=0.275, periods_per_year=365,
                              max_gross_leverage=10.0)
    w = s.compute_weights({"A": r}, ["A"])
    realized = abs(w["A"]) * r.std(ddof=1) * np.sqrt(365)
    _approx(realized, 0.275, tol=0.02)


def test_leverage_cap_binds_for_low_vol():
    r = np.full(100, 0.0)  # zero vol -> k falls back to 1.0, capped
    s = VolatilityTargetSizer(target_annual_vol=0.275, max_gross_leverage=1.0)
    w = s.compute_weights({"A": r}, ["A"])
    assert w["A"] <= 1.0 + 1e-9


def test_two_assets_portfolio_vol_hits_target():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.03, 400)
    b = rng.normal(0, 0.05, 400)
    s = VolatilityTargetSizer(target_annual_vol=0.275, max_gross_leverage=10.0)
    w = s.compute_weights({"A": a, "B": b}, ["A", "B"])
    R = np.column_stack([a, b])
    cov = np.cov(R, rowvar=False) * 365
    wv = np.array([w["A"], w["B"]])
    port_vol = np.sqrt(wv @ cov @ wv)
    _approx(port_vol, 0.275, tol=0.02)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASSED")
