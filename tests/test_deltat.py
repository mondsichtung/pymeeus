# -*- coding: utf-8 -*-


# PyMeeus: Python module implementing astronomical algorithms.
# Copyright (C) 2018  Dagoberto Salazar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math

import pytest

from pymeeus.DeltaT import (
    DeltaTTable,
    delta_t,
    set_delta_t_table,
)


DAY_S = 86400.0


def _synthetic_wobble_table():
    """Like _synthetic_iers_table() but with a realistic non-linear DeltaT
    wobble over a longer span, so the in-table interpolation actually exercises
    the fused-multiply-add path (a purely linear ramp does not). Built directly
    from (tt, val), so the wobble amplitude is irrelevant to leap detection."""
    base = 32.184 + 12.0
    start_mjd = 41684.0
    n = 800
    first_val = 43.4724
    tt = []
    val = []
    for k in range(n):
        tt.append(start_mjd + k + base / DAY_S + 2400000.5)
        val.append(first_val + 0.0007 * k
                   + 0.35 * math.sin(k * 0.23) + 0.12 * math.cos(k * 0.07))
    return DeltaTTable(tt, val)


def _julian_year_to_jde(j):
    """Inverse of jde_to_julian_year(): Julian year to Julian Ephemeris Date."""
    return j * 365.25 + 1721045.0


def _synthetic_iers_table():
    """Build a synthetic IERS-style DeltaT table whose START Julian year falls
    in the interior of a Table-S15 segment.

    The real ``finals2000A.all`` starts at UTC MJD 41684.0 (1973-01-02), whose
    Julian year (about 1973.0034) lands strictly inside the S15 segment
    [1971, 1974]. We replicate that exact start so the boundary-containing
    segment is the one that must be kept and continuity-adjusted, matching the
    real-world case. DeltaT is a slow linear ramp; only the start matters for
    the boundary behaviour."""
    tt_minus_utc = 32.184 + 12.0  # base offset used by parse_finals2000a
    start_mjd = 41684.0
    n = 400
    first_val = 43.4724
    tt = []
    val = []
    for k in range(n):
        mjd = start_mjd + k
        tt.append(mjd + tt_minus_utc / DAY_S + 2400000.5)
        val.append(first_val + 0.0005 * k)
    return DeltaTTable(tt, val)


def test_deltat_continuity_at_iers_table_start():
    """DeltaT must be continuous where the IERS table begins.

    Just left of the table start, DeltaT comes from the Table-S15 (2020)
    splines whose boundary segment is adjusted to meet the first IERS value
    (skyfield build_delta_t()); just at the start it comes from the IERS
    interpolation. The two must agree: there must be no step at the seam. The
    historical bug rejected the boundary-containing segment and produced a
    ~0.094 s jump here."""
    table = _synthetic_iers_table()
    set_delta_t_table(table)

    jde_start = table.tt[0]
    # A point a hair to the left of the table start, so it takes the S15 path.
    j_start = (jde_start - 1721045.0) / 365.25
    jde_just_before = _julian_year_to_jde(j_start - 1.0e-7)

    val_just_before = delta_t(jde_just_before)
    val_at_start = delta_t(jde_start)

    jump = abs(val_at_start - val_just_before)
    assert jump < 0.01, \
        "ERROR: DeltaT discontinuity of {0} s at the IERS table start".format(
            jump)
    # The IERS interpolation at the start equals the table's first value, and
    # the adjusted S15 segment must converge to it.
    assert abs(val_at_start - table.val[0]) < 1.0e-9, \
        "ERROR: DeltaT at the IERS start does not match the first table value"
    assert abs(val_just_before - table.val[0]) < 1.0e-3, \
        "ERROR: adjusted S15 segment does not meet the first IERS value"


def test_deltat_window_1971_to_1973_reference_values():
    """In [1971, 1973.003] (left of the IERS table, inside the boundary S15
    segment) DeltaT must match the boundary-adjusted spline.

    These reference values come from the boundary-containing S15 segment
    [1971, 1974] with its linear term adjusted so the spline reaches the first
    IERS value (43.4724) at the boundary, exactly as skyfield build_delta_t()
    does. The historical bug evaluated the previous segment [1968, 1971]
    extrapolated past its own right edge, giving errors up to ~0.11 s here."""
    table = _synthetic_iers_table()
    set_delta_t_table(table)

    # year -> expected DeltaT (computed from the adjusted boundary segment).
    expected = {
        1971.0: 40.951,
        1972.0: 42.19452333212584,
        1972.5: 42.83071555374431,
        1973.0: 43.46804666425167,
    }
    for year, ref in expected.items():
        got = delta_t(_julian_year_to_jde(year))
        assert abs(got - ref) < 1.0e-9, \
            "ERROR: DeltaT at year {0} is {1}, expected {2}".format(
                year, got, ref)


def test_deltat_matches_skyfield_across_window():
    """Cross-check against skyfield build_delta_t() over [1971, 1973.003], from
    the identical synthetic table. Skipped when skyfield is not installed."""
    skyfield_timelib = pytest.importorskip("skyfield.timelib")
    numpy = pytest.importorskip("numpy")

    table = _synthetic_iers_table()
    set_delta_t_table(table)

    tt_arr = numpy.array(table.tt)
    val_arr = numpy.array(table.val)
    sf = skyfield_timelib.build_delta_t((tt_arr, val_arr))

    boundary = (table.tt[0] - 1721045.0) / 365.25
    years = numpy.linspace(1971.0, boundary, 500)
    for year in years:
        jde = _julian_year_to_jde(year)
        got = delta_t(jde)
        ref = float(sf(numpy.array([jde]))[0])
        assert abs(got - ref) < 1.0e-9, \
            "ERROR: DeltaT at year {0} is {1}, skyfield {2}".format(
                year, got, ref)


def test_deltat_empty_table_extrapolation_unchanged():
    """The extrapolation-only path (no IERS table) is independent of the
    boundary-segment logic; its seam at the Table-S15 end (Julian year 2019.0)
    must stay smooth. This guards against the fix accidentally touching the
    empty-table boundary, which lands exactly on a segment edge."""
    set_delta_t_table(DeltaTTable.empty())

    just_before = delta_t(_julian_year_to_jde(2019.0 - 1.0e-5))
    at_edge = delta_t(_julian_year_to_jde(2019.0))
    assert abs(at_edge - just_before) < 1.0e-3, \
        "ERROR: extrapolation-only DeltaT is discontinuous at Julian year 2019"
    # Known value of the last S15 segment evaluated at its right edge (t = 1).
    assert abs(at_edge - 69.24) < 1.0e-6, \
        "ERROR: extrapolation-only DeltaT at Julian year 2019 changed"


def test_deltat_patched_paths_match_skyfield():
    """Bit-exact (0 ULP) vs skyfield at exactly the paths the skyfield-alignment
    patches touched. Each point here diverged by 1 ULP before its patch:

    - the SMH-2016 parabola derivative, used by the left / far-left / right
      splines (now baked as ``2*32.5/width * t`` like skyfield's
      ``Splines.derivative``, not ``2*32.5*t/width``);
    - the in-table ``numpy.interp`` lerp (now a fused multiply-add);
    - the right-extrapolation segment seam (now ``j < j1``).

    Skipped when skyfield is not installed."""
    skyfield_timelib = pytest.importorskip("skyfield.timelib")
    numpy = pytest.importorskip("numpy")

    table = _synthetic_wobble_table()
    set_delta_t_table(table)
    sf = skyfield_timelib.build_delta_t(
        (numpy.array(table.tt), numpy.array(table.val)))

    def sky(jde):
        return float(sf(numpy.array([jde]))[0])

    def assert_bit_exact(jde, label):
        got = delta_t(jde)
        ref = sky(jde)
        assert got.hex() == ref.hex(), \
            "ERROR: {0}: fork {1!r} != skyfield {2!r}".format(label, got, ref)

    # (1) parabola derivative — left and far-left splines.
    for year in (-1505.8, -1300.0, -800.0):
        assert_bit_exact(_julian_year_to_jde(year),
                         "parabola-deriv left @ {0}".format(year))

    # ... and the right-side spline, which also uses the parabola derivative,
    # plus (3) the seam at exactly j == j1 (the `<` vs `<=` segment choice).
    j0 = (table.tt[-1] - 1721045.0) / 365.25
    j1 = math.floor((j0 + 800.0) / 100.0) * 100.0
    assert_bit_exact(_julian_year_to_jde(j0 + 50.0), "parabola-deriv right")
    assert_bit_exact(_julian_year_to_jde(j1), "right seam @ j1")

    # (2) in-table interp FMA: find an in-table point where the fused
    # multiply-add differs from the naive product-then-sum, and prove the fork
    # takes the FMA branch (== skyfield) and NOT the naive one.
    found = False
    for jde in numpy.linspace(table.tt[0], table.tt[-1], 50000):
        jde = float(jde)
        lo, hi = 0, len(table.tt) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if table.tt[mid] <= jde:
                lo = mid
            else:
                hi = mid
        slope = (table.val[hi] - table.val[lo]) / (table.tt[hi] - table.tt[lo])
        fma = math.fma(slope, jde - table.tt[lo], table.val[lo])
        naive = slope * (jde - table.tt[lo]) + table.val[lo]
        if fma.hex() != naive.hex():
            got = delta_t(jde)
            assert got.hex() == fma.hex(), "ERROR: in-table interp is not an FMA"
            assert got.hex() == sky(jde).hex(), \
                "ERROR: in-table interp does not match skyfield"
            assert got.hex() != naive.hex(), \
                "ERROR: in-table interp matched the naive (non-FMA) form"
            found = True
            break
    assert found, \
        "ERROR: no FMA-sensitive in-table point found; enrich the wobble table"


def test_deltat_empty_table_right_matches_skyfield_splines():
    """No-IERS right extrapolation (empty table) anchors at the Table-S15 end.
    skyfield has no finals-absent mode, so the oracle is skyfield's own spline
    primitives: the S15-end slope from ``Splines(s15).derivative`` and a Hermite
    patch to the parabola via ``build_spline_given_ends``. The per-term S15-end
    slope patch (``3a3/w + 2a2/w + a1/w``, not ``(3a3+2a2+a1)/w``) makes the fork
    match these bit-for-bit. Skipped when skyfield is not installed."""
    curvelib = pytest.importorskip("skyfield.curvelib")
    sky_timelib = pytest.importorskip("skyfield.timelib")
    functions = pytest.importorskip("skyfield.functions")
    pytest.importorskip("numpy")

    set_delta_t_table(DeltaTTable.empty())

    s15 = functions.load_bundled_npy("delta_t.npz")["Table-S15.2020.txt"]
    s = curvelib.Splines(s15)
    sd = s.derivative
    parabola = sky_timelib.delta_t_parabola_stephenson_morrison_hohenkerk_2016
    pd = parabola.derivative
    x0 = float(s.upper[-1])                       # Table-S15 end, Julian year 2019
    x1 = (x0 + 800.0) // 100.0 * 100.0            # patch end (multiple of 100)
    right = curvelib.build_spline_given_ends(
        x0, float(s(x0)), float(sd(x0)),
        x1, float(parabola(x1)), float(pd(x1)))

    def eval_spline(spline, j):
        a3, a2, a1, a0 = spline[2], spline[3], spline[4], spline[5]
        t = (j - spline[0]) / (spline[1] - spline[0])
        return ((a3 * t + a2) * t + a1) * t + a0

    for year in (2100.0, 2300.0, 2600.0):
        got = delta_t(_julian_year_to_jde(year))
        ref = eval_spline(right, year)
        assert got.hex() == ref.hex(), \
            "ERROR: no-IERS right @ {0}: fork {1!r} != skyfield-spline {2!r}".format(
                year, got, ref)
