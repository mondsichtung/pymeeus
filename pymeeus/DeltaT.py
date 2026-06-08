# -*- coding: utf-8 -*-

# PyMeeus: Python module implementing astronomical algorithms.
# Copyright (C) 2018  Dagoberto Salazar
#
# This file is part of PyMeeus.
#
# PyMeeus is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# PyMeeus is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PyMeeus. If not, see <https://www.gnu.org/licenses/>.

"""IERS-based DeltaT (TT - UT1).

This module computes DeltaT = TT - UT1 from the IERS ``finals2000A.all`` daily
Earth-orientation table (linearly interpolated within range), falling back to
the Morrison-Stephenson-Hohenkerk-Zawilski Table-S15 (2020) historical splines
and the Stephenson-Morrison-Hohenkerk 2016 long-term parabola outside it. The
extrapolation logic matches Skyfield's ``build_delta_t()``.

This replaces the older Espenak/Meeus polynomial approximation that
:meth:`Epoch.tt2ut` used to return.

The ``finals2000A.all`` file is downloaded lazily from the IERS on first use
and cached in the system temp directory for 24 hours. When no network and no
cache are available, only the Table-S15 / parabola extrapolation is used, which
is accurate for historical and far-future dates but loses the IERS-observed
accuracy of the recent decades.

Data source:
    https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all

References:
    skyfield/timelib.py:build_delta_t()
    skyfield/data/iers.py:build_timescale_arrays()
    Morrison, Stephenson, Hohenkerk & Zawilski (2021), Table-S15.2020
"""

import math
import os
import ssl
import tempfile
import time
import urllib.request


"""Seconds per day."""
DAY_S = 86400.0

"""URL of the IERS finals2000A.all rapid Earth-orientation product."""
FINALS2000A_URL = (
    "https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all"
)

"""Cached finals2000A.all copy and its maximum age before re-downloading."""
_CACHE_PATH = os.path.join(tempfile.gettempdir(), "pymeeus_finals2000A.all")
_CACHE_MAX_AGE_S = 86400.0  # 24 hours
_DOWNLOAD_TIMEOUT_S = 30.0

# Morrison-Stephenson Table-S15 (2020): 58 cubic spline segments covering
# Julian years -720 to 2019. Each segment is [x0, x1, a3, a2, a1, a0], evaluated
# as t = (J - x0)/(x1 - x0); value = ((a3*t + a2)*t + a1)*t + a0.
#
# Source: skyfield bundled delta_t.npz key "Table-S15.2020.txt"; Morrison,
# Stephenson, Hohenkerk & Zawilski (2021).
TABLE_S15_2020 = [
    [-720.0, -100.0, 409.16, 776.247, -9999.586, 20371.848],
    [-100.0, 400.0, -503.433, 1303.151, -5822.27, 11557.668],
    [400.0, 1000.0, 1085.087, -298.291, -5671.519, 6535.116],
    [1000.0, 1150.0, -25.346, 184.811, -753.21, 1650.393],
    [1150.0, 1300.0, -24.641, 108.771, -459.628, 1056.647],
    [1300.0, 1500.0, -29.414, 61.953, -421.345, 681.149],
    [1500.0, 1600.0, 16.197, -6.572, -192.841, 292.343],
    [1600.0, 1650.0, 3.018, 10.505, -78.697, 109.127],
    [1650.0, 1720.0, -2.127, 38.333, -68.089, 43.952],
    [1720.0, 1800.0, -37.939, 41.731, 2.507, 12.068],
    [1800.0, 1810.0, 1.918, -1.126, -3.481, 18.367],
    [1810.0, 1820.0, -3.812, 4.629, 0.021, 15.678],
    [1820.0, 1830.0, 3.25, -6.806, -2.157, 16.516],
    [1830.0, 1840.0, -0.096, 2.944, -6.018, 10.804],
    [1840.0, 1850.0, -0.539, 2.658, -0.416, 7.634],
    [1850.0, 1855.0, -0.883, 0.261, 1.642, 9.338],
    [1855.0, 1860.0, 1.558, -2.389, -0.486, 10.357],
    [1860.0, 1865.0, -2.477, 2.284, -0.591, 9.04],
    [1865.0, 1870.0, 2.72, -5.148, -3.456, 8.255],
    [1870.0, 1875.0, -0.914, 3.011, -5.593, 2.371],
    [1875.0, 1880.0, -0.039, 0.269, -2.314, -1.126],
    [1880.0, 1885.0, 0.563, 0.152, -1.893, -3.21],
    [1885.0, 1890.0, -1.438, 1.842, 0.101, -4.388],
    [1890.0, 1895.0, 1.871, -2.474, -0.531, -3.884],
    [1895.0, 1900.0, -0.232, 3.138, 0.134, -5.017],
    [1900.0, 1905.0, -1.257, 2.443, 5.715, -1.977],
    [1905.0, 1910.0, 0.72, -1.329, 6.828, 4.923],
    [1910.0, 1915.0, -0.825, 0.831, 6.33, 11.142],
    [1915.0, 1920.0, 0.262, -1.643, 5.518, 17.479],
    [1920.0, 1925.0, 0.008, -0.856, 3.02, 21.617],
    [1925.0, 1930.0, 0.127, -0.831, 1.333, 23.789],
    [1930.0, 1935.0, 0.142, -0.449, 0.052, 24.418],
    [1935.0, 1940.0, 0.702, -0.022, -0.419, 24.164],
    [1940.0, 1945.0, -1.106, 2.086, 1.645, 24.426],
    [1945.0, 1950.0, 0.614, -1.232, 2.499, 27.05],
    [1950.0, 1953.0, -0.277, 0.22, 1.127, 28.932],
    [1953.0, 1956.0, 0.631, -0.61, 0.737, 30.002],
    [1956.0, 1959.0, -0.799, 1.282, 1.409, 30.76],
    [1959.0, 1962.0, 0.507, -1.115, 1.577, 32.652],
    [1962.0, 1965.0, 0.199, 0.406, 0.868, 33.621],
    [1965.0, 1968.0, -0.414, 1.002, 2.275, 35.093],
    [1968.0, 1971.0, 0.202, -0.242, 3.035, 37.956],
    [1971.0, 1974.0, -0.229, 0.364, 3.157, 40.951],
    [1974.0, 1977.0, 0.172, -0.323, 3.199, 44.244],
    [1977.0, 1980.0, -0.192, 0.193, 3.069, 47.291],
    [1980.0, 1983.0, 0.081, -0.384, 2.878, 50.361],
    [1983.0, 1986.0, -0.165, -0.14, 2.354, 52.936],
    [1986.0, 1989.0, 0.448, -0.637, 1.577, 54.984],
    [1989.0, 1992.0, -0.276, 0.708, 1.648, 56.373],
    [1992.0, 1995.0, 0.11, -0.121, 2.235, 58.453],
    [1995.0, 1998.0, -0.313, 0.21, 2.324, 60.678],
    [1998.0, 2001.0, 0.109, -0.729, 1.804, 62.898],
    [2001.0, 2004.0, 0.199, -0.402, 0.674, 64.083],
    [2004.0, 2007.0, -0.017, 0.194, 0.466, 64.553],
    [2007.0, 2010.0, -0.084, 0.144, 0.804, 65.197],
    [2010.0, 2013.0, 0.128, -0.109, 0.839, 66.061],
    [2013.0, 2016.0, -0.095, 0.277, 1.007, 66.92],
    [2016.0, 2019.0, -0.139, -0.007, 1.277, 68.109],
]


def _round_half_away_from_zero(x):
    """Round to nearest integer, ties away from zero (matches Rust f64::round)."""
    if x >= 0.0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


class DeltaTTable(object):
    """A tabulated DeltaT (TT - UT1) series: parallel ``tt`` (JD_TT) and ``val``
    (DeltaT seconds) arrays, sorted ascending by ``tt``."""

    def __init__(self, tt=None, val=None):
        self.tt = tt if tt is not None else []
        self.val = val if val is not None else []

    @classmethod
    def empty(cls):
        """An empty table (no IERS data; extrapolation-only)."""
        return cls([], [])

    def is_empty(self):
        return len(self.tt) == 0

    @classmethod
    def parse_finals2000a(cls, content):
        """Parse the IERS ``finals2000A.all`` fixed-width text into a DeltaT
        table.

        Extracts UTC MJD (columns 8-15) and UT1-UTC (columns 59-68), detects
        leap seconds from jumps > 0.9 s in UT1-UTC, and computes
        DeltaT = TT - UT1 = (32.184 + cumulative_leap_seconds + 12) - (UT1-UTC).
        The leap-second counting here is purely internal to deriving DeltaT.

        :param content: Contents of finals2000A.all.
        :type content: str

        :returns: A :class:`DeltaTTable`.
        :rtype: :class:`DeltaTTable`
        """
        utc_mjds = []
        dut1s = []
        for line in content.splitlines():
            # 0-indexed slices: MJD = [7:15], UT1-UTC = [58:68].
            if len(line) < 68:
                continue
            try:
                mjd = float(line[7:15].strip())
                dut1 = float(line[58:68].strip())
            except ValueError:
                continue
            utc_mjds.append(mjd)
            dut1s.append(dut1)

        if not utc_mjds:
            return cls.empty()

        n = len(utc_mjds)
        # Detect leap seconds: jumps > 0.9 s in UT1-UTC.
        leap_mask = [False] * n
        for i in range(1, len(dut1s)):
            if dut1s[i] - dut1s[i - 1] > 0.9:
                leap_mask[i] = True

        # Cumulative leap seconds + base offset (32.184 + 12.0 = 44.184). The
        # 12.0 is the TAI-UTC offset at the finals2000A start (1973-01-02).
        tt_minus_utc = []
        cum_leaps = 0.0
        for is_leap in leap_mask:
            if is_leap:
                cum_leaps += 1.0
            tt_minus_utc.append(cum_leaps + 32.184 + 12.0)

        tt = []
        val = []
        for i in range(n):
            daily_tt = utc_mjds[i] + tt_minus_utc[i] / DAY_S + 2400000.5
            daily_dt = _round_half_away_from_zero(
                (tt_minus_utc[i] - dut1s[i]) * 1e7) / 1e7
            tt.append(daily_tt)
            val.append(daily_dt)

        return cls(tt, val)

    def interp(self, jde):
        """Linearly interpolate DeltaT within the table range, mirroring
        ``numpy.interp``. Returns ``None`` if ``jde`` is outside the range."""
        n = len(self.tt)
        if n == 0 or jde < self.tt[0] or jde > self.tt[n - 1]:
            return None
        lo = 0
        hi = n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.tt[mid] <= jde:
                lo = mid
            else:
                hi = mid
        # numpy.interp evaluates slope*(x - xl) + yl as a fused multiply-add
        # (the C compiler contracts it), so match it with math.fma to stay
        # bit-identical to skyfield's in-table ΔT.
        slope = (self.val[hi] - self.val[lo]) / (self.tt[hi] - self.tt[lo])
        return math.fma(slope, jde - self.tt[lo], self.val[lo])

    def last_value_and_slope(self):
        """Last value and one-year slope, for right-side spline extrapolation."""
        n = len(self.val)
        lookback = min(n, 366)
        y0 = self.val[n - 1]
        slope = (self.val[n - 1] - self.val[n - lookback]) * lookback / 365.0
        return (y0, slope)


def jde_to_julian_year(jde):
    """Convert a Julian Ephemeris Date to a Julian year for the long-term
    parabola. Ref: skyfield/timelib.py:1081."""
    return (jde - 1721045.0) / 365.25


def _smh2016_parabola(j):
    """Stephenson-Morrison-Hohenkerk 2016 long-term parabola at Julian year j."""
    t = (j - 1825.0) / 100.0
    return (32.5 * t + 0.0) * t + (-320.0)


def _smh2016_parabola_deriv(j):
    """Derivative of the SMH2016 parabola (seconds per Julian year).

    Bit-matches skyfield's ``Splines.derivative``, which bakes the coefficient
    ``2*32.5/100`` at build time and evaluates ``coeff * t`` (not ``2*32.5*t/100``,
    which rounds differently by up to 1 ULP). Ref: skyfield/curvelib.py."""
    width = 100.0
    t = (j - 1825.0) / width
    return (2.0 * 32.5 / width) * t


def _build_spline(x0, y0, slope0, x1, y1, slope1):
    """Build a cubic Hermite spline given endpoint values and slopes.

    Returns ``(x0, width, a3, a2, a1, a0)`` evaluated as
    ``((a3*t + a2)*t + a1)*t + a0`` with ``t = (x - x0)/width``.
    Ref: skyfield/curvelib.py:build_spline_given_ends()."""
    width = x1 - x0
    s0 = slope0 * width
    s1 = slope1 * width
    a0 = y0
    a1 = s0
    a2 = -2.0 * s0 - s1 - 3.0 * y0 + 3.0 * y1
    a3 = s0 + s1 + 2.0 * y0 - 2.0 * y1
    return (x0, width, a3, a2, a1, a0)


def _eval_spline(spline, j):
    """Evaluate a cubic spline segment at Julian year j."""
    x0, width, a3, a2, a1, a0 = spline
    t = (j - x0) / width
    return ((a3 * t + a2) * t + a1) * t + a0


def _extrapolate_right(j, j0, y0, slope, patch_width):
    """Right-side extrapolation: Hermite spline from (j0, y0, slope) blending
    to the SMH2016 parabola. Ref: skyfield/timelib.py:build_delta_t()."""
    j1 = math.floor((j0 + patch_width) / 100.0) * 100.0
    # Strict `<`: at exactly j == j1 skyfield's segment search picks the
    # far_right segment (t=0), not the first spline at t=1; match that seam.
    if j < j1:
        spline = _build_spline(
            j0, y0, slope,
            j1, _smh2016_parabola(j1), _smh2016_parabola_deriv(j1))
        return _eval_spline(spline, j)
    # Second ("far_right") spline; skyfield clamps to the last segment beyond j2.
    parabola_width = 100.0
    j2 = j1 + parabola_width
    spline = _build_spline(
        j1, _smh2016_parabola(j1), _smh2016_parabola_deriv(j1),
        j2, _smh2016_parabola(j2), _smh2016_parabola_deriv(j2))
    return _eval_spline(spline, j)


def _extrapolate_left(j):
    """Left-side spline connecting the long-term parabola to the Table-S15
    start (-720), then a clamped far-left parabola segment. Returns ``None``
    for j > S15 start. Ref: skyfield/timelib.py:build_delta_t()."""
    patch_width = 800.0
    parabola_width = 100.0

    first = TABLE_S15_2020[0]
    s15_x0 = first[0]                 # -720.0
    s15_width = first[1] - first[0]   # 620.0
    s15_a1 = first[4]
    s15_a0 = first[5]
    s15_val = s15_a0                  # at t=0: a0
    s15_deriv = s15_a1 / s15_width    # derivative at t=0: a1/width

    left_x1 = s15_x0                  # -720.0
    left_x0 = left_x1 - patch_width   # -1520.0
    far_left_x1 = left_x0             # -1520.0
    far_left_x0 = far_left_x1 - parabola_width  # -1620.0

    if j > left_x1:
        return None

    if j >= left_x0:
        spline = _build_spline(
            left_x0, _smh2016_parabola(left_x0),
            _smh2016_parabola_deriv(left_x0),
            left_x1, s15_val, s15_deriv)
        return _eval_spline(spline, j)
    # Far-left: pure parabola segment, clamped for j < far_left_x0.
    spline = _build_spline(
        far_left_x0, _smh2016_parabola(far_left_x0),
        _smh2016_parabola_deriv(far_left_x0),
        far_left_x1, _smh2016_parabola(far_left_x1),
        _smh2016_parabola_deriv(far_left_x1))
    return _eval_spline(spline, j)


def _effective_end_point(table):
    """Right-side anchor: IERS table end if available, else Table-S15 end."""
    if not table.is_empty():
        j = jde_to_julian_year(table.tt[-1])
        val, slope = table.last_value_and_slope()
        return (j, val, slope)
    last = TABLE_S15_2020[-1]
    end_j = last[1]
    a3, a2, a1, a0 = last[2], last[3], last[4], last[5]
    end_val = ((a3 + a2) + a1) + a0           # t = 1
    width = last[1] - last[0]
    # Per-term divide, matching skyfield's derivative-spline coefficients
    # (3*a3/w, 2*a2/w, a1/w) summed at t=1 — not (3a3+2a2+a1)/w, which rounds
    # differently by up to 1 ULP.
    end_slope = 3.0 * a3 / width + 2.0 * a2 / width + a1 / width
    return (end_j, end_val, end_slope)


def _s15_2020_boundary_segment_index(truncate_j):
    """Index of the last Table-S15 segment kept when the table is truncated at
    Julian year ``truncate_j`` (the IERS table start). Mirrors skyfield's
    ``i = searchsorted(s15_table[0], x); s15_table = s15_table[:, :i]``: keep
    every segment whose LEFT edge is strictly below ``truncate_j``, so the
    boundary-containing segment (its right edge above ``truncate_j``) is kept
    and becomes the last one. Ref: skyfield/timelib.py:build_delta_t()."""
    cap = 0
    for i, seg in enumerate(TABLE_S15_2020):
        if seg[0] < truncate_j:
            cap = i
        else:
            break
    return cap


def _find_s15_2020_segment_index(j, truncate_j):
    """Index of the Table-S15 segment used to evaluate Julian year ``j`` when
    the table is truncated at ``truncate_j``. Returns the segment that contains
    ``j``, clamped to the boundary segment so that ``j`` in
    ``[last_left_edge, truncate_j)`` is evaluated on the boundary-containing
    segment (the one whose linear term is adjusted for continuity). Returns
    ``None`` when ``j`` is outside ``[TABLE_S15_2020[0][0], truncate_j]``."""
    first_x0 = TABLE_S15_2020[0][0]
    if j < first_x0 or j > truncate_j:
        return None
    cap = _s15_2020_boundary_segment_index(truncate_j)
    seg_idx = 0
    for i, seg in enumerate(TABLE_S15_2020):
        if j >= seg[0]:
            seg_idx = i
        else:
            break
    return min(seg_idx, cap)


def _eval_table_s15_2020_adjusted(j, iers):
    """Evaluate Table-S15 at Julian year ``j``, with the boundary segment's
    linear term adjusted to meet the first IERS DeltaT value when an IERS table
    is loaded. Ref: skyfield/timelib.py:build_delta_t()."""
    if not iers.is_empty():
        truncate_j = jde_to_julian_year(iers.tt[0])
        adjust = True
    else:
        truncate_j = TABLE_S15_2020[-1][1]  # 2019.0
        adjust = False

    seg_idx = _find_s15_2020_segment_index(j, truncate_j)
    if seg_idx is None:
        return None
    seg = TABLE_S15_2020[seg_idx]
    x0 = seg[0]
    width = seg[1] - seg[0]
    a3, a2, a1, a0 = seg[2], seg[3], seg[4], seg[5]

    # The boundary-containing segment (the last one kept after truncating at
    # ``truncate_j``) gets its linear term adjusted so the spline meets the
    # first IERS DeltaT value at the boundary, matching skyfield.
    boundary_idx = _s15_2020_boundary_segment_index(truncate_j)
    if adjust and seg_idx == boundary_idx and seg[0] <= truncate_j:
        t_boundary = (truncate_j - x0) / width
        if t_boundary > 0.0:
            current_y = (((a3 * t_boundary + a2) * t_boundary + a1)
                         * t_boundary + a0)
            desired_y = iers.val[0]
            a1 = a1 + (desired_y - current_y) / t_boundary

    t = (j - x0) / width
    return ((a3 * t + a2) * t + a1) * t + a0


def _delta_t_from_table(jde, table):
    """Compute DeltaT (TT - UT1) for ``jde`` against a given table, mirroring
    skyfield's branch structure: in-table interpolation, then right-side
    spline, then Table-S15 (boundary-adjusted), then left-side spline."""
    v = table.interp(jde)
    if v is not None:
        return v

    j = jde_to_julian_year(jde)
    patch_width = 800.0

    # Right of the IERS table.
    if not table.is_empty() and jde > table.tt[-1]:
        j0 = jde_to_julian_year(table.tt[-1])
        y0, slope = table.last_value_and_slope()
        return _extrapolate_right(j, j0, y0, slope, patch_width)

    # Left of the IERS table: Table-S15 (boundary-adjusted to IERS start).
    v = _eval_table_s15_2020_adjusted(j, table)
    if v is not None:
        return v

    # Post-S15 right extrapolation when no IERS table is loaded.
    end_j, end_val, end_slope = _effective_end_point(table)
    if j > end_j:
        return _extrapolate_right(j, end_j, end_val, end_slope, patch_width)

    # Before Table-S15 (-720): left spline -> clamped far-left parabola.
    v = _extrapolate_left(j)
    if v is not None:
        return v

    return _smh2016_parabola(j)


# Lazily-loaded global IERS table (None until first use or an explicit set).
_table = None
_table_loaded = False


def set_delta_t_table(table):
    """Install a custom DeltaT table, bypassing the lazy download. Pass an empty
    table (:meth:`DeltaTTable.empty`) to force extrapolation-only mode."""
    global _table, _table_loaded
    _table = table
    _table_loaded = True


def get_delta_t_table():
    """Return the active DeltaT table, lazily downloading finals2000A on first
    use. Falls back to an empty (extrapolation-only) table when offline."""
    global _table, _table_loaded
    if not _table_loaded:
        content = _fetch_finals2000a()
        _table = (DeltaTTable.parse_finals2000a(content)
                  if content is not None else DeltaTTable.empty())
        _table_loaded = True
    return _table


def is_delta_t_table_loaded():
    """Whether a non-empty IERS table is currently loaded."""
    return _table is not None and not _table.is_empty()


def _ssl_context():
    """SSL context for the IERS download. Uses the ``certifi`` CA bundle when
    available (fixing the common macOS "unable to get local issuer certificate"
    failure), otherwise the system default."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _fetch_finals2000a():
    """Return finals2000A.all contents, downloading to the temp cache when the
    cache is missing or older than 24 h. Returns the cached copy on download
    failure, or ``None`` if neither network nor cache is available."""
    need_download = True
    if os.path.exists(_CACHE_PATH):
        age = time.time() - os.path.getmtime(_CACHE_PATH)
        need_download = age > _CACHE_MAX_AGE_S

    if need_download:
        try:
            with urllib.request.urlopen(
                    FINALS2000A_URL, timeout=_DOWNLOAD_TIMEOUT_S,
                    context=_ssl_context()) as resp:
                data = resp.read().decode("ascii", errors="replace")
            if data:
                with open(_CACHE_PATH, "w") as f:
                    f.write(data)
                return data
        except Exception:
            pass  # fall through to any cached copy

    try:
        with open(_CACHE_PATH, "r") as f:
            return f.read()
    except OSError:
        return None


def download_finals2000a(force=False):
    """Explicitly (re)download finals2000A.all into the cache and reload the
    global table. Returns ``True`` on success (a non-empty table is loaded).

    :param force: Re-download even if the cache is still fresh.
    :type force: bool

    :rtype: bool
    """
    global _table, _table_loaded
    if force:
        try:
            os.remove(_CACHE_PATH)
        except OSError:
            pass
    content = _fetch_finals2000a()
    _table = (DeltaTTable.parse_finals2000a(content)
              if content is not None else DeltaTTable.empty())
    _table_loaded = True
    return not _table.is_empty()


def delta_t(jde):
    """DeltaT = TT - UT1, in seconds, for the given Julian Ephemeris Date.

    Uses the IERS finals2000A table within range (downloaded lazily and cached),
    and Table-S15 (2020) / SMH-2016-parabola extrapolation outside it.

    :param jde: Julian Ephemeris Date (TT).
    :type jde: float

    :returns: DeltaT in seconds.
    :rtype: float
    """
    return _delta_t_from_table(jde, get_delta_t_table())
