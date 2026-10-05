# -*- coding: utf-8 -*-


# PyMeeus: Python module implementing astronomical algorithms.
# Copyright (C) 2021  Dagoberto Salazar
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


from pymeeus.base import TOL
from pymeeus.Eclipse import Eclipse
from pymeeus.Epoch import Epoch


# Reference gamma / magnitude values below are taken from NASA GSFC's Five
# Millennium Catalog of Solar and Lunar Eclipses (Espenak & Meeus):
#   solar: https://eclipse.gsfc.nasa.gov/SEcat5/SEcatalog.html
#   lunar: https://eclipse.gsfc.nasa.gov/LEcat5/LEcatalog.html
#
# Tolerance for comparing the approximate chapter-54 geometry against those
# canon values.  The method is only a truncated-series geocentric
# approximation, so gamma and the umbral magnitude agree with the canon to a
# few thousandths; 0.02 absorbs that while staying far tighter than the gap
# between any two eclipse-type boundaries, so a classification regression still
# trips the 'kind' assertions below.
CATALOG_TOL = 0.02


# Eclipse class

def test_eclipse_solar_1993_may():
    """Tests the method 'solar_eclipse()' on the 1993 May 21 partial solar
    eclipse (worked example 54.a of Meeus' book)"""

    epoch = Epoch(1993, 5, 21.0)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
    y, m, d, h, mi, s = max_epoch.get_full_date()
    mag = round(mag, 3)
    gamma = round(gamma, 4)
    u = round(u, 4)

    assert y == 1993, \
        "ERROR: 1st 'solar_eclipse()' test, 'year' value doesn't match"

    assert m == 5, \
        "ERROR: 2nd 'solar_eclipse()' test, 'month' value doesn't match"

    assert d == 21, \
        "ERROR: 3rd 'solar_eclipse()' test, 'day' value doesn't match"

    assert h == 14, \
        "ERROR: 4th 'solar_eclipse()' test, 'hour' value doesn't match"

    assert mi == 20, \
        "ERROR: 5th 'solar_eclipse()' test, 'minute' value doesn't match"

    assert kind == "Partial", \
        "ERROR: 6th 'solar_eclipse()' test, 'eclipse_type' doesn't match"

    assert abs(mag - 0.740) < TOL, \
        "ERROR: 7th 'solar_eclipse()' test, 'magnitude' value doesn't match"

    assert abs(gamma - 1.1348) < TOL, \
        "ERROR: 8th 'solar_eclipse()' test, 'gamma' value doesn't match"

    assert abs(u - 0.0097) < TOL, \
        "ERROR: 9th 'solar_eclipse()' test, 'u' value doesn't match"


def test_eclipse_lunar_1973_june():
    """Tests the method 'lunar_eclipse()' on the 1973 June 15 penumbral lunar
    eclipse (worked example 54.b of Meeus' book)"""

    epoch = Epoch(1973, 6, 15.0)
    result = Eclipse.lunar_eclipse(epoch)
    max_epoch, kind, umag, pmag, sd_par, sd_tot, sd_pen = result
    y, m, d, h, mi, s = max_epoch.get_full_date()
    umag = round(umag, 3)
    pmag = round(pmag, 3)
    sd_par = round(sd_par, 1)
    sd_tot = round(sd_tot, 1)
    sd_pen = round(sd_pen, 1)

    assert y == 1973, \
        "ERROR: 1st 'lunar_eclipse()' test, 'year' value doesn't match"

    assert m == 6, \
        "ERROR: 2nd 'lunar_eclipse()' test, 'month' value doesn't match"

    assert d == 15, \
        "ERROR: 3rd 'lunar_eclipse()' test, 'day' value doesn't match"

    assert h == 20, \
        "ERROR: 4th 'lunar_eclipse()' test, 'hour' value doesn't match"

    assert mi == 50, \
        "ERROR: 5th 'lunar_eclipse()' test, 'minute' value doesn't match"

    assert kind == "Penumbral", \
        "ERROR: 6th 'lunar_eclipse()' test, 'eclipse_type' doesn't match"

    assert abs(umag - (-0.609)) < TOL, \
        "ERROR: 7th 'lunar_eclipse()' test, 'umag' value doesn't match"

    assert abs(pmag - 0.462) < TOL, \
        "ERROR: 8th 'lunar_eclipse()' test, 'pmag' value doesn't match"

    assert abs(sd_par - 0.0) < TOL, \
        "ERROR: 9th 'lunar_eclipse()' test, 'sd_partial' value doesn't match"

    assert abs(sd_tot - 0.0) < TOL, \
        "ERROR: 10th 'lunar_eclipse()' test, 'sd_total' value doesn't match"

    assert abs(sd_pen - 101.5) < TOL, \
        "ERROR: 11th 'lunar_eclipse()' test, 'sd_penumbral' value doesn't\
            match"


def test_eclipse_solar_no_eclipse():
    """Tests that 'solar_eclipse()' returns 'No eclipse' when the closest new
    moon to the input epoch happens too far from a node for the Moon to cover
    any part of the Sun"""

    # The new moon nearest to 1993 February 21 is far from a node, so no
    # solar eclipse occurs at that lunation.  Because the |sin F| > 0.36
    # filter rejects it before any geometry is computed, the magnitude, gamma
    # and u are all reported as None (not computed) rather than a fake 0.0.
    epoch = Epoch(1993, 2, 21.0)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)

    assert kind == "No eclipse", \
        "ERROR: 1st 'solar_eclipse()' no-eclipse test, 'kind' doesn't match"

    assert mag is None, \
        "ERROR: 2nd 'solar_eclipse()' no-eclipse test, 'magnitude' not None"

    assert gamma is None, \
        "ERROR: 3rd 'solar_eclipse()' no-eclipse test, 'gamma' not None"

    assert u is None, \
        "ERROR: 4th 'solar_eclipse()' no-eclipse test, 'u' not None"


def test_eclipse_solar_total():
    """Catalog regression: a clear total solar eclipse must classify as
    'Total'.

    The 2017 August 21 'Great American Eclipse' (NASA canon: gamma = +0.4367,
    magnitude 1.0306) is central with a negative umbral-cone radius u, the
    discriminant for a total eclipse."""

    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(
        Epoch(2017, 8, 21.0))
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (2017, 8, 21), \
        "ERROR: 'solar_eclipse()' total test, picked wrong lunation"

    assert kind == "Total", \
        "ERROR: 'solar_eclipse()' total test, 'kind' is not 'Total'"

    assert u < 0.0, \
        "ERROR: 'solar_eclipse()' total test, u should be negative for a total"

    assert mag is None, \
        "ERROR: 'solar_eclipse()' total test, magnitude only set for partials"

    assert abs(gamma - 0.4367) < CATALOG_TOL, \
        "ERROR: 'solar_eclipse()' total test, 'gamma' off the NASA canon"


def test_eclipse_solar_annular():
    """Catalog regression: a clear annular solar eclipse must classify as
    'Annular'.

    The 2023 October 14 annular eclipse (NASA canon: gamma = +0.3753,
    magnitude 0.9520) is central with u above the 0.0047 annular threshold."""

    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(
        Epoch(2023, 10, 14.0))
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (2023, 10, 14), \
        "ERROR: 'solar_eclipse()' annular test, picked wrong lunation"

    assert kind == "Annular", \
        "ERROR: 'solar_eclipse()' annular test, 'kind' is not 'Annular'"

    assert u > 0.0047, \
        "ERROR: 'solar_eclipse()' annular test, u below the annular threshold"

    assert mag is None, \
        "ERROR: 'solar_eclipse()' annular test, 'magnitude' should be None"

    assert abs(gamma - 0.3753) < CATALOG_TOL, \
        "ERROR: 'solar_eclipse()' annular test, 'gamma' off the NASA canon"


def test_eclipse_solar_hybrid():
    """Catalog regression: a hybrid (annular-total) solar eclipse must classify
    as 'Hybrid'.

    The 2013 November 3 hybrid eclipse (NASA canon: gamma = +0.3272, magnitude
    1.0159) sits on the annular/total boundary: u is positive but smaller than
    omega = 0.00464 * sqrt(1 - gamma^2), which is exactly the test that
    separates 'Hybrid' from 'Annular'.  This is the tightest of the five
    classification branches and the most likely to rot silently."""

    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(
        Epoch(2013, 11, 3.0))
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (2013, 11, 3), \
        "ERROR: 'solar_eclipse()' hybrid test, picked wrong lunation"

    assert kind == "Hybrid", \
        "ERROR: 'solar_eclipse()' hybrid test, 'kind' is not 'Hybrid'"

    assert mag is None, \
        "ERROR: 'solar_eclipse()' hybrid test, magnitude only set for partials"

    assert abs(gamma - 0.3272) < CATALOG_TOL, \
        "ERROR: 'solar_eclipse()' hybrid test, 'gamma' off the NASA canon"


def test_eclipse_solar_no_eclipse_axis_miss():
    """Tests the computed-geometry 'No eclipse' branch of 'solar_eclipse()'.

    Unlike the early |sin F| > 0.36 filter (which returns before any geometry
    is computed), this branch fires when the syzygy is close enough to a node
    to compute the shadow geometry but the axis still passes outside the Earth
    (|gamma| > 1.5433 + u).  The new moon of 1960 February 26 misses by a clear
    margin (|gamma| ~ 1.62), so it is a genuine non-eclipse, yet gamma and u
    are real numbers while only the magnitude is None."""

    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(
        Epoch(1960, 2, 26.0))
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1960, 2, 26), \
        "ERROR: 'solar_eclipse()' axis-miss test, picked wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'solar_eclipse()' axis-miss test, 'kind' is not 'No eclipse'"

    assert mag is None, \
        "ERROR: 'solar_eclipse()' axis-miss test, 'magnitude' should be None"

    assert gamma is not None and u is not None, \
        "ERROR: 'solar_eclipse()' axis-miss test, geometry should be computed"

    assert abs(gamma) > 1.5433 + u, \
        "ERROR: 'solar_eclipse()' axis-miss test, axis should miss the Earth"


def test_eclipse_lunar_total():
    """Catalog regression: a clear total lunar eclipse must classify as
    'Total'.

    The 2019 January 21 total lunar eclipse (NASA canon: umbral magnitude
    1.1953) has an umbral magnitude above 1.0 and a non-zero total-phase
    semi-duration."""

    result = Eclipse.lunar_eclipse(Epoch(2019, 1, 21.0))
    max_epoch, kind, umag, pmag, sd_par, sd_tot, sd_pen = result
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (2019, 1, 21), \
        "ERROR: 'lunar_eclipse()' total test, picked wrong lunation"

    assert kind == "Total", \
        "ERROR: 'lunar_eclipse()' total test, 'kind' is not 'Total'"

    assert umag > 1.0, \
        "ERROR: 'lunar_eclipse()' total test, umag should exceed 1.0"

    assert sd_tot > 0.0, \
        "ERROR: 'lunar_eclipse()' total test, total phase should occur"

    assert abs(umag - 1.1953) < CATALOG_TOL, \
        "ERROR: 'lunar_eclipse()' total test, 'umag' off the NASA canon"


def test_eclipse_lunar_partial():
    """Catalog regression: a clear partial lunar eclipse must classify as
    'Partial'.

    The 2019 July 16 partial lunar eclipse (NASA canon: umbral magnitude
    0.6531) has an umbral magnitude between 0.0 and 1.0, so totality never
    occurs (sd_total is 0.0) but the partial phase does (sd_partial > 0)."""

    result = Eclipse.lunar_eclipse(Epoch(2019, 7, 16.0))
    max_epoch, kind, umag, pmag, sd_par, sd_tot, sd_pen = result
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (2019, 7, 16), \
        "ERROR: 'lunar_eclipse()' partial test, picked wrong lunation"

    assert kind == "Partial", \
        "ERROR: 'lunar_eclipse()' partial test, 'kind' is not 'Partial'"

    assert 0.0 < umag < 1.0, \
        "ERROR: 'lunar_eclipse()' partial test, umag not in the partial range"

    assert abs(sd_tot - 0.0) < TOL and sd_par > 0.0, \
        "ERROR: 'lunar_eclipse()' partial test, phase semi-durations wrong"

    assert abs(umag - 0.6531) < CATALOG_TOL, \
        "ERROR: 'lunar_eclipse()' partial test, 'umag' off the NASA canon"


def test_eclipse_lunar_no_eclipse_near_miss():
    """Tests the computed-magnitude 'No eclipse' branch of 'lunar_eclipse()'.

    Unlike the early |sin F| > 0.36 filter, this branch fires when the geometry
    is computed but the Moon misses even the penumbra (penumbral magnitude
    below 0).  The full moon of 1940 September 16 misses by a clear margin
    (pmag ~ -0.31), a genuine non-eclipse, yet both magnitudes are real numbers
    while the three semi-durations are None (not computed)."""

    result = Eclipse.lunar_eclipse(Epoch(1940, 9, 16.0))
    max_epoch, kind, umag, pmag, sd_par, sd_tot, sd_pen = result
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1940, 9, 16), \
        "ERROR: 'lunar_eclipse()' near-miss test, picked wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'lunar_eclipse()' near-miss test, 'kind' is not 'No eclipse'"

    assert umag is not None and pmag is not None, \
        "ERROR: 'lunar_eclipse()' near-miss test, magnitudes not computed"

    assert pmag < 0.0, \
        "ERROR: 'lunar_eclipse()' near-miss test, pmag should be negative"

    assert sd_par is None and sd_tot is None and sd_pen is None, \
        "ERROR: 'lunar_eclipse()' near-miss test, semi-durations not None"


def test_eclipse_solar_closest_new_moon():
    """Tests that 'solar_eclipse()' resolves to the new moon closest to the
    input epoch and not to the one suggested by the rounded decimal year.

    For 1990 January 10 the new moon on 1989 December 28 is about three days
    closer than the 1990 January 26 annular eclipse, so the latter (which the
    rounded-year selection used to return) must not be reported."""

    epoch = Epoch(1990, 1, 10.0)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1989, 12, 28), \
        "ERROR: 'solar_eclipse()' closest-new-moon test, picked wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'solar_eclipse()' closest-new-moon test, 'kind' doesn't match"


def test_eclipse_lunar_closest_full_moon():
    """Tests that 'lunar_eclipse()' resolves to the full moon closest to the
    input epoch and not to the one suggested by the rounded decimal year.

    For 1990 January 25 the full moon on 1990 January 11 is about two weeks
    closer than the 1990 February 9 total eclipse, so the latter (which the
    rounded-year selection used to return) must not be reported."""

    epoch = Epoch(1990, 1, 25.0)
    result = Eclipse.lunar_eclipse(epoch)
    max_epoch, kind = result[0], result[1]
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1990, 1, 11), \
        "ERROR: 'lunar_eclipse()' closest-full-moon test, wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'lunar_eclipse()' closest-full-moon test, 'kind' doesn't match"


def test_eclipse_solar_closest_corrected_time():
    """Tests that 'solar_eclipse()' compares the corrected (not the mean) phase
    times when picking the closest new moon.

    For 1919 May 14 20:42 TT the mean new moon of 1919 May 29 is marginally
    closer, but once the periodic correction is applied the 1919 April 30 new
    moon wins.  Selecting on mean times used to return the May eclipse."""

    epoch = Epoch(1919, 5, 14, 20, 42, 4.29)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1919, 4, 30), \
        "ERROR: 'solar_eclipse()' corrected-time test, picked wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'solar_eclipse()' corrected-time test, 'kind' doesn't match"


def test_eclipse_lunar_closest_corrected_time():
    """Tests that 'lunar_eclipse()' compares the corrected (not the mean) phase
    times when picking the closest full moon.

    For 1919 April 30 03:12 TT the mean full moon of 1919 May 15 is marginally
    closer, but once the periodic correction is applied the 1919 April 15 full
    moon wins.  Selecting on mean times used to return the May eclipse."""

    epoch = Epoch(1919, 4, 30, 3, 12, 5.94)
    result = Eclipse.lunar_eclipse(epoch)
    max_epoch, kind = result[0], result[1]
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert (y, m, d) == (1919, 4, 15), \
        "ERROR: 'lunar_eclipse()' corrected-time test, picked wrong lunation"

    assert kind == "No eclipse", \
        "ERROR: 'lunar_eclipse()' corrected-time test, 'kind' doesn't match"


def test_eclipse_solar_no_eclipse_corrected_time():
    """Tests that the 'No eclipse' result of 'solar_eclipse()' carries the
    corrected maximum-eclipse time even when the early |sin F| rejection fires.

    For 1993 February 21 the early branch used to return the uncorrected mean
    new moon (11:56 TT); it must instead report the corrected time (12:48 TT),
    matching the convention of the later rejection branches."""

    epoch = Epoch(1993, 2, 21.0)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert kind == "No eclipse", \
        "ERROR: 'solar_eclipse()' no-eclipse-time test, 'kind' doesn't match"

    assert (d, h, mi) == (21, 12, 48), \
        "ERROR: 'solar_eclipse()' no-eclipse-time test, time is not corrected"


def test_eclipse_lunar_no_eclipse_corrected_time():
    """Tests that the 'No eclipse' result of 'lunar_eclipse()' carries the
    corrected maximum-eclipse time even when the early |sin F| rejection fires.

    For 1993 February 6 the early branch used to return the uncorrected mean
    full moon (17:34 TT); it must instead report the corrected time (23:34 TT),
    matching the convention of the later rejection branches."""

    epoch = Epoch(1993, 2, 6.0)
    result = Eclipse.lunar_eclipse(epoch)
    max_epoch, kind = result[0], result[1]
    y, m, d, h, mi, s = max_epoch.get_full_date()

    assert kind == "No eclipse", \
        "ERROR: 'lunar_eclipse()' no-eclipse-time test, 'kind' doesn't match"

    assert (d, h, mi) == (6, 23, 34), \
        "ERROR: 'lunar_eclipse()' no-eclipse-time test, time is not corrected"
