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


import pytest

from pymeeus.BesselianElements import (BesselianElements,
                                       LunarBesselianElements, _places)
from pymeeus.Eclipse import Eclipse
from pymeeus.Epoch import Epoch


def test_no_eclipse_negative_u():
    """Regression: when u is negative and gamma is just beyond the
    no-eclipse threshold, the classifier must return 'No eclipse' rather
    than a spurious Partial with negative magnitude.

    With abs(u) in the cutoff the threshold is inflated, allowing these
    near-miss geometries to slip through as bogus partial eclipses."""

    for date in [(1953, 1, 15.0), (2051, 5, 10.0)]:
        epoch = Epoch(*date)

        _, meeus_kind, _, _, _ = Eclipse.solar_eclipse(epoch)
        assert meeus_kind == "No eclipse"

        be = BesselianElements(epoch)
        assert be.eclipse_type == "No eclipse", (
            "BesselianElements classified {} as '{}', "
            "expected 'No eclipse'".format(date, be.eclipse_type)
        )
        assert be.magnitude is None


def test_gamma_sign_southern_eclipses():
    """Regression: gamma must be negative for southern-hemisphere eclipses.

    sqrt(x*x + y*y) loses the sign of y, which encodes whether the shadow
    axis passes north (+) or south (-) of Earth's center."""

    southern = [
        (2023, 4, 20.0),
        (2021, 12, 4.0),
        (2010, 7, 11.0),
    ]
    for date in southern:
        epoch = Epoch(*date)
        _, _, _, gamma_meeus, _ = Eclipse.solar_eclipse(epoch)
        be = BesselianElements(epoch)
        assert gamma_meeus < 0.0, (
            "Expected negative Meeus gamma for {}".format(date)
        )
        assert be.gamma < 0.0, (
            "BesselianElements gamma={:+.5f} for {}, expected negative".format(
                be.gamma, date)
        )


def test_gamma_sign_northern_eclipses():
    """Verify gamma stays positive for northern-hemisphere eclipses."""

    northern = [
        (1993, 5, 21.0),
        (2024, 4, 8.0),
    ]
    for date in northern:
        epoch = Epoch(*date)
        _, _, _, gamma_meeus, _ = Eclipse.solar_eclipse(epoch)
        be = BesselianElements(epoch)
        assert gamma_meeus > 0.0, (
            "Expected positive Meeus gamma for {}".format(date)
        )
        assert be.gamma > 0.0, (
            "BesselianElements gamma={:+.5f} for {}, expected positive".format(
                be.gamma, date)
        )


def test_mu_matches_published_convention():
    """mu is the ephemeris hour angle (sidereal time at TT), as published.

    NASA (Espenak) lists mu0 = 89.59122 deg at t0 = 18h TDT for the
    2024 April 8 eclipse.  Evaluating sidereal time at UT1 instead would
    give a value lower by DeltaT * 15 arcsec/s, about 0.29 deg."""

    be = BesselianElements(Epoch(2024, 4, 8.0))
    assert be.t0 == 18
    assert abs(be.mu[0] - 89.59122) < 0.001


def test_window_after_2330_tt_stays_on_eclipse_day():
    """Regression: a maximum after 23:30 TT rounds t0 up to 0h of the next
    day.  Keeping the old date put the fit window 24 h before the eclipse,
    so t_max came from extrapolating the polynomials by a whole day."""

    be = BesselianElements(Epoch(225, 11, 17.0))
    assert be.t0 == 0
    assert abs(be.t_max) < 0.5


@pytest.mark.parametrize(
    "date, kind",
    [
        ((1909, 6, 17.0), "Hybrid"),
        ((1927, 1, 3.0), "Annular"),
        ((1966, 5, 20.0), "Annular"),
        ((2013, 11, 3.0), "Hybrid"),
    ],
)
def test_borderline_types_match_nasa_catalog(date, kind):
    """Types of eclipses near the total/hybrid/annular boundaries, as listed
    in NASA's Five Millennium Catalog.  Getting them right needs the canon's
    umbral lunar radius and the computed tan f2 instead of Meeus's 0.00464."""

    assert BesselianElements(Epoch(*date)).eclipse_type == kind


# Rows of the Five Millennium Catalog of Lunar Eclipses (NASA TP-2009-214173):
# date, TD of greatest eclipse, type, gamma, penumbral and umbral magnitude,
# and the penumbral, partial and total durations in minutes.
@pytest.mark.parametrize(
    "date, td, kind, gamma, pmag, umag, durations",
    [
        ((2025, 3, 14.0), (6, 59, 56), "Total", 0.3484, 2.2595, 1.1784,
         (362.6, 218.3, 65.4)),
        ((2023, 10, 28.0), (20, 15, 18), "Partial", 0.9471, 1.1181, 0.1220,
         (264.6, 77.4, None)),
        ((2023, 5, 5.0), (17, 24, 5), "Penumbral", -1.0349, 0.9636, -0.0457,
         (257.5, None, None)),
    ],
)
def test_lunar_matches_nasa_canon(date, td, kind, gamma, pmag, umag,
                                  durations):
    le = LunarBesselianElements(Epoch(*date))
    h, mi, s = td
    expected_max = Epoch(date[0], date[1], date[2] + (h + mi / 60.0
                                                      + s / 3600.0) / 24.0)

    assert le.eclipse_type == kind
    assert abs(le.t_max_epoch - expected_max) * 86400.0 < 20.0
    assert abs(le.gamma - gamma) < 0.002
    assert abs(le.penumbral_magnitude - pmag) < 0.003
    assert abs(le.umbral_magnitude - umag) < 0.003

    c = le.contacts
    for (first, last), expected in zip(
            (("P1", "P4"), ("U1", "U4"), ("U2", "U3")), durations):
        if expected is None:
            assert c[first] is None and c[last] is None
        else:
            assert abs((c[last] - c[first]) * 1440.0 - expected) < 0.5


@pytest.mark.parametrize("cls, date", [(BesselianElements, (2024, 4, 8.0)),
                                       (LunarBesselianElements,
                                        (2025, 3, 14.0))])
def test_from_places_with_the_builtin_places_is_the_builtin_fit(cls, date):
    builtin = cls(Epoch(*date))
    fitted = cls.from_places(_places, builtin.t_max_epoch.jde())
    assert vars(fitted) == vars(builtin)
