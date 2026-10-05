
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


from math import (sin, cos, asin, atan2, sqrt, tan, pi, degrees, radians,
                  copysign, hypot)

from pymeeus.Angle import Angle
from pymeeus.Coordinates import (
    ecliptical2equatorial,
    nutation_longitude,
    true_obliquity,
)
from pymeeus.Eclipse import Eclipse
from pymeeus.Epoch import Epoch, JDE2000, DAY2SEC
from pymeeus.Moon import PERIODIC_TERMS_B_TABLE, PERIODIC_TERMS_LR_TABLE
from pymeeus.Sun import Sun


"""
.. module:: BesselianElements
   :synopsis: Compute Besselian elements for solar and lunar eclipses from
      VSOP87/ELP
   :license: GNU Lesser General Public License v3 (LGPLv3)

.. moduleauthor:: Mondsichtung.de

Algorithm
---------

This module computes Besselian elements for solar and lunar eclipses from
first principles.  The solar elements follow the method described by
Friedrich Bessel in 1824.

The implementation uses:

- **Sun positions** from the VSOP87 theory (Bretagnon & Francou, 1988),
  via pymeeus.Sun (complete VSOP87 series).

- **Moon positions** from the periodic terms of Meeus ch. 47, which is
  based on the ELP-2000/82 theory (Chapront-Touze & Chapront, 1983) with
  ~120 of the largest periodic terms retained.  The mean arguments are those
  of Chapront, Chapront-Touze & Francou (2002), as in the Five Millennium
  Canons, and the positions are corrected for light travel time.

- **Besselian element computation** translated from Greg Miller's JavaScript
  implementation ``BesselianElementGenerator.js`` (public domain, 2023),
  downloaded from:
  https://astrogreg.com/eclipsesGeneratingBesselianElements/example01/BesselianElementGenerator.js
  which follows the procedure described in the Explanatory Supplement to
  the Astronomical Almanac.

At nine hourly time points from t0-4h to t0+4h, the geocentric apparent
positions of the Sun and Moon are converted to the eight Besselian elements
(x, y, d, mu, l1, l2, tan f1, tan f2) using shadow cone geometry.
Third-degree polynomials are then fitted to these samples via least-squares
regression.  The window covers the penumbral contacts, which lie up to
about 3.5 h from t0.

Either class can instead be fitted to another ephemeris's apparent places,
around a given estimate of greatest eclipse, with ``from_places``.

Lunar eclipses use the same Sun and Moon positions and the same nine
samples.  The elements are the Moon's position relative to the axis of the
Earth's shadow (x, y) and the angular radii of the penumbra (f1), the umbra
(f2) and the Moon (f3).  The shadow radii follow Danjon's rule, as in the
Five Millennium Canon of Lunar Eclipses.

The same approach was used by Espenak & Meeus in the Five Millennium Canon
of Solar Eclipses (NASA TP-2006-214141), except they retained far more
ELP periodic terms (coefficients >= 0.0005 arcsec, thousands of terms).

Accuracy of the time of greatest eclipse and of gamma against JPL DE441
(via Skyfield), for the 451 solar eclipses of 1900-2100 and 1000 sampled
from the years 0-3000.  Greatest eclipse is the closest approach of the
shadow axis to the Earth's centre.  Mean (maximum) absolute errors:

=========================  ==============  ============  ===============
Method                     Time 1900-2100  Time 0-3000   Gamma 0-3000
=========================  ==============  ============  ===============
Eclipse.py (Meeus ch. 54)  21 s (70 s)     28 s (116 s)  0.0007 (0.0033)
BesselianElements          5 s (19 s)      8 s (47 s)    0.0002 (0.0011)
=========================  ==============  ============  ===============

References
----------

- Bessel, F.W. (1824), "Ueber die Bestimmung der geographischen Laenge
  durch Beobachtungen der Sternbedeckungen," Astronomische Nachrichten.

- Bretagnon, P. & Francou, G. (1988), "Planetary theories in rectangular
  and spherical variables. VSOP87 solutions," A&A 202, 309.

- Chapront-Touze, M. & Chapront, J. (1983), "The lunar ephemeris
  ELP-2000," A&A 124, 50.

- Chapront, J., Chapront-Touze, M. & Francou, G. (2002), "A new
  determination of lunar orbital parameters, precession constant and
  tidal acceleration from LLR measurements," A&A 387, 700.

- Espenak, F. & Meeus, J. (2006), "Five Millennium Canon of Solar
  Eclipses: -1999 to +3000," NASA TP-2006-214141.

- Espenak, F. & Meeus, J. (2009), "Five Millennium Canon of Lunar
  Eclipses: -1999 to +3000," NASA TP-2009-214172.

- Laskar, J. (1986), "Secular terms of classical planetary theories using
  the results of general theory," A&A 157, 59.

- Meeus, J. (1991), "Astronomical Algorithms," Willmann-Bell, ch. 47, 54.

- Miller, G. (2023), ``BesselianElementGenerator.js``, astrogreg.com
  (public domain).
"""

AU_KM = 149597870.7
EARTH_RADIUS_KM = 6378.137
LIGHT_SPEED_KM_S = 299792.458
# Radius of the Sun in Earth radii, from a semi-diameter of 959.63" at 1 AU.
# The canons do not state it, but it reproduces the tan f1 and tan f2 of
# NASA's published elements to 1e-7.  The IAU's 695700 km does not.
SUN_K = AU_KM * sin(radians(959.63 / 3600.0)) / EARTH_RADIUS_KM
# The two lunar radii of the Five Millennium Canons (solar sec. 1.5, lunar
# sec. 1.3): a mean over the limb for solar penumbral contacts and for lunar
# eclipses, and a mean minimum radius for the solar umbra so that annular
# eclipses are not reported as total.
MOON_K = 0.2724880
MOON_K_UMBRA = 0.272281

DAY2HOURS = 24.0
# Sample times in hours from t0
_HOURS = [float(t) for t in range(-4, 5)]

# Mean arguments of the Moon from Chapront, Chapront-Touze & Francou (2002,
# Table 4), as in the Five Millennium Canons: a constant in degrees and the
# coefficients of t to t^4 in arcseconds, t in Julian centuries from J2000.0.
# With Meeus's own mean arguments, eclipses of the years 0-500 come about a
# minute early against DE441.
_W1 = (218 + 18 / 60 + 59.8782 / 3600,
       (1732559343.3328, -6.8700, 0.006604, -0.00003169))
_D = (297 + 51 / 60 + 0.6902 / 3600,
      (1602961601.0312, -6.8498, 0.006595, -0.00003184))
_M = (357 + 31 / 60 + 44.7744 / 3600,
      (129596581.0733, -0.5529, 0.000147, 0.00000015))
_MPRIME = (134 + 57 / 60 + 48.2264 / 3600,
           (1717915923.0024, 31.3939, 0.051651, -0.00024470))
_F = (93 + 16 / 60 + 19.5517 / 3600,
      (1739527263.2179, -13.2293, -0.001021, 0.00000417))
# W1 is measured from the fixed equinox of J2000.0.  The general precession
# in longitude of Laskar (1986), with T in Julian millennia, moves it to the
# mean equinox of date of VSOP87, which it matches to 0.5" back to the year
# -2000.  The IAU 1976 expression is 60" off there.
_PRECESSION = (0.0, (50290.966, 111.1971, 0.07732, -0.235316, -0.0018055,
                     0.00017451, 1.3095e-5, 2.424e-7, -4.759e-8, -8.66e-10))


def _sun_equatorial(epoch):
    """Return Sun apparent RA (rad), Dec (rad), distance (Earth radii)."""
    lon, lat, r_au = Sun.apparent_geocentric_position(epoch)
    eps = true_obliquity(epoch)
    ra_angle, dec_angle = ecliptical2equatorial(lon, lat, eps)
    ra = ra_angle.to_positive().rad()
    dec = dec_angle.rad()
    r = r_au * AU_KM / EARTH_RADIUS_KM
    return ra, dec, r


def _mean_argument(arg, t):
    """Evaluate one of the mean arguments above, in degrees, at t."""
    deg, coeffs = arg
    arcsec = 0.0
    for c in reversed(coeffs):
        arcsec = (arcsec + c) * t
    return deg + arcsec / 3600.0


def _moon_series(t, lprime, d, m, mprime, f):
    """Return the Moon's longitude and latitude (degrees) and distance (km)
    from the periodic terms of Meeus ch. 47, given the mean arguments in
    degrees and t in Julian centuries from J2000.0."""
    lprime, d, m, mprime, f = (radians(x % 360.0)
                               for x in (lprime, d, m, mprime, f))
    a1 = radians((119.75 + 131.849 * t) % 360.0)
    a2 = radians((53.09 + 479264.290 * t) % 360.0)
    a3 = radians((313.45 + 481266.484 * t) % 360.0)
    # Eccentricity of the Earth's orbit, applied once per unit of M
    e = 1.0 + (-0.002516 - 0.0000074 * t) * t
    sigmal = 0.0
    sigmar = 0.0
    for cd, cm, cmp, cf, cl, cr in PERIODIC_TERMS_LR_TABLE:
        arg = cd * d + cm * m + cmp * mprime + cf * f
        sigmal += e ** abs(cm) * cl * sin(arg)
        sigmar += e ** abs(cm) * cr * cos(arg)
    sigmab = 0.0
    for cd, cm, cmp, cf, cb in PERIODIC_TERMS_B_TABLE:
        arg = cd * d + cm * m + cmp * mprime + cf * f
        sigmab += e ** abs(cm) * cb * sin(arg)
    sigmal += 3958.0 * sin(a1) + 1962.0 * sin(lprime - f) + 318.0 * sin(a2)
    sigmab += (-2235.0 * sin(lprime) + 382.0 * sin(a3)
               + 175.0 * sin(a1 - f) + 175.0 * sin(a1 + f)
               + 127.0 * sin(lprime - mprime) - 115.0 * sin(lprime + mprime))
    return (degrees(lprime) + sigmal / 1000000.0, sigmab / 1000000.0,
            385000.56 + sigmar / 1000.0)


def _moon_ecliptical(epoch):
    """Return the Moon's geometric longitude and latitude (degrees), referred
    to the mean equinox of date, and its distance in kilometers."""
    t = (epoch - JDE2000) / 36525.0
    lprime = _mean_argument(_W1, t) + _mean_argument(_PRECESSION, t / 10.0)
    return _moon_series(t, lprime, _mean_argument(_D, t),
                        _mean_argument(_M, t), _mean_argument(_MPRIME, t),
                        _mean_argument(_F, t))


def _moon_equatorial(epoch):
    """Return Moon apparent RA (rad), Dec (rad), distance (Earth radii)."""
    # The Moon is seen where it was when its light left, about 1.3 s
    # earlier.  Meeus ch. 47 builds this into the constant of L', which is
    # 0.7" smaller than W1 above.
    dist_km = _moon_ecliptical(epoch)[2]
    lon, lat, dist_km = _moon_ecliptical(
        Epoch(epoch.jde() - dist_km / LIGHT_SPEED_KM_S / DAY2SEC))
    lon = Angle(lon) + nutation_longitude(epoch)
    ra_angle, dec_angle = ecliptical2equatorial(lon, Angle(lat),
                                                true_obliquity(epoch))
    ra = ra_angle.to_positive().rad()
    dec = dec_angle.rad()
    r = dist_km / EARTH_RADIUS_KM
    return ra, dec, r


def _greenwich_apparent_sidereal_time(epoch):
    """Return Greenwich apparent sidereal time in radians.

    Evaluated at TT as if it were UT1 (ephemeris sidereal time), which is
    the convention of published Besselian elements.
    """
    eps = true_obliquity(epoch)
    dpsi = nutation_longitude(epoch)
    return epoch.apparent_sidereal_time(eps, dpsi) * 2.0 * pi


def _project(ra, dec, r, a, d):
    """Return the coordinates (x, y, z) of the body at (ra, dec, r) in the
    frame whose z axis points to right ascension a and declination d."""
    h = ra - a
    x = r * cos(dec) * sin(h)
    y = r * (sin(dec) * cos(d) - cos(dec) * sin(d) * cos(h))
    z = r * (sin(dec) * sin(d) + cos(dec) * cos(d) * cos(h))
    return x, y, z


def _places(epoch):
    """Return the Sun's and the Moon's apparent (ra, dec, distance) in
    radians and Earth radii, and the Greenwich apparent sidereal time in
    radians, at the TT epoch: the contract of ``from_places``."""
    return (_sun_equatorial(epoch), _moon_equatorial(epoch),
            _greenwich_apparent_sidereal_time(epoch))


def _elements_at_instant(epoch):
    """Compute raw Besselian elements at a single instant (TT)."""
    return _elements_from_places(*_places(epoch))


def _elements_from_places(sun, moon, theta):
    """Compute raw Besselian elements from the places of ``_places``.

    Returns dict with x, y, d (deg), l1, l2, mu (deg), tanf1, tanf2.
    """
    sun_ra, sun_dec, sun_r = sun
    moon_ra, moon_dec, moon_r = moon

    sun_xyz = [
        sun_r * cos(sun_dec) * cos(sun_ra),
        sun_r * cos(sun_dec) * sin(sun_ra),
        sun_r * sin(sun_dec),
    ]
    moon_xyz = [
        moon_r * cos(moon_dec) * cos(moon_ra),
        moon_r * cos(moon_dec) * sin(moon_ra),
        moon_r * sin(moon_dec),
    ]

    gx = sun_xyz[0] - moon_xyz[0]
    gy = sun_xyz[1] - moon_xyz[1]
    gz = sun_xyz[2] - moon_xyz[2]
    g = sqrt(gx * gx + gy * gy + gz * gz)

    a = atan2(gy, gx)
    d = asin(gz / g)

    mu = (theta - a) % (2 * pi)

    x, y, z = _project(moon_ra, moon_dec, moon_r, a, d)

    sinf1 = (SUN_K + MOON_K) / g
    sinf2 = (SUN_K - MOON_K_UMBRA) / g

    tanf1 = tan(asin(sinf1))
    tanf2 = tan(asin(sinf2))

    c1 = z + MOON_K / sinf1
    c2 = z - MOON_K_UMBRA / sinf2

    l1 = c1 * tanf1
    l2 = c2 * tanf2

    return {
        "x": x,
        "y": y,
        "d": degrees(d),
        "l1": l1,
        "l2": l2,
        "mu": degrees(mu),
        "tanf1": tanf1,
        "tanf2": tanf2,
    }


def _lunar_elements_at_instant(epoch):
    """Compute raw lunar eclipse elements at a single instant (TT)."""
    return _lunar_elements_from_places(_sun_equatorial(epoch),
                                       _moon_equatorial(epoch))


def _lunar_elements_from_places(sun, moon):
    """Compute raw lunar eclipse elements from the Sun's and the Moon's
    places of ``_places``.

    Returns dict with x, y, f1, f2, f3, all in degrees as seen from the
    Earth's centre.
    """
    # The shadow axis points away from the apparent (aberrated) Sun.  This
    # matches the NASA canon; the geometric Sun shifts times by about 40 s.
    sun_ra, sun_dec, sun_r = sun
    moon_ra, moon_dec, moon_r = moon
    x, y, z = _project(moon_ra, moon_dec, moon_r, sun_ra + pi, -sun_dec)

    # Rescale (x, y) so that hypot(x, y) is the angular distance between the
    # Moon and the shadow axis, which is what Danjon's radii are compared to.
    rho = hypot(x, y)
    scale = degrees(atan2(rho, z)) / rho

    pi_moon = asin(1.0 / moon_r)
    pi_sun = asin(1.0 / sun_r)
    s_sun = asin(SUN_K / sun_r)

    # Danjon: 1.01 = 1 + 1/85 (atmosphere) - 1/594 (Earth's flattening)
    return {
        "x": x * scale,
        "y": y * scale,
        "f1": degrees(1.01 * pi_moon + pi_sun + s_sun),
        "f2": degrees(1.01 * pi_moon + pi_sun - s_sun),
        "f3": degrees(asin(MOON_K / moon_r)),
    }


def _eliminate_angle_wrap(values):
    """Fix 360-degree jumps in a sequence of angle values (degrees)."""
    for i in range(len(values) - 1):
        diff = values[i + 1] - values[i]
        if abs(diff) > 180:
            values[i + 1] -= 360 * (1 if diff > 0 else -1)


def _fit_poly3(t_values, y_values):
    """Fit a degree-3 polynomial to the data points using least squares.

    Returns [c0, c1, c2, c3] where y = c0 + c1*t + c2*t^2 + c3*t^3.
    Uses normal equations (A^T A)x = A^T b with Gauss elimination.
    """
    n = len(t_values)
    ncoef = 4

    ata = [[0.0] * ncoef for _ in range(ncoef)]
    atb = [0.0] * ncoef

    for i in range(n):
        row = [1.0]
        for _ in range(1, ncoef):
            row.append(row[-1] * t_values[i])
        for j in range(ncoef):
            for k in range(ncoef):
                ata[j][k] += row[j] * row[k]
            atb[j] += row[j] * y_values[i]

    # Gauss elimination with partial pivoting
    aug = [ata[i][:] + [atb[i]] for i in range(ncoef)]
    for col in range(ncoef):
        max_row = col
        for row in range(col + 1, ncoef):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        for row in range(col + 1, ncoef):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, ncoef + 1):
                aug[row][j] -= factor * aug[col][j]

    coeffs = [0.0] * ncoef
    for i in range(ncoef - 1, -1, -1):
        coeffs[i] = aug[i][ncoef]
        for j in range(i + 1, ncoef):
            coeffs[i] -= aug[i][j] * coeffs[j]
        coeffs[i] /= aug[i][i]

    return coeffs


def _reference_hour(jde):
    """Return (t0, jde_t0): the whole TT hour nearest to jde, as an hour of
    the day and as a JDE."""
    jde_t0 = round((jde - 0.5) * DAY2HOURS) / DAY2HOURS + 0.5
    t0 = int(round((jde_t0 - 0.5) % 1.0 * DAY2HOURS)) % 24
    return t0, jde_t0


def _poly(coeffs, t):
    """Evaluate polynomial coeffs[0] + coeffs[1]*t + ... at t."""
    result = 0.0
    power = 1.0
    for c in coeffs:
        result += c * power
        power *= t
    return result


def _derivative(coeffs):
    """Return the coefficients of the derivative polynomial."""
    return [i * coeffs[i] for i in range(1, len(coeffs))]


def _closest_approach(x, y):
    """Return t (hours from t0) where x(t)^2 + y(t)^2 is minimized.

    Uses Newton's method on d/dt(x^2 + y^2) = 0, starting at t0.
    """
    dx = _derivative(x)
    dy = _derivative(y)
    ddx = _derivative(dx)
    ddy = _derivative(dy)
    t = 0.0
    for _ in range(20):
        xv = _poly(x, t)
        yv = _poly(y, t)
        dxv = _poly(dx, t)
        dyv = _poly(dy, t)

        f = xv * dxv + yv * dyv
        fp = (dxv * dxv + dyv * dyv
              + xv * _poly(ddx, t) + yv * _poly(ddy, t))
        dt = -f / fp
        t += dt
        if abs(dt) < 1e-10:
            break
    else:
        raise RuntimeError("Search for greatest eclipse did not converge")
    if fp <= 0.0 or not _HOURS[0] < t < _HOURS[-1]:
        raise RuntimeError("No closest approach inside the fitted window")
    return t


def _bisect(f, inside, outside):
    """Return the root of f between 'inside' (f < 0) and 'outside' (f > 0)."""
    if f(outside) <= 0.0:
        raise RuntimeError("Contact lies outside the fitted window")
    for _ in range(50):
        mid = 0.5 * (inside + outside)
        if f(mid) < 0.0:
            inside = mid
        else:
            outside = mid
    return 0.5 * (inside + outside)


def _fit_around(instant, jde, refit=True):
    """Fit cubics in hours from t0 to the elements returned by
    instant(epoch), sampled at t0 + _HOURS, where t0 is the whole TT hour
    nearest to greatest eclipse.

    jde is an estimate of greatest eclipse.  Returns (t0, jde_t0, t_max,
    fits): t_max is greatest eclipse in hours from t0, and fits maps each
    element to its coefficients [c0, c1, c2, c3].
    """
    t0, jde_t0 = _reference_hour(jde)
    samples = [instant(Epoch(jde_t0 + t / DAY2HOURS)) for t in _HOURS]
    fits = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if key == "mu":
            _eliminate_angle_wrap(values)
        fits[key] = _fit_poly3(_HOURS, values)
    t_max = _closest_approach(fits["x"], fits["y"])
    # Meeus ch. 54 estimates greatest eclipse to within minutes.  Refit once
    # when the computed maximum lies nearer to another whole hour.
    jde_max = jde_t0 + t_max / DAY2HOURS
    if refit and _reference_hour(jde_max)[1] != jde_t0:
        return _fit_around(instant, jde_max, refit=False)
    return t0, jde_t0, t_max, fits


class BesselianElements(object):
    """
    Class BesselianElements computes Besselian elements for solar eclipses
    from first principles, using VSOP87 for the Sun and the ELP-based lunar
    theory from Meeus ch. 47 for the Moon.

    The method follows Bessel's 1824 approach as described in the Explanatory
    Supplement to the Astronomical Almanac and used by Espenak & Meeus in the
    Five Millennium Canon of Solar Eclipses (NASA TP-2006-214141).

    Given an epoch near a new moon, the class computes the 8 Besselian
    elements (x, y, d, mu, l1, l2, tan f1, tan f2) as 3rd-degree polynomial
    coefficients in hourly time from t0.

    >>> epoch = Epoch(1993, 5, 21.0)
    >>> be = BesselianElements(epoch)
    >>> y, m, d, h, mi, s = be.t_max_epoch.get_full_date()
    >>> print("{}/{}/{} {}:{}".format(y, m, d, h, mi))
    1993/5/21 14:20
    >>> print(round(be.gamma, 3))
    1.137
    >>> print(be.eclipse_type)
    Partial
    >>> epoch2 = Epoch(2009, 7, 22.0)
    >>> be2 = BesselianElements(epoch2)
    >>> print(be2.eclipse_type)
    Total
    """

    def __init__(self, epoch):
        """Compute Besselian elements for the new moon nearest to epoch.

        :param epoch: Approximate epoch near the desired new moon.
        :type epoch: :py:class:`Epoch`
        :raises: TypeError if input value is of wrong type.

        The constructor computes and stores:

        - ``x``, ``y``, ``d``, ``mu``, ``l1``, ``l2``: lists of 4 polynomial
          coefficients [c0, c1, c2, c3] each.  ``mu`` is the ephemeris hour
          angle (sidereal time at TT), as in published elements; the true
          Greenwich hour angle is ``mu - 0.00417807 * DeltaT`` degrees.
        - ``tan_f1``, ``tan_f2``: scalar values at t0.
        - ``t0``: the reference hour (integer hour nearest to maximum).
        - ``jde_t0``: JDE of t0.
        - ``t_max``: fractional hour offset from t0 of greatest eclipse.
        - ``t_max_epoch``: Epoch of greatest eclipse.
        - ``gamma``: minimum distance of shadow axis from Earth center.
        - ``eclipse_type``: one of "Total", "Annular", "Hybrid", "Partial",
          or "No eclipse".
        - ``magnitude``: eclipse magnitude for partial eclipses, None for
          every other type, as in :py:meth:`Eclipse.solar_eclipse`.
        """
        if not isinstance(epoch, Epoch):
            raise TypeError("Invalid input type")

        self._fit(_elements_at_instant, Eclipse.solar_eclipse(epoch)[0].jde())

    @classmethod
    def from_places(cls, places, jde):
        """Compute Besselian elements from another ephemeris.

        :param places: Function of a TT :py:class:`Epoch` returning
            ``(sun, moon, theta)``: the Sun's and the Moon's apparent
            ``(ra, dec, distance)`` of date in radians and Earth radii, and
            the Greenwich apparent sidereal time in radians.
        :type places: callable
        :param jde: TT JDE within a few hours of greatest eclipse, such as
            the new moon, in place of the Meeus ch. 54 estimate.
        :type jde: float
        :returns: The elements, with the attributes of the constructor.
        :rtype: :py:class:`BesselianElements`
        """
        elements = cls.__new__(cls)
        elements._fit(lambda epoch: _elements_from_places(*places(epoch)), jde)
        return elements

    def _fit(self, instant, jde):
        """Fit the elements returned by instant(epoch) around jde."""
        self.t0, self.jde_t0, self.t_max, fits = _fit_around(instant, jde)
        self.t_max_epoch = Epoch(self.jde_t0 + self.t_max / DAY2HOURS)
        self.x, self.y, self.d = fits["x"], fits["y"], fits["d"]
        self.l1, self.l2, self.mu = fits["l1"], fits["l2"], fits["mu"]
        # mu is unwrapped from the t0-4h sample, which can put mu(t0) above
        # 360 degrees
        self.mu[0] %= 360.0
        self.tan_f1 = fits["tanf1"][0]
        self.tan_f2 = fits["tanf2"][0]

        xv = _poly(self.x, self.t_max)
        yv = _poly(self.y, self.t_max)
        self.gamma = copysign(hypot(xv, yv), yv)
        self.eclipse_type, self.magnitude = self._classify()

    def _classify(self):
        """Return (eclipse_type, magnitude) at greatest eclipse.

        Follows the decision tree of Meeus ch. 54, using the computed l1, l2
        and tan f2 where Meeus uses constant approximations of them.  0.9972
        is Meeus's radius of the flattened Earth on the fundamental plane.
        """
        l1 = _poly(self.l1, self.t_max)
        u = _poly(self.l2, self.t_max)
        gamma_abs = abs(self.gamma)

        if gamma_abs > 0.9972 + l1:
            return "No eclipse", None
        if gamma_abs < 0.9972:
            if u < 0.0:
                return "Total", None
            # The umbral radius at height zeta above the fundamental plane is
            # u - zeta * tan f2.  The surface point of greatest eclipse lies
            # at zeta = sqrt(1 - gamma^2), the ends of the path at zeta = 0.
            if u < self.tan_f2 * sqrt(1.0 - gamma_abs * gamma_abs):
                return "Hybrid", None
            return "Annular", None
        if gamma_abs < 0.9972 + abs(u):
            return ("Total" if u < 0.0 else "Annular"), None
        return "Partial", (0.9972 + l1 - gamma_abs) / (l1 + u)

    def get_elements_at(self, t):
        """Evaluate all Besselian elements at hour offset t from t0.

        :param t: Hours from t0.
        :type t: float
        :returns: dict with x, y, d, mu, l1, l2.
        :rtype: dict
        """
        return {key: _poly(getattr(self, key), t)
                for key in ("x", "y", "d", "mu", "l1", "l2")}


class LunarBesselianElements(object):
    """
    Class LunarBesselianElements computes Besselian elements for lunar
    eclipses from the same Sun and Moon theories as
    :py:class:`BesselianElements`.

    The elements are the Moon's position relative to the axis of the Earth's
    shadow (x, y) and the angular radii of the penumbra (f1), the umbra (f2)
    and the Moon (f3), all in degrees as seen from the Earth's centre, as
    3rd-degree polynomial coefficients in hourly time from t0.  The shadow
    radii follow Danjon's rule, as in the Five Millennium Canon of Lunar
    Eclipses (NASA TP-2009-214172).

    >>> le = LunarBesselianElements(Epoch(2025, 3, 14.0))
    >>> y, m, d, h, mi, s = le.t_max_epoch.get_full_date()
    >>> print("{}/{}/{} {}:{}".format(y, m, d, h, mi))
    2025/3/14 6:59
    >>> print(le.eclipse_type)
    Total
    >>> print(round(le.gamma, 3), round(le.umbral_magnitude, 3))
    0.349 1.178
    """

    def __init__(self, epoch):
        """Compute Besselian elements for the lunar eclipse nearest to epoch.

        :param epoch: Approximate epoch near the desired full moon.
        :type epoch: :py:class:`Epoch`
        :raises: TypeError if input value is of wrong type.

        The constructor computes and stores:

        - ``x``, ``y``, ``f1``, ``f2``, ``f3``: lists of 4 polynomial
          coefficients [c0, c1, c2, c3] each, in degrees.  ``hypot(x, y)``
          is the angular distance of the Moon's centre from the shadow axis.
        - ``t0``: the reference hour (integer hour nearest to maximum).
        - ``jde_t0``: JDE of t0.
        - ``t_max``: fractional hour offset from t0 of greatest eclipse.
        - ``t_max_epoch``: Epoch of greatest eclipse.
        - ``gamma``: minimum distance of the Moon's centre from the shadow
          axis, in Earth radii, negative when the Moon passes south of it.
        - ``penumbral_magnitude``, ``umbral_magnitude``: fraction of the
          Moon's diameter immersed in each shadow at greatest eclipse.
        - ``eclipse_type``: one of "Total", "Partial", "Penumbral" or
          "No eclipse".
        - ``contacts``: dict with keys "P1", "U1", "U2", "U3", "U4", "P4"
          mapping to the Epoch of each contact, or None for phases that do
          not occur.
        """
        if not isinstance(epoch, Epoch):
            raise TypeError("Invalid input type")

        self._fit(_lunar_elements_at_instant,
                  Eclipse.lunar_eclipse(epoch)[0].jde())

    @classmethod
    def from_places(cls, places, jde):
        """Compute lunar eclipse elements from another ephemeris.

        :param places: As in :py:meth:`BesselianElements.from_places`; the
            sidereal time is not used.
        :type places: callable
        :param jde: TT JDE within a few hours of greatest eclipse, such as
            the full moon, in place of the Meeus ch. 54 estimate.
        :type jde: float
        :returns: The elements, with the attributes of the constructor.
        :rtype: :py:class:`LunarBesselianElements`
        """
        elements = cls.__new__(cls)
        elements._fit(
            lambda epoch: _lunar_elements_from_places(*places(epoch)[:2]),
            jde)
        return elements

    def _fit(self, instant, jde):
        """Fit the elements returned by instant(epoch) around jde."""
        self.t0, self.jde_t0, self.t_max, fits = _fit_around(instant, jde)
        self.t_max_epoch = Epoch(self.jde_t0 + self.t_max / DAY2HOURS)
        self.x, self.y = fits["x"], fits["y"]
        self.f1, self.f2, self.f3 = fits["f1"], fits["f2"], fits["f3"]

        at_max = self.get_elements_at(self.t_max)
        delta = hypot(at_max["x"], at_max["y"])
        f1, f2, f3 = at_max["f1"], at_max["f2"], at_max["f3"]
        # The Moon's distance in Earth radii is MOON_K / sin(f3)
        self.gamma = copysign(
            MOON_K * sin(radians(delta)) / sin(radians(f3)), at_max["y"])
        self.penumbral_magnitude = (f1 + f3 - delta) / (2 * f3)
        self.umbral_magnitude = (f2 + f3 - delta) / (2 * f3)

        if self.penumbral_magnitude < 0.0:
            self.eclipse_type = "No eclipse"
        elif self.umbral_magnitude <= 0.0:
            self.eclipse_type = "Penumbral"
        elif self.umbral_magnitude < 1.0:
            self.eclipse_type = "Partial"
        else:
            self.eclipse_type = "Total"

        self.contacts = {}
        for first, last, radius in (
            ("P1", "P4", lambda e: e["f1"] + e["f3"]),
            ("U1", "U4", lambda e: e["f2"] + e["f3"]),
            ("U2", "U3", lambda e: e["f2"] - e["f3"]),
        ):
            self.contacts[first], self.contacts[last] = self._contacts(radius)

    def _contacts(self, radius):
        """Return the Epochs before and after greatest eclipse at which the
        Moon's centre is radius(elements) degrees from the shadow axis, or
        (None, None) if it never gets that close."""

        def gap(t):
            e = self.get_elements_at(t)
            return hypot(e["x"], e["y"]) - radius(e)

        if gap(self.t_max) >= 0.0:
            return None, None
        return tuple(
            Epoch(self.jde_t0 + _bisect(gap, self.t_max, edge) / DAY2HOURS)
            for edge in (_HOURS[0], _HOURS[-1])
        )

    def get_elements_at(self, t):
        """Evaluate all lunar eclipse elements at hour offset t from t0.

        :param t: Hours from t0.
        :type t: float
        :returns: dict with x, y, f1, f2, f3 (degrees).
        :rtype: dict
        """
        return {key: _poly(getattr(self, key), t)
                for key in ("x", "y", "f1", "f2", "f3")}


def main():
    """Entry point for the BesselianElements module."""

    def print_me(msg, val):
        print("{}: {}".format(msg, val))

    print("\n" + 50 * "*")
    print("*** Besselian Elements from VSOP87 + ELP")
    print(50 * "*" + "\n")

    # 1993 May 21 partial solar eclipse (Meeus ch. 54 example)
    epoch = Epoch(1993, 5, 21.0)
    be = BesselianElements(epoch)

    y, m, d, h, mi, s = be.t_max_epoch.get_full_date()
    print("Solar eclipse near 1993/5/21:")
    print(
        "Greatest eclipse (TD): {}/{}/{} {}:{}:{}".format(
            y, m, d, h, mi, round(s)
        )
    )
    print_me("gamma", round(be.gamma, 4))
    print_me("t0", be.t0)
    print_me("t_max (hours from t0)", round(be.t_max, 6))
    print_me("tan f1", round(be.tan_f1, 7))
    print_me("tan f2", round(be.tan_f2, 7))

    print("\nPolynomial coefficients:")
    for name in ["x", "y", "d", "l1", "l2", "mu"]:
        coeffs = getattr(be, name)
        formatted = [round(c, 7) for c in coeffs]
        print_me("  " + name, formatted)

    print("\n--- Comparison ---")
    print("Meeus formula 54.1:  14h 21m 00s TD, gamma=1.1348")
    print("Meeus accurate:      14h 20m 14s TD, gamma=1.1370")
    print("Espenak canon:       14h 19m 41s TD")
    print(
        "This (VSOP87+ELP):   {}h {}m {}s TD, gamma={:.4f}".format(
            h, mi, round(s), be.gamma
        )
    )

    print("\n" + 50 * "-")

    # 2024 April 8 total solar eclipse
    epoch2 = Epoch(2024, 4, 8.0)
    be2 = BesselianElements(epoch2)

    y2, m2, d2, h2, mi2, s2 = be2.t_max_epoch.get_full_date()
    print("\nSolar eclipse near 2024/4/8:")
    print(
        "Greatest eclipse (TD): {}/{}/{} {}:{}:{}".format(
            y2, m2, d2, h2, mi2, round(s2)
        )
    )
    print_me("gamma", round(be2.gamma, 4))
    print("Espenak canon: 18:18 TD, gamma=0.3431")


if __name__ == "__main__":

    main()
