
# -*- coding: utf-8 -*-


# PyMeeus: Python module implementing astronomical algorithms.
# Copyright (C) 2026 Bünyamin Olgun (Mondsichtung.de / Moonstalkers.com)
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


from math import sin, cos, sqrt

from pymeeus.Angle import Angle
from pymeeus.Epoch import Epoch


"""
.. module:: Eclipse
   :synopsis: Class to compute solar and lunar eclipse circumstances
   :license: GNU Lesser General Public License v3 (LGPLv3)

.. moduleauthor:: Bünyamin Olgun (Mondsichtung.de / Moonstalkers.com)
"""


class Eclipse(object):
    """
    Class Eclipse computes the approximate geocentric circumstances of solar
    and lunar eclipses using the approximate (truncated-series) method of
    chapter 54 of Meeus' book: the instant of greatest (geocentric) eclipse,
    the eclipse type, gamma, u and the magnitudes.

    It does NOT compute local circumstances (observer-specific contact times,
    altitudes or obscuration), nor greatest-eclipse coordinates or the central
    path.  Accuracy is limited: benchmarked over years 0-3000 against JPL
    DE441, the time of maximum eclipse has a mean error near 19 s in the
    modern era (1900-2100; up to ~70 s), degrading for ancient dates (mean
    ~55 s before year 500).  For higher accuracy, use the BesselianElements
    class.
    """

    def __init__(self):
        """Eclipse constructor.

        :returns: Eclipse object.
        :rtype: :py:class:`Eclipse`
        """

    @staticmethod
    def _mean_phase_jde(k):
        """Return the JDE of the mean lunar phase for lunation number 'k'.

        An integer 'k' yields a mean new moon, while an integer plus 0.5
        yields a mean full moon (Meeus, chapter 49).
        """
        t = k / 1236.85
        return (2451550.09766 + 29.530588861 * k
                + (0.00015437 + (-0.00000015 + 0.00000000073 * t) * t) * t * t)

    @staticmethod
    def _phase_elements(k):
        """Compute the mean phase time and the orbital arguments shared by the
        solar and lunar eclipse series for lunation number 'k'.

        :returns: ``(jde, E, Mr, Mprimer, Fr, F1r, A1r, Omegar)`` where ``jde``
            is the mean-phase JDE and the remaining values are the relevant
            angles expressed in radians.
        """
        t = k / 1236.85
        jde = Eclipse._mean_phase_jde(k)
        # Eccentricity of Earth's orbit around the Sun
        E = 1.0 + (-0.002516 - 0.0000074 * t) * t
        # Sun's mean anomaly
        M = 2.5534 + 29.1053567 * k + (-0.0000014 - 0.00000011 * t) * t * t
        # Moon's mean anomaly
        Mprime = (201.5643 + 385.81693528 * k
                  + (0.0107582 + (0.00001238 - 0.000000058 * t) * t) * t * t)
        # Moon's argument of latitude
        F = (160.7108 + 390.67050284 * k
             + (-0.0016118 + (-0.00000227 + 0.000000011 * t) * t) * t * t)
        # Longitude of the ascending node of the lunar orbit
        Omega = (124.7746 - 1.56375588 * k
                 + (0.0020672 + 0.00000215 * t) * t * t)
        M = Angle(Angle.reduce_deg(M)).to_positive()
        Mprime = Angle(Angle.reduce_deg(Mprime)).to_positive()
        F = Angle(Angle.reduce_deg(F)).to_positive()
        Omega = Angle(Angle.reduce_deg(Omega)).to_positive()
        # F1 is F corrected for a small offset depending on Omega
        F1 = F - Angle(0.02665 * sin(Omega.rad()))
        # Auxiliary planetary argument
        A1 = Angle(Angle.reduce_deg(299.77 + 0.107408 * k
                                    - 0.009173 * t * t)).to_positive()
        return (jde, E, M.rad(), Mprime.rad(), F.rad(), F1.rad(), A1.rad(),
                Omega.rad())

    @staticmethod
    def _phase_correction(c0, c1, E, Mr, Mprimer, F1r, A1r, Omegar):
        """Periodic correction (in days) from the mean phase to the instant of
        maximum eclipse (Meeus, chapter 54).

        ``c0`` and ``c1`` are the two leading coefficients, which are the only
        terms that differ between solar (-0.4075, 0.1721) and lunar (-0.4065,
        0.1727) eclipses.
        """
        return (c0 * sin(Mprimer)
                + c1 * E * sin(Mr)
                + 0.0161 * sin(2.0 * Mprimer)
                - 0.0097 * sin(2.0 * F1r)
                + 0.0073 * E * sin(Mprimer - Mr)
                - 0.0050 * E * sin(Mprimer + Mr)
                - 0.0023 * sin(Mprimer - 2.0 * F1r)
                + 0.0021 * E * sin(2.0 * Mr)
                + 0.0012 * sin(Mprimer + 2.0 * F1r)
                + 0.0006 * E * sin(2.0 * Mprimer + Mr)
                - 0.0004 * sin(3.0 * Mprimer)
                - 0.0003 * E * sin(Mr + 2.0 * F1r)
                + 0.0003 * sin(A1r)
                - 0.0002 * E * sin(Mr - 2.0 * F1r)
                - 0.0002 * E * sin(2.0 * Mprimer - Mr)
                - 0.0002 * sin(Omegar))

    @staticmethod
    def _solar_max_jde(k):
        """Corrected time of maximum solar eclipse for lunation 'k'."""
        jde, E, Mr, Mprimer, Fr, F1r, A1r, Omegar = Eclipse._phase_elements(k)
        return jde + Eclipse._phase_correction(-0.4075, 0.1721, E, Mr, Mprimer,
                                               F1r, A1r, Omegar)

    @staticmethod
    def _lunar_max_jde(k):
        """Corrected time of maximum lunar eclipse for lunation 'k'."""
        jde, E, Mr, Mprimer, Fr, F1r, A1r, Omegar = Eclipse._phase_elements(k)
        return jde + Eclipse._phase_correction(-0.4065, 0.1727, E, Mr, Mprimer,
                                               F1r, A1r, Omegar)

    @staticmethod
    def _closest_k(epoch, k_approx, max_jde):
        """Pick the lunation number closest to 'epoch'.

        The decimal-year expression used to seed 'k_approx' only locates the
        approximate lunation, so its rounding can land on a syzygy that is not
        the one nearest the input epoch.  Evaluate the neighbouring candidates
        and keep the one whose corrected time of maximum eclipse (computed by
        'max_jde') is closest to 'epoch', measured as a Julian Ephemeris Day
        distance.  Comparing corrected (rather than mean) times matters near
        the midpoint between two lunations, where the periodic correction can
        decide which one is actually closer.
        """
        target = float(epoch)
        k = k_approx
        best = None
        for candidate in (k_approx - 1.0, k_approx, k_approx + 1.0):
            dist = abs(max_jde(candidate) - target)
            if best is None or dist < best:
                best = dist
                k = candidate
        return k

    @staticmethod
    def solar_eclipse(epoch):
        """This method computes the approximate geocentric circumstances of
        the solar eclipse that occurs at the new moon closest to the provided
        epoch (if any).  The resulting time of maximum eclipse is expressed in
        the uniform time scale of Dynamical Time (TT).

        .. note:: This is the approximate (truncated-series) method of Meeus'
           chapter 54.  It gives geocentric quantities only (no local
           circumstances or central path), and the time of maximum is accurate
           to roughly tens of seconds in the modern era, degrading for ancient
           dates.  Use the BesselianElements class when higher accuracy is
           required.

        :param epoch: Approximate epoch we want to find the solar eclipse for.
        :type epoch: :py:class:`Epoch`

        :returns: A tuple ``(max_epoch, eclipse_type, magnitude, gamma, u)``.
            ``max_epoch`` is the instant of maximum eclipse.  ``eclipse_type``
            is one of "Total", "Annular", "Hybrid", "Partial" or "No eclipse".
            ``magnitude`` is the eclipse magnitude, defined by Meeus only for
            partial eclipses; it is ``None`` for the central kinds (total,
            annular, hybrid) and for "No eclipse", where no magnitude is
            computed.  ``gamma`` is the least distance from the axis of the
            Moon's shadow to the centre of the Earth, in units of equatorial
            Earth radii; ``u`` is the radius of the Moon's umbral cone in the
            fundamental plane, also in equatorial Earth radii.  ``gamma`` and
            ``u`` are floats whenever the shadow geometry is evaluated, which
            includes the "No eclipse" axis-miss case (the shadow axis passes
            close to but outside the Earth).  They are ``None`` only when the
            syzygy lies too far from a node (``|sin F| > 0.36``) for any
            geometry to be computed.  In short, a field is ``None`` when it is
            not computed, never a placeholder 0.0.
        :rtype: tuple
        :raises: TypeError if input value is of wrong type.

        >>> epoch = Epoch(1993, 5, 21.0)
        >>> max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
        >>> y, m, d, h, mi, s = max_epoch.get_full_date()
        >>> print("{}/{}/{} {}:{}".format(y, m, d, h, mi))
        1993/5/21 14:20
        >>> print(kind)
        Partial
        >>> print(round(mag, 3))
        0.74
        """

        # First check that input value is of correct type
        if not isinstance(epoch, Epoch):
            raise TypeError("Invalid input type")
        # Let's start computing the year with decimals
        y, m, d = epoch.get_date()
        num_days_year = 365.0
        if Epoch.is_leap(y):
            num_days_year = 366.0
        doy = Epoch.get_doy(y, m, d)
        year = y + doy / num_days_year
        # 'k' is an integer for solar eclipses (new moons).  Seed it from the
        # decimal year, then pick the candidate new moon whose corrected time
        # of maximum eclipse is closest to the input epoch
        k_approx = round((year - 2000.0) * 12.3685, 0)
        k = Eclipse._closest_k(epoch, k_approx, Eclipse._solar_max_jde)
        (jde, E, Mr, Mprimer, Fr,
         F1r, A1r, Omegar) = Eclipse._phase_elements(k)
        # Correct the mean phase to the instant of maximum eclipse.  This is
        # applied before the no-eclipse test so 'max_epoch' carries the same
        # (corrected) meaning whichever rejection branch fires.
        jde += Eclipse._phase_correction(-0.4075, 0.1721, E, Mr, Mprimer,
                                         F1r, A1r, Omegar)
        max_epoch = Epoch(jde)
        # No eclipse is possible if |sin F| > 0.36.  The syzygy is too far
        # from a node to compute the shadow geometry, so the magnitude, gamma
        # and u are reported as None (not computed) rather than a fake 0.0.
        if abs(sin(Fr)) > 0.36:
            return (max_epoch, "No eclipse", None, None, None)
        # P, Q and gamma describe the geometry of the shadow axis
        P = (0.2070 * E * sin(Mr)
             + 0.0024 * E * sin(2.0 * Mr)
             - 0.0392 * sin(Mprimer)
             + 0.0116 * sin(2.0 * Mprimer)
             - 0.0073 * E * sin(Mprimer + Mr)
             + 0.0067 * E * sin(Mprimer - Mr)
             + 0.0118 * sin(2.0 * F1r))
        Q = (5.2207
             - 0.0048 * E * cos(Mr)
             + 0.0020 * E * cos(2.0 * Mr)
             - 0.3299 * cos(Mprimer)
             - 0.0060 * E * cos(Mprimer + Mr)
             + 0.0041 * E * cos(Mprimer - Mr))
        W = abs(cos(F1r))
        gamma = (P * cos(F1r) + Q * sin(F1r)) * (1.0 - 0.0048 * W)
        # u is the radius of the Moon's umbral cone (Earth equator radii)
        u = (0.0059
             + 0.0046 * E * cos(Mr)
             - 0.0182 * cos(Mprimer)
             + 0.0004 * cos(2.0 * Mprimer)
             - 0.0005 * cos(Mr + Mprimer))
        gamma_abs = abs(gamma)
        # Axis-miss: the geometry is computed (gamma, u are real), but the
        # shadow axis passes outside the Earth, so no magnitude applies.
        if gamma_abs > 1.5433 + u:
            return (max_epoch, "No eclipse", None, gamma, u)
        # 'magnitude' is defined by Meeus only for partial eclipses; the
        # central kinds below leave it as None (not computed).
        magnitude = None
        if gamma_abs < 0.9972:
            # Central eclipse: total, annular or hybrid
            if u < 0.0:
                kind = "Total"
            elif u > 0.0047:
                kind = "Annular"
            else:
                omega = 0.00464 * sqrt(1.0 - gamma * gamma)
                if u < omega:
                    kind = "Hybrid"
                else:
                    kind = "Annular"
        elif gamma_abs < 0.9972 + abs(u):
            # Non-central total or annular eclipse (only seen at poles)
            if u < 0.0:
                kind = "Total"
            else:
                kind = "Annular"
        else:
            kind = "Partial"
            magnitude = (1.5433 + u - gamma_abs) / (0.5461 + 2.0 * u)
        return (max_epoch, kind, magnitude, gamma, u)

    @staticmethod
    def lunar_eclipse(epoch):
        """This method computes the approximate geocentric circumstances of
        the lunar eclipse that occurs at the full moon closest to the provided
        epoch (if any).  The resulting time of maximum eclipse is expressed in
        the uniform time scale of Dynamical Time (TT).

        .. note:: This is the approximate (truncated-series) method of Meeus'
           chapter 54.  It gives geocentric quantities only (no local
           circumstances), and the time of maximum is accurate to roughly tens
           of seconds in the modern era, degrading for ancient dates.  Use the
           BesselianElements class when higher accuracy is required.

        :param epoch: Approximate epoch we want to find the lunar eclipse for.
        :type epoch: :py:class:`Epoch`

        :returns: A tuple ``(max_epoch, eclipse_type, umbral_magnitude,
            penumbral_magnitude, sd_partial, sd_total, sd_penumbral)``.
            ``max_epoch`` is the instant of maximum eclipse.  ``eclipse_type``
            is one of "Total", "Partial", "Penumbral" or "No eclipse".  The
            two magnitudes refer to the umbral and penumbral cones of the
            Earth's shadow; they are floats whenever the shadow geometry is
            evaluated, which includes the "No eclipse" near-miss case (the
            Moon misses even the penumbra, giving a negative penumbral
            magnitude).  They are ``None`` only when the syzygy lies too far
            from a node (``|sin F| > 0.36``) for any geometry to be computed.
            ``sd_partial``, ``sd_total`` and ``sd_penumbral`` are the
            semi-durations of the partial, total and penumbral phases
            respectively, expressed in minutes.  When the geometry is
            computed, a semi-duration of 0.0 indicates that the corresponding
            phase does not occur; the three are ``None`` together when no
            eclipse geometry is computed at all.
        :rtype: tuple
        :raises: TypeError if input value is of wrong type.

        >>> epoch = Epoch(1973, 6, 15.0)
        >>> result = Eclipse.lunar_eclipse(epoch)
        >>> max_epoch, kind, umag, pmag, sdp, sdt, sdpen = result
        >>> y, m, d, h, mi, s = max_epoch.get_full_date()
        >>> print("{}/{}/{} {}:{}".format(y, m, d, h, mi))
        1973/6/15 20:50
        >>> print(kind)
        Penumbral
        >>> print(round(pmag, 3))
        0.462
        >>> print(round(sdpen, 1))
        101.5
        """

        # First check that input value is of correct type
        if not isinstance(epoch, Epoch):
            raise TypeError("Invalid input type")
        # Let's start computing the year with decimals
        y, m, d = epoch.get_date()
        num_days_year = 365.0
        if Epoch.is_leap(y):
            num_days_year = 366.0
        doy = Epoch.get_doy(y, m, d)
        year = y + doy / num_days_year
        # 'k' is an integer plus 0.5 for lunar eclipses (full moons).  Seed it
        # from the decimal year, then pick the candidate full moon whose
        # corrected time of maximum eclipse is closest to the input epoch
        k_approx = round((year - 2000.0) * 12.3685 - 0.5, 0) + 0.5
        k = Eclipse._closest_k(epoch, k_approx, Eclipse._lunar_max_jde)
        (jde, E, Mr, Mprimer, Fr,
         F1r, A1r, Omegar) = Eclipse._phase_elements(k)
        # Correct the mean phase to the instant of maximum eclipse.  This is
        # applied before the no-eclipse test so 'max_epoch' carries the same
        # (corrected) meaning whichever rejection branch fires.
        jde += Eclipse._phase_correction(-0.4065, 0.1727, E, Mr, Mprimer,
                                         F1r, A1r, Omegar)
        max_epoch = Epoch(jde)
        # No eclipse is possible if |sin F| > 0.36.  The syzygy is too far
        # from a node to compute the shadow geometry, so the magnitudes and
        # semi-durations are reported as None (not computed) rather than 0.0.
        if abs(sin(Fr)) > 0.36:
            return (max_epoch, "No eclipse", None, None, None, None, None)
        # P, Q and gamma describe the geometry of the shadow axis
        P = (0.2070 * E * sin(Mr)
             + 0.0024 * E * sin(2.0 * Mr)
             - 0.0392 * sin(Mprimer)
             + 0.0116 * sin(2.0 * Mprimer)
             - 0.0073 * E * sin(Mprimer + Mr)
             + 0.0067 * E * sin(Mprimer - Mr)
             + 0.0118 * sin(2.0 * F1r))
        Q = (5.2207
             - 0.0048 * E * cos(Mr)
             + 0.0020 * E * cos(2.0 * Mr)
             - 0.3299 * cos(Mprimer)
             - 0.0060 * E * cos(Mprimer + Mr)
             + 0.0041 * E * cos(Mprimer - Mr))
        W = abs(cos(F1r))
        gamma = (P * cos(F1r) + Q * sin(F1r)) * (1.0 - 0.0048 * W)
        # u is the radius of the Moon's umbral cone (Earth equator radii)
        u = (0.0059
             + 0.0046 * E * cos(Mr)
             - 0.0182 * cos(Mprimer)
             + 0.0004 * cos(2.0 * Mprimer)
             - 0.0005 * cos(Mr + Mprimer))
        gamma_abs = abs(gamma)
        # Penumbral and umbral magnitudes
        umag = (1.0128 - u - gamma_abs) / 0.5450
        pmag = (1.5573 + u - gamma_abs) / 0.5450
        # Near-miss: the magnitudes are computed (and negative), but no phase
        # occurs, so the semi-durations are None (not computed), not 0.0.
        if pmag < 0.0:
            return (max_epoch, "No eclipse", umag, pmag, None, None, None)
        if umag <= 0.0:
            kind = "Penumbral"
        elif umag < 1.0:
            kind = "Partial"
        else:
            kind = "Total"
        # Semi-durations of the three phases (in minutes)
        n = 0.5458 + 0.0400 * cos(Mprimer)
        h_pen = 1.5573 + u
        p_umb = 1.0128 - u
        t_tot = 0.4678 - u
        gamma2 = gamma * gamma
        sd_penumbral = 0.0
        sd_partial = 0.0
        sd_total = 0.0
        disc = h_pen * h_pen - gamma2
        if disc > 0.0:
            sd_penumbral = (60.0 / n) * sqrt(disc)
        disc = p_umb * p_umb - gamma2
        if disc > 0.0:
            sd_partial = (60.0 / n) * sqrt(disc)
        disc = t_tot * t_tot - gamma2
        if disc > 0.0:
            sd_total = (60.0 / n) * sqrt(disc)
        return (max_epoch, kind, umag, pmag,
                sd_partial, sd_total, sd_penumbral)


def main():

    # Let's define a small helper function
    def print_me(msg, val):
        print("{}: {}".format(msg, val))

    # Let's show some uses of the Eclipse class

    print("\n" + 35 * "*")
    print("*** Use of Eclipse class")
    print(35 * "*" + "\n")

    # Compute the circumstances of the solar eclipse of 1993 May 21
    epoch = Epoch(1993, 5, 21.0)
    max_epoch, kind, mag, gamma, u = Eclipse.solar_eclipse(epoch)
    y, m, d, h, mi, s = max_epoch.get_full_date()
    print("Solar eclipse near 1993/5/21:")
    print("Instant of maximum: {}/{}/{} {}:{}:{}".format(y, m, d, h, mi,
                                                         round(s)))
    print_me("Eclipse type", kind)
    print_me("Magnitude", round(mag, 3))
    print_me("gamma", round(gamma, 4))
    print_me("u", round(u, 4))

    print("")

    # Compute the circumstances of the lunar eclipse of 1973 June
    epoch = Epoch(1973, 6, 15.0)
    result = Eclipse.lunar_eclipse(epoch)
    max_epoch, kind, umag, pmag, sd_par, sd_tot, sd_pen = result
    y, m, d, h, mi, s = max_epoch.get_full_date()
    print("Lunar eclipse near 1973/6/15:")
    print("Instant of maximum: {}/{}/{} {}:{}:{}".format(y, m, d, h, mi,
                                                         round(s)))
    print_me("Eclipse type", kind)
    print_me("Umbral magnitude", round(umag, 3))
    print_me("Penumbral magnitude", round(pmag, 3))
    print_me("Partial-phase semi-duration (min)", round(sd_par, 1))
    print_me("Total-phase semi-duration (min)", round(sd_tot, 1))
    print_me("Penumbral-phase semi-duration (min)", round(sd_pen, 1))


if __name__ == "__main__":

    main()
