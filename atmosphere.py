import numpy as np  # type: ignore # pylint: disable=import-error


class Atmosphere:
    def __init__(self):
        """
        Initialize the Atmosphere class with standard atmospheric constants.
        """
        # US units constants
        self.P_0 = 2116.22  # [lbs/ft^2]
        self.rho_0 = 0.002377  # [slug/ft^3]
        self.a_0 = 661  # [kts]
        self.T_0 = 288.15  # [K]

        self.k1 = 6.87559e-6  # L/T0
        self.k2 = 5.2559  # g0/RL
        self.k3 = 4.80614e-5  # g0/RTa

        self.height_divide = 36089  # [ft]

        self.kt_fps = 1.68781  # [ft/s]
        self.celsius_kelvin = 273.15  # [K]

    def get_std_delta(self, h):
        """
        Calculate the standard pressure ratio (delta) at a given altitude.

        Parameters:
        h (float): Altitude in feet.

        Returns:
        float: Standard pressure ratio.
        """
        if h < self.height_divide:
            return (1 - self.k1 * h) ** self.k2
        else:
            return 0.223358 * np.exp(-self.k3 * (h - self.height_divide))

    def get_h_from_delta(self, delta):
        """
        Calculate the altitude from the standard pressure ratio (delta).

        Parameters:
        delta (float): Standard pressure ratio.

        Returns:
        float: Altitude in feet.
        """
        if delta > 0.223358:
            return (1 - delta ** (1 / self.k2)) / self.k1
        else:
            return self.height_divide - np.log(delta / 0.223358) / self.k3

    def get_std_sigma(self, h):
        """
        Calculate the standard density ratio (sigma) at a given altitude.

        Parameters:
        h (float): Altitude in feet.

        Returns:
        float: Standard density ratio.
        """
        if h < self.height_divide:
            return (1 - self.k1 * h) ** (self.k2 - 1)
        else:
            return 0.29707 * np.exp(-self.k3 * (h - self.height_divide))

    def get_std_theta(self, h):
        """
        Calculate the standard temperature ratio (theta) at a given altitude.

        Parameters:
        h (float): Altitude in feet.

        Returns:
        float: Standard temperature ratio.
        """
        if h < self.height_divide:
            return 1 - self.k1 * h
        else:
            return 0.7519

    def calc_sigma(self, h_p, t_a):
        """
        Calculate the density ratio (sigma) at a given altitude.

        Parameters:
        h_p (float): Altitude in feet.
        T_a (float): Ambient temperature in Kelvin.

        Returns:
        float: Density ratio.
        """
        delta = self.get_std_delta(h_p)
        return delta * self.T_0 / t_a

    def kts_to_fps(self, kts):
        """
        Convert speed from knots to feet per second.

        Parameters:
        kts (float): Speed in knots.

        Returns:
        float: Speed in feet per second.
        """
        return kts * self.kt_fps

    def fps_to_kts(self, fps):
        """
        Convert speed from feet per second to knots.

        Parameters:
        fps (float): Speed in feet per second.

        Returns:
        float: Speed in knots.
        """
        return fps / self.kt_fps

    def celsius_to_kelvin(self, celsius):
        """
        Convert temperature from Celsius to Kelvin.

        Parameters:
        celsius (float): Temperature in Celsius.

        Returns:
        float: Temperature in Kelvin.
        """
        return celsius + self.celsius_kelvin

    def calc_qc(self, vc):
        """
        Calculate the impact pressure (qc) from calibrated airspeed (vc).

        Parameters:
        vc (float): Calibrated airspeed in feet per second.

        Returns:
        float: Impact pressure in pounds per square foot.
        """
        return self.P_0 * ((self.rho_0 / self.P_0 * vc**2 / 7 + 1) ** (7 / 2) - 1)

    def calc_Pa(self, h):
        """
        Calculate the ambient pressure (Pa) at a given altitude.

        Parameters:
        h (float): Altitude in feet.

        Returns:
        float: Ambient pressure in pounds per square foot.
        """
        return self.P_0 * self.get_std_delta(h)

    def calc_ve(self, Pa, qc):
        """
        Calculate the equivalent airspeed (ve) from ambient pressure (Pa) and impact pressure (qc).

        Parameters:
        Pa (float): Ambient pressure in pounds per square foot.
        qc (float): Impact pressure in pounds per square foot.

        Returns:
        float: Equivalent airspeed in feet per second.
        """
        return np.sqrt(7 * Pa / self.rho_0 * ((qc / Pa + 1) ** (2 / 7) - 1))

    def calc_vt(self, ve, h):
        """
        Calculate the true airspeed (vt) from equivalent airspeed (ve) and altitude (h).

        Parameters:
        ve (float): Equivalent airspeed in feet per second.
        h (float): Altitude in feet.

        Returns:
        float: True airspeed in feet per second.
        """
        return ve / np.sqrt(self.get_std_sigma(h))

    def calc_mach(self, vt, h):
        """
        Calculate the Mach number from true airspeed (vt) and altitude (h).

        Parameters:
        vt (float): True airspeed in feet per second.
        h (float): Altitude in feet.

        Returns:
        float: Mach number.
        """
        return vt / (self.a_0 * np.sqrt(self.get_std_theta(h)))

    def calc_vt_from_mach(self, mach, h):
        """
        Calculate the true airspeed (vt) from Mach number and altitude (h).

        Parameters:
        mach (float): Mach number.
        h (float): Altitude in feet.

        Returns:
        float: True airspeed in feet per second.
        """
        return mach * self.a_0 * np.sqrt(self.get_std_theta(h))
