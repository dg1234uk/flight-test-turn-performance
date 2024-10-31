import numpy as np


class Atmosphere:
    def __init__(self):
        # US units constants
        self.P_0 = 2116.22  # [lbs/ft^2]
        self.rho_0 = 0.002377  # [slug/ft^3]
        self.a_0 = 661  # [kts]
        self.T_0 = 288.15  # [K]

        self.k1 = 6.87559e-6  # L/T0
        self.k2 = 5.2559  # g0/RL
        self.k3 = 4.80614e-5  # g0/RTa

        self.height_divide = 36089  # [ft]

        self.kt_fps = 1.68781

    def get_std_delta(self, h):
        if h < self.height_divide:
            return (1 - self.k1 * h) ** self.k2
        else:
            return 0.223358 * np.exp(-self.k3 * (h - self.height_divide))

    def get_std_sigma(self, h):
        if h < self.height_divide:
            return (1 - self.k1 * h) ** (self.k2 - 1)
        else:
            return 0.29707 * np.exp(-self.k3 * (h - self.height_divide))

    def get_std_theta(self, h):
        if h < self.height_divide:
            return 1 - self.k1 * h
        else:
            return 0.7519

    def kts_to_fps(self, kts):
        return kts * self.kt_fps

    def fps_to_kts(self, fps):
        return fps / self.kt_fps

    def calc_qc(self, vc):
        return self.P_0 * ((self.rho_0 / self.P_0 * vc**2 / 7 + 1) ** (7 / 2) - 1)

    def calc_Pa(self, h):
        return self.P_0 * self.get_std_delta(h)

    def calc_ve(self, Pa, qc):
        return np.sqrt(7 * Pa / self.rho_0 * ((qc / Pa + 1) ** (2 / 7) - 1))

    def calc_vt(self, ve, h):
        return ve / np.sqrt(self.get_std_sigma(h))

    def calc_mach(self, vt, h):
        return vt / (self.a_0 * np.sqrt(self.get_std_theta(h)))

    def calc_vt_from_mach(self, mach, h):
        return mach * self.a_0 * np.sqrt(self.get_std_theta(h))
