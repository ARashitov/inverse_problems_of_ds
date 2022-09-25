from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field, PrivateAttr


class HeatEquationMeta(BaseModel):

    a: int = Field(default=1, description="Thermal diffusion constant")
    L: int = Field(default=100, description="Length of domain")
    N: int = Field(default=1000, description="Number of discretization points")
    dt: float = Field(default=0.1, description="Discretization time-stamp")
    upper_time_limit: int = Field(default=100, description="Latest time stamp")
    t: Union[List[float], None] = Field(default=None, description="Time frequency domain")
    dx: Union[float, None] = Field(default=None, description="Length of the single discrete section")
    y: Union[List[float], None] = Field(default=None, description="Fourier frequency domain")
    x: Union[List[float], None] = Field(default=None, description="Solution domain")
    kappa: Union[List[float], None] = Field(
        default=None,
        description="Discrete wavepoints or sin component",
    )
    u: Union[List[float], None] = Field(
        default=None,
        description="Temperature state",
    )
    uhat: Union[List[float], None] = Field(
        default=None,
        description="Diffenretial heat state",
    )
    uhat_ri: Union[List[float], None] = Field(
        default=None,
        description="N-element complex vector",
    )
    _is_initialized: bool = PrivateAttr(default=False)

    def init_dx(self):
        self.dx = self.L / self.N

    def init_kappa(self):
        self.kappa = (2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)).tolist()

    def init_domain(self):
        domain_half = self.L / 2
        self.x = np.arange(-domain_half, domain_half, self.dx)

    def init_start_condition(self):
        """
        NOTE: change code here for different temperature start states
        """
        INITIAL_TEMPERATURE = 1
        _u0 = np.zeros_like(self.x)
        left_bound = int((self.L/2 - self.L/10) / self.dx)
        right_bound = int((self.L/2 + self.L/10) / self.dx)
        _u0[left_bound: right_bound] = INITIAL_TEMPERATURE
        self.u = _u0
        self.uhat = np.fft.fft(self.u)
        self.uhat_ri = np.concatenate([self.uhat.real, self.uhat.imag])

    def init_time_frequenc_domain(self):
        self.t = np.arange(0, self.upper_time_limit, self.dt)

    def init_components(self):
        """
        Function performs initialization of heat
        equation metadata instance for further solution
        """
        self.init_dx()
        self.init_kappa()
        self.init_domain()
        self.init_start_condition()
        self.init_time_frequenc_domain()
        self._is_initialized = True

    def check_initialization(self):
        if not self._is_initialized:
            msg = (
                "You must initilize your components first by using "
                "init_components() function"
            )
            raise ValueError(msg)

    def plot_inital_condition(self):
        self.check_initialization()
        df = pd.DataFrame({
            "x": range(len(self.u)),
            "y": self.u,
        })
        fig = px.line(df, x="x", y="y", title='Temperature state')
        fig.show()


class HeatEquationFFTSolver(object):

    @staticmethod
    def _refresh_heat_state(uhat_ri: List[float], meta: HeatEquationMeta):
        return uhat_ri[:meta.N] + (1j) * uhat_ri[meta.N:]

    @staticmethod
    def rhsHeat(uhat_ri: List[float], t: float, meta: HeatEquationMeta) -> np.array:
        uhat = HeatEquationFFTSolver._refresh_heat_state(uhat_ri, meta)
        d_uhat = -(meta.a**2) * (np.power(meta.kappa, 2)) * uhat
        d_uhat_ri = np.concatenate((
            d_uhat.real,
            d_uhat.imag,
        )).astype(np.float64)
        return d_uhat_ri

    def unpack_differentials(self, meta: HeatEquationMeta):
        meta.uhat = (
            meta.uhat_ri[:, :meta.N] +
            (1j) + (meta.uhat_ri[:, meta.N:])
        )
        meta.u = np.zeros_like(meta.uhat)
        return meta

    def run_heat_fourier_transforms(
            self,
            meta: HeatEquationMeta,
    ):
        _u = np.zeros_like(meta.uhat)
        for k in range(len(meta.t)):
            _u[k, :] = np.fft.ifft(meta.uhat[k, :])
        return _u.real
