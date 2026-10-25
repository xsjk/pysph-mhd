from typing import override

from compyle.types import annotate, declare
from pysph.sph.integrator_step import IntegratorStep

from .constants import ARGON_MASS, H_MASS, K_B
from .tabulated_eos import species_state


def initialize_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e, d_epred, d_e0, d_h0, d_converged, d_omega, d_alpha1, d_alpha10, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0):  # ruff: ignore[invalid-argument-name] -- Match PySPH particle property names.
    d_x0[d_idx] = d_x[d_idx]
    d_y0[d_idx] = d_y[d_idx]
    d_z0[d_idx] = d_z[d_idx]
    d_u0[d_idx] = d_u[d_idx]
    d_v0[d_idx] = d_v[d_idx]
    d_w0[d_idx] = d_w[d_idx]
    d_upred[d_idx] = d_u[d_idx]
    d_vpred[d_idx] = d_v[d_idx]
    d_wpred[d_idx] = d_w[d_idx]
    d_e0[d_idx] = d_e[d_idx]
    d_epred[d_idx] = d_e[d_idx]
    d_h0[d_idx] = d_h[d_idx]
    d_converged[d_idx] = 0
    d_omega[d_idx] = 1.0
    d_alpha10[d_idx] = d_alpha1[d_idx]
    d_Bevolx0[d_idx] = d_Bevolx[d_idx]
    d_Bevoly0[d_idx] = d_Bevoly[d_idx]
    d_Bevolz0[d_idx] = d_Bevolz[d_idx]
    d_psi0[d_idx] = d_psi[d_idx]
    d_Bevolxpred[d_idx] = d_Bevolx[d_idx]
    d_Bevolypred[d_idx] = d_Bevoly[d_idx]
    d_Bevolzpred[d_idx] = d_Bevolz[d_idx]
    d_psipred[d_idx] = d_psi[d_idx]


@annotate(d_itype="gintp", dt="double")
def predict_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e0, d_e, d_epred, d_au, d_av, d_aw, d_ae, d_h, d_h0, d_ah, d_cs, d_alpha1, d_alpha10, d_alphaloc, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt):  # ruff: ignore[invalid-argument-name] -- Match PySPH particle property names.
    if d_itype[d_idx] == 1:
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]
        d_upred[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_vpred[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_wpred[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]
        d_epred[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]
        d_h[d_idx] = d_h0[d_idx] + dt * d_ah[d_idx]
        decay_rate = 0.1 * d_cs[d_idx] / d_h[d_idx]
        if d_alpha10[d_idx] < d_alphaloc[d_idx]:
            d_alpha1[d_idx] = d_alphaloc[d_idx]
        else:
            d_alpha1[d_idx] = (d_alpha10[d_idx] + dt * d_alphaloc[d_idx] * decay_rate) / (1.0 + dt * decay_rate)
        d_Bevolx[d_idx] = d_Bevolx0[d_idx] + dtb2 * d_aBevolx[d_idx]
        d_Bevoly[d_idx] = d_Bevoly0[d_idx] + dtb2 * d_aBevoly[d_idx]
        d_Bevolz[d_idx] = d_Bevolz0[d_idx] + dtb2 * d_aBevolz[d_idx]
        d_psi[d_idx] = d_psi0[d_idx] + dtb2 * d_apsi[d_idx]
        d_Bevolxpred[d_idx] = d_Bevolx0[d_idx] + dt * d_aBevolx[d_idx]
        d_Bevolypred[d_idx] = d_Bevoly0[d_idx] + dt * d_aBevoly[d_idx]
        d_Bevolzpred[d_idx] = d_Bevolz0[d_idx] + dt * d_aBevolz[d_idx]
        d_psipred[d_idx] = d_psi0[d_idx] + dt * d_apsi[d_idx]


@annotate(dt="double")
def correct_mhd(d_idx, d_u, d_v, d_w, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_Bx, d_By, d_Bz, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt):  # ruff: ignore[invalid-argument-name] -- Match PySPH particle property names.
    dtb2 = 0.5 * dt
    d_u[d_idx] += dtb2 * d_au[d_idx]
    d_v[d_idx] += dtb2 * d_av[d_idx]
    d_w[d_idx] += dtb2 * d_aw[d_idx]
    d_e[d_idx] += dtb2 * d_ae[d_idx]
    d_Bevolx[d_idx] += dtb2 * d_aBevolx[d_idx]
    d_Bevoly[d_idx] += dtb2 * d_aBevoly[d_idx]
    d_Bevolz[d_idx] += dtb2 * d_aBevolz[d_idx]
    d_psi[d_idx] += dtb2 * d_apsi[d_idx]
    d_Bx[d_idx] = d_Bevolx[d_idx] * d_rho[d_idx]
    d_By[d_idx] = d_Bevoly[d_idx] * d_rho[d_idx]
    d_Bz[d_idx] = d_Bevolz[d_idx] * d_rho[d_idx]


MHD_HELPERS = [initialize_mhd, predict_mhd, correct_mhd]


class MHDStep(IntegratorStep):
    def __init__(self, gamma):
        assert gamma > 1.0
        self.gamma1 = gamma - 1.0

    def _get_helpers_(self):  # ruff: ignore[bad-dunder-method-name] -- PySPH's helper registration hook.
        return MHD_HELPERS

    @override
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e, d_epred, d_e0, d_h0, d_converged, d_omega, d_alpha1, d_alpha10, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0):
        initialize_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e, d_epred, d_e0, d_h0, d_converged, d_omega, d_alpha1, d_alpha10, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0)

    @override
    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e0, d_e, d_epred, d_au, d_av, d_aw, d_ae, d_h, d_h0, d_ah, d_cs, d_alpha1, d_alpha10, d_alphaloc, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt):
        predict_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e0, d_e, d_epred, d_au, d_av, d_aw, d_ae, d_h, d_h0, d_ah, d_cs, d_alpha1, d_alpha10, d_alphaloc, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt)

    @override
    def stage2(self, d_idx, d_u, d_v, d_w, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_p, d_Bx, d_By, d_Bz, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt):
        if d_itype[d_idx] == 1:
            correct_mhd(d_idx, d_u, d_v, d_w, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_Bx, d_By, d_Bz, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt)
            d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_e[d_idx]


class PlasmaMHDStep(IntegratorStep):
    def __init__(self):
        from .plasma_eos import N_COUNT, T_COUNT  # ruff: ignore[import-outside-top-level] -- Only enabled plasma runs need the table.

        self.n_count = N_COUNT
        self.t_count = T_COUNT

    def _get_helpers_(self):  # ruff: ignore[bad-dunder-method-name] -- PySPH's helper registration hook.
        # ArgonHydrogenEOS emits the shared lookup helpers in the same CUDA module.
        return MHD_HELPERS

    @override
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e, d_epred, d_e0, d_h0, d_converged, d_omega, d_alpha1, d_alpha10, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0):
        initialize_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e, d_epred, d_e0, d_h0, d_converged, d_omega, d_alpha1, d_alpha10, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0)

    @override
    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e0, d_e, d_epred, d_au, d_av, d_aw, d_ae, d_h, d_h0, d_ah, d_cs, d_alpha1, d_alpha10, d_alphaloc, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt):
        predict_mhd(d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0, d_w0, d_u, d_v, d_w, d_upred, d_vpred, d_wpred, d_e0, d_e, d_epred, d_au, d_av, d_aw, d_ae, d_h, d_h0, d_ah, d_cs, d_alpha1, d_alpha10, d_alphaloc, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_psipred, d_Bevolx0, d_Bevoly0, d_Bevolz0, d_psi0, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt)

    @override
    def stage2(self, d_idx, d_u, d_v, d_w, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_p, d_cs, d_T, d_mean_charge, d_material_id, d_Bx, d_By, d_Bz, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_itype, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, d_eos_n, d_eos_T, d_eos_e, d_eos_z, d_eos_h_e, d_eos_h_z, dt):
        state = declare("matrix(4)")
        if d_itype[d_idx] == 1:
            correct_mhd(d_idx, d_u, d_v, d_w, d_e, d_au, d_av, d_aw, d_ae, d_rho, d_Bx, d_By, d_Bz, d_Bevolx, d_Bevoly, d_Bevolz, d_psi, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, dt)
        if d_material_id[d_idx] == 0:
            species_state(d_rho[d_idx], d_e[d_idx], d_eos_n, d_eos_T, d_eos_e, d_eos_z, state, self.n_count, self.t_count, ARGON_MASS, K_B)
        elif d_material_id[d_idx] == 2:
            species_state(d_rho[d_idx], d_e[d_idx], d_eos_n, d_eos_T, d_eos_h_e, d_eos_h_z, state, self.n_count, self.t_count, H_MASS, K_B)
        d_T[d_idx] = state[0]
        d_mean_charge[d_idx] = state[1]
        d_p[d_idx] = state[2]
        d_cs[d_idx] = state[3]
