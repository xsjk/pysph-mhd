from math import sqrt
from typing import override

from pysph.sph.equation import Equation

MU0 = 1.0
DENSITY_TOLERANCE = 1.0e-4
COURANT_FACTOR = 0.3
FORCE_FACTOR = 0.25
PAIR_DISTANCE_EPSILON = 1.0e-12
INTERACTION_DISTANCE_EPSILON = 1.0e-8
FLOAT_TINY = 2.2250738585072014e-308


class DensityIteration(Equation):
    def __init__(self, dest, sources, *, k, dim, iterate_only_once):
        self.density_iterations = True
        self.iterate_only_once = iterate_only_once
        self.dim = dim
        self.k = k
        self.htol = DENSITY_TOLERANCE
        self.equation_has_converged = 1
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_rho, d_div, d_grhox, d_grhoy, d_grhoz, d_arho, d_dwdh):
        d_rho[d_idx] = 0.0
        d_div[d_idx] = 0.0

        d_grhox[d_idx] = 0.0
        d_grhoy[d_idx] = 0.0
        d_grhoz[d_idx] = 0.0
        d_arho[d_idx] = 0.0

        d_dwdh[d_idx] = 0.0
        self.equation_has_converged = 1

    @override
    def loop(self, d_idx, s_idx, d_rho, d_grhox, d_grhoy, d_grhoz, d_arho, d_dwdh, s_m, d_converged, d_upred, d_vpred, d_wpred, s_upred, s_vpred, s_wpred, WI, DWI, GHI):
        mj = s_m[s_idx]
        dvx = d_upred[d_idx] - s_upred[s_idx]
        dvy = d_vpred[d_idx] - s_vpred[s_idx]
        dvz = d_wpred[d_idx] - s_wpred[s_idx]
        vijdotdwij = dvx * DWI[0] + dvy * DWI[1] + dvz * DWI[2]

        d_rho[d_idx] += mj * WI
        d_arho[d_idx] += mj * vijdotdwij
        d_grhox[d_idx] += mj * DWI[0]
        d_grhoy[d_idx] += mj * DWI[1]
        d_grhoz[d_idx] += mj * DWI[2]
        d_dwdh[d_idx] += mj * GHI

    @override
    def post_loop(self, d_idx, d_arho, d_rho, d_div, d_omega, d_dwdh, d_h0, d_h, d_m, d_ah, d_converged):
        if self.density_iterations and d_converged[d_idx] != 1:
            mi = d_m[d_idx]
            hi = d_h[d_idx]
            hi0 = d_h0[d_idx]
            rhoi = mi / pow(hi / self.k, self.dim)
            dhdrhoi = -hi / (self.dim * rhoi)
            dwdhi = d_dwdh[d_idx]
            omegai = 1.0 - dhdrhoi * dwdhi

            gradhi = 1.0 / omegai
            d_omega[d_idx] = gradhi

            func = rhoi - d_rho[d_idx]
            dfdh1 = dhdrhoi / omegai if omegai > 0.0 else dhdrhoi / abs(omegai + 2.220446049250313e-16)
            hnew = hi - func * dfdh1

            if hnew > 1.2 * hi:
                hnew = 1.2 * hi
            elif hnew < 0.8 * hi:
                hnew = 0.8 * hi

            diff = abs(hnew - hi) / hi0
            finalize_density = ((diff < self.htol) and (omegai > 0.0) and (hi > 0.0)) or self.iterate_only_once

            if not finalize_density:
                self.equation_has_converged = -1
                d_h[d_idx] = hnew
                d_converged[d_idx] = 0
            else:
                d_arho[d_idx] *= d_omega[d_idx]
                d_ah[d_idx] = d_arho[d_idx] * dhdrhoi
                d_h[d_idx] = self.k * pow(mi / d_rho[d_idx], 1.0 / self.dim)
                d_converged[d_idx] = 1

        d_div[d_idx] = -d_arho[d_idx] / d_rho[d_idx]

    @override
    def converged(self):
        if hasattr(self, "_pull"):
            self._pull("equation_has_converged")
        return self.equation_has_converged


class IdealGasEOS(Equation):
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        self.gamma1 = gamma - 1.0
        super().__init__(dest, sources)

    @override
    def loop(self, d_idx, d_p, d_rho, d_epred, d_cs):
        d_p[d_idx] = self.gamma1 * d_rho[d_idx] * d_epred[d_idx]
        d_cs[d_idx] = sqrt(self.gamma * d_p[d_idx] / d_rho[d_idx])


class DensityFromSmoothingLength(Equation):
    def __init__(self, dest, sources, k, dim):
        self.k = k
        self.dim = dim
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_m, d_h, d_rho, d_rho_sum):
        d_rho_sum[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = d_m[d_idx] / pow(d_h[d_idx] / self.k, self.dim)


class SmoothingLengthRateFromDensityRelation(Equation):
    def __init__(self, dest, sources, k, dim):
        self.k = k
        self.dim = dim
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_m, d_h, d_rho, d_ah):
        rhoh = d_m[d_idx] / pow(d_h[d_idx] / self.k, self.dim)
        d_ah[d_idx] *= rhoh / d_rho[d_idx]


class SmoothingLengthRateFromForceDivergence(Equation):
    def __init__(self, dest, sources, dim):
        self.dim1 = 1.0 / dim
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_h, d_div, d_ah):
        d_ah[d_idx] = d_h[d_idx] * d_div[d_idx] * self.dim1


class MagneticStressAccelerations(Equation):
    def __init__(self, dest, sources):
        self.mu0 = MU0
        super().__init__(dest, sources)

    @override
    def loop(self, d_idx, s_idx, d_m, s_m, d_rho, s_rho, d_Bxpred, d_Bypred, d_Bzpred, s_Bxpred, s_Bypred, s_Bzpred, d_omega, s_omega, d_au, d_av, d_aw, XIJ, DWI, DWJ, RIJ):
        if RIJ > PAIR_DISTANCE_EPSILON:
            runix = XIJ[0] / RIJ
            runiy = XIJ[1] / RIJ
            runiz = XIJ[2] / RIJ
        else:
            runix = 0.0
            runiy = 0.0
            runiz = 0.0

        grad_i = d_omega[d_idx] * (runix * DWI[0] + runiy * DWI[1] + runiz * DWI[2])
        grad_j = s_omega[s_idx] * (runix * DWJ[0] + runiy * DWJ[1] + runiz * DWJ[2])
        rho_i = d_rho[d_idx]
        rho_j = s_rho[s_idx]
        brhox_i = d_Bxpred[d_idx] / rho_i
        brhoy_i = d_Bypred[d_idx] / rho_i
        brhoz_i = d_Bzpred[d_idx] / rho_i
        brhox_j = s_Bxpred[s_idx] / rho_j
        brhoy_j = s_Bypred[s_idx] / rho_j
        brhoz_j = s_Bzpred[s_idx] / rho_j
        bro2_i = (brhox_i * brhox_i + brhoy_i * brhoy_i + brhoz_i * brhoz_i) / self.mu0
        bro2_j = (brhox_j * brhox_j + brhoy_j * brhoy_j + brhoz_j * brhoz_j) / self.mu0
        gradp = s_m[s_idx] * (0.5 * bro2_i * grad_i + 0.5 * bro2_j * grad_j)

        sxx_i = -d_m[d_idx] * brhox_i * brhox_i / self.mu0
        sxy_i = -d_m[d_idx] * brhox_i * brhoy_i / self.mu0
        sxz_i = -d_m[d_idx] * brhox_i * brhoz_i / self.mu0
        syy_i = -d_m[d_idx] * brhoy_i * brhoy_i / self.mu0
        syz_i = -d_m[d_idx] * brhoy_i * brhoz_i / self.mu0
        szz_i = -d_m[d_idx] * brhoz_i * brhoz_i / self.mu0
        sxx_j = -s_m[s_idx] * brhox_j * brhox_j / self.mu0
        sxy_j = -s_m[s_idx] * brhox_j * brhoy_j / self.mu0
        sxz_j = -s_m[s_idx] * brhox_j * brhoz_j / self.mu0
        syy_j = -s_m[s_idx] * brhoy_j * brhoy_j / self.mu0
        syz_j = -s_m[s_idx] * brhoy_j * brhoz_j / self.mu0
        szz_j = -s_m[s_idx] * brhoz_j * brhoz_j / self.mu0

        projsx = (sxx_i * runix + sxy_i * runiy + sxz_i * runiz) * grad_i + (sxx_j * runix + sxy_j * runiy + sxz_j * runiz) * grad_j
        projsy = (sxy_i * runix + syy_i * runiy + syz_i * runiz) * grad_i + (sxy_j * runix + syy_j * runiy + syz_j * runiz) * grad_j
        projsz = (sxz_i * runix + syz_i * runiy + szz_i * runiz) * grad_i + (sxz_j * runix + syz_j * runiy + szz_j * runiz) * grad_j
        ax = -runix * gradp - projsx
        ay = -runiy * gradp - projsy
        az = -runiz * gradp - projsz

        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az


class MHDAccelerations(Equation):
    def __init__(self, dest, sources):
        self.beta = 2.0
        self.mu0 = MU0
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_au, d_av, d_aw, d_ae, d_dt_cfl, d_div, d_div_for_psi):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_ae[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0
        d_div[d_idx] = 0.0
        d_div_for_psi[d_idx] = 0.0

    @override
    def loop(self, d_idx, s_idx, d_m, s_m, d_p, s_p, d_cs, s_cs, d_epred, s_epred, d_rho, s_rho, d_Bxpred, d_Bypred, d_Bzpred, s_Bxpred, s_Bypred, s_Bzpred, d_upred, d_vpred, d_wpred, s_upred, s_vpred, s_wpred, d_au, d_av, d_aw, d_ae, d_omega, s_omega, XIJ, DWI, DWJ, d_alpha1, s_alpha1, RIJ, RHOIJ, d_dt_cfl, d_div):
        p_i = d_p[d_idx]
        pj = s_p[s_idx]
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        rhoi2 = rhoi * rhoi
        rhoj2 = rhoj * rhoj
        pibrhoi2 = p_i / rhoi2
        pjbrhoj2 = pj / rhoj2
        mj = s_m[s_idx]
        bi2 = d_Bxpred[d_idx] * d_Bxpred[d_idx] + d_Bypred[d_idx] * d_Bypred[d_idx] + d_Bzpred[d_idx] * d_Bzpred[d_idx]
        bj2 = s_Bxpred[s_idx] * s_Bxpred[s_idx] + s_Bypred[s_idx] * s_Bypred[s_idx] + s_Bzpred[s_idx] * s_Bzpred[s_idx]
        vwave_i = sqrt(d_cs[d_idx] * d_cs[d_idx] + bi2 / (self.mu0 * rhoi))
        vwave_j = sqrt(s_cs[s_idx] * s_cs[s_idx] + bj2 / (self.mu0 * rhoj))

        if RIJ < INTERACTION_DISTANCE_EPSILON:
            runix = 0.0
            runiy = 0.0
            runiz = 0.0
        else:
            runix = XIJ[0] / RIJ
            runiy = XIJ[1] / RIJ
            runiz = XIJ[2] / RIJ

        dvx = d_upred[d_idx] - s_upred[s_idx]
        dvy = d_vpred[d_idx] - s_vpred[s_idx]
        dvz = d_wpred[d_idx] - s_wpred[s_idx]
        dot = dvx * runix + dvy * runiy + dvz * runiz
        omegai = d_omega[d_idx]
        omegaj = s_omega[s_idx]
        grad_i = omegai * (runix * DWI[0] + runiy * DWI[1] + runiz * DWI[2])
        grad_j = omegaj * (runix * DWJ[0] + runiy * DWJ[1] + runiz * DWJ[2])
        d_div[d_idx] += d_m[d_idx] * dot * grad_i
        pdiff = abs(p_i - pj)
        vsig2 = sqrt(pdiff / RHOIJ)
        dt_cfl = vwave_i - self.beta * dot
        dt_cfl = max(dt_cfl, vwave_j - self.beta * dot)
        dt_cfl = max(dt_cfl, 0.0)
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], dt_cfl)

        if dot < 0.0:
            vsigav_i = max(d_alpha1[d_idx] * vwave_i - self.beta * dot, 0.0)
            vsigav_j = max(s_alpha1[s_idx] * vwave_j - self.beta * dot, 0.0)
            qrho2_i = -0.5 / rhoi * vsigav_i * dot
            qrho2_j = -0.5 / rhoj * vsigav_j * dot
            d_au[d_idx] += -mj * (qrho2_i * omegai * DWI[0] + qrho2_j * omegaj * DWJ[0])
            d_av[d_idx] += -mj * (qrho2_i * omegai * DWI[1] + qrho2_j * omegaj * DWJ[1])
            d_aw[d_idx] += -mj * (qrho2_i * omegai * DWI[2] + qrho2_j * omegaj * DWJ[2])
            d_ae[d_idx] += mj * qrho2_i * dot * grad_i

        d_au[d_idx] += -mj * (pibrhoi2 * omegai * DWI[0] + pjbrhoj2 * omegaj * DWJ[0])
        d_av[d_idx] += -mj * (pibrhoi2 * omegai * DWI[1] + pjbrhoj2 * omegaj * DWJ[1])
        d_aw[d_idx] += -mj * (pibrhoi2 * omegai * DWI[2] + pjbrhoj2 * omegaj * DWJ[2])

        vijdotdwi = dvx * DWI[0] + dvy * DWI[1] + dvz * DWI[2]
        d_ae[d_idx] += mj * pibrhoi2 * omegai * vijdotdwi

        eij = d_epred[d_idx] - s_epred[s_idx]
        d_ae[d_idx] += vsig2 * eij * (0.5 * d_m[d_idx] / rhoi * grad_i + 0.5 * mj / rhoj * grad_j)

    @override
    def post_loop(self, d_idx, d_rho, d_div, d_div_for_psi):
        d_div_for_psi[d_idx] = -d_div[d_idx] / d_rho[d_idx]
        d_div[d_idx] = -d_div[d_idx] / d_rho[d_idx]


class CullenDehnenAlphaDiagnostics(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_alphaloc, d_xilimiter, d_divvdt, d_cd_rxx, d_cd_rxy, d_cd_rxz, d_cd_ryy, d_cd_ryz, d_cd_rzz, d_cd_vxx, d_cd_vxy, d_cd_vxz, d_cd_vyx, d_cd_vyy, d_cd_vyz, d_cd_vzx, d_cd_vzy, d_cd_vzz, d_cd_axx, d_cd_axy, d_cd_axz, d_cd_ayx, d_cd_ayy, d_cd_ayz, d_cd_azx, d_cd_azy, d_cd_azz):
        d_alphaloc[d_idx] = 0.0
        d_xilimiter[d_idx] = 1.0
        d_divvdt[d_idx] = 0.0
        d_cd_rxx[d_idx] = 0.0
        d_cd_rxy[d_idx] = 0.0
        d_cd_rxz[d_idx] = 0.0
        d_cd_ryy[d_idx] = 0.0
        d_cd_ryz[d_idx] = 0.0
        d_cd_rzz[d_idx] = 0.0
        d_cd_vxx[d_idx] = 0.0
        d_cd_vxy[d_idx] = 0.0
        d_cd_vxz[d_idx] = 0.0
        d_cd_vyx[d_idx] = 0.0
        d_cd_vyy[d_idx] = 0.0
        d_cd_vyz[d_idx] = 0.0
        d_cd_vzx[d_idx] = 0.0
        d_cd_vzy[d_idx] = 0.0
        d_cd_vzz[d_idx] = 0.0
        d_cd_axx[d_idx] = 0.0
        d_cd_axy[d_idx] = 0.0
        d_cd_axz[d_idx] = 0.0
        d_cd_ayx[d_idx] = 0.0
        d_cd_ayy[d_idx] = 0.0
        d_cd_ayz[d_idx] = 0.0
        d_cd_azx[d_idx] = 0.0
        d_cd_azy[d_idx] = 0.0
        d_cd_azz[d_idx] = 0.0

    @override
    def loop(self, d_idx, s_idx, s_m, d_au, d_av, d_aw, s_au, s_av, s_aw, d_cd_rxx, d_cd_rxy, d_cd_rxz, d_cd_ryy, d_cd_ryz, d_cd_rzz, d_cd_vxx, d_cd_vxy, d_cd_vxz, d_cd_vyx, d_cd_vyy, d_cd_vyz, d_cd_vzx, d_cd_vzy, d_cd_vzz, d_cd_axx, d_cd_axy, d_cd_axz, d_cd_ayx, d_cd_ayy, d_cd_ayz, d_cd_azx, d_cd_azy, d_cd_azz, d_upred, d_vpred, d_wpred, s_upred, s_vpred, s_wpred, XIJ, DWI):
        runx = s_m[s_idx] * DWI[0]
        runy = s_m[s_idx] * DWI[1]
        runz = s_m[s_idx] * DWI[2]
        dvx = d_upred[d_idx] - s_upred[s_idx]
        dvy = d_vpred[d_idx] - s_vpred[s_idx]
        dvz = d_wpred[d_idx] - s_wpred[s_idx]
        dax = d_au[d_idx] - s_au[s_idx]
        day = d_av[d_idx] - s_av[s_idx]
        daz = d_aw[d_idx] - s_aw[s_idx]

        d_cd_rxx[d_idx] += -XIJ[0] * runx
        d_cd_rxy[d_idx] += -XIJ[0] * runy
        d_cd_rxz[d_idx] += -XIJ[0] * runz
        d_cd_ryy[d_idx] += -XIJ[1] * runy
        d_cd_ryz[d_idx] += -XIJ[1] * runz
        d_cd_rzz[d_idx] += -XIJ[2] * runz

        d_cd_vxx[d_idx] += dvx * runx
        d_cd_vxy[d_idx] += dvx * runy
        d_cd_vxz[d_idx] += dvx * runz
        d_cd_vyx[d_idx] += dvy * runx
        d_cd_vyy[d_idx] += dvy * runy
        d_cd_vyz[d_idx] += dvy * runz
        d_cd_vzx[d_idx] += dvz * runx
        d_cd_vzy[d_idx] += dvz * runy
        d_cd_vzz[d_idx] += dvz * runz

        d_cd_axx[d_idx] += dax * runx
        d_cd_axy[d_idx] += dax * runy
        d_cd_axz[d_idx] += dax * runz
        d_cd_ayx[d_idx] += day * runx
        d_cd_ayy[d_idx] += day * runy
        d_cd_ayz[d_idx] += day * runz
        d_cd_azx[d_idx] += daz * runx
        d_cd_azy[d_idx] += daz * runy
        d_cd_azz[d_idx] += daz * runz

    @override
    def post_loop(self, d_idx, d_rho, d_rho_sum, d_h, d_cs, d_omega, d_alphaloc, d_xilimiter, d_divvdt, d_cd_rxx, d_cd_rxy, d_cd_rxz, d_cd_ryy, d_cd_ryz, d_cd_rzz, d_cd_vxx, d_cd_vxy, d_cd_vxz, d_cd_vyx, d_cd_vyy, d_cd_vyz, d_cd_vzx, d_cd_vzy, d_cd_vzz, d_cd_axx, d_cd_axy, d_cd_axz, d_cd_ayx, d_cd_ayy, d_cd_ayz, d_cd_azx, d_cd_azy, d_cd_azz):
        rxx = d_cd_rxx[d_idx]
        rxy = d_cd_rxy[d_idx]
        rxz = d_cd_rxz[d_idx]
        ryy = d_cd_ryy[d_idx]
        ryz = d_cd_ryz[d_idx]
        rzz = d_cd_rzz[d_idx]
        denom = rxx * ryy * rzz + 2.0 * rxy * rxz * ryz - rxx * ryz * ryz - ryy * rxz * rxz - rzz * rxy * rxy
        termnorm = d_omega[d_idx] / d_rho[d_idx]

        if abs(denom) > FLOAT_TINY:
            ddenom = 1.0 / denom
            term_xx = ryy * rzz - ryz * ryz
            term_xy = rxz * ryz - rzz * rxy
            term_xz = rxy * ryz - rxz * ryy
            term_yy = rzz * rxx - rxz * rxz
            term_yz = rxy * rxz - rxx * ryz
            term_zz = rxx * ryy - rxy * rxy

            gradaxdx = (d_cd_axx[d_idx] * term_xx + d_cd_axy[d_idx] * term_xy + d_cd_axz[d_idx] * term_xz) * ddenom
            gradaydy = (d_cd_ayx[d_idx] * term_xy + d_cd_ayy[d_idx] * term_yy + d_cd_ayz[d_idx] * term_yz) * ddenom
            gradazdz = (d_cd_azx[d_idx] * term_xz + d_cd_azy[d_idx] * term_yz + d_cd_azz[d_idx] * term_zz) * ddenom
            div_a = -(gradaxdx + gradaydy + gradazdz)

            gradvxdx = (d_cd_vxx[d_idx] * term_xx + d_cd_vxy[d_idx] * term_xy + d_cd_vxz[d_idx] * term_xz) * ddenom
            gradvxdy = (d_cd_vxx[d_idx] * term_xy + d_cd_vxy[d_idx] * term_yy + d_cd_vxz[d_idx] * term_yz) * ddenom
            gradvxdz = (d_cd_vxx[d_idx] * term_xz + d_cd_vxy[d_idx] * term_yz + d_cd_vxz[d_idx] * term_zz) * ddenom
            gradvydx = (d_cd_vyx[d_idx] * term_xx + d_cd_vyy[d_idx] * term_xy + d_cd_vyz[d_idx] * term_xz) * ddenom
            gradvydy = (d_cd_vyx[d_idx] * term_xy + d_cd_vyy[d_idx] * term_yy + d_cd_vyz[d_idx] * term_yz) * ddenom
            gradvydz = (d_cd_vyx[d_idx] * term_xz + d_cd_vyy[d_idx] * term_yz + d_cd_vyz[d_idx] * term_zz) * ddenom
            gradvzdx = (d_cd_vzx[d_idx] * term_xx + d_cd_vzy[d_idx] * term_xy + d_cd_vzz[d_idx] * term_xz) * ddenom
            gradvzdy = (d_cd_vzx[d_idx] * term_xy + d_cd_vzy[d_idx] * term_yy + d_cd_vzz[d_idx] * term_yz) * ddenom
            gradvzdz = (d_cd_vzx[d_idx] * term_xz + d_cd_vzy[d_idx] * term_yz + d_cd_vzz[d_idx] * term_zz) * ddenom

            dvxdx = -gradvxdx
            dvxdy = -gradvxdy
            dvxdz = -gradvxdz
            dvydx = -gradvydx
            dvydy = -gradvydy
            dvydz = -gradvydz
            dvzdx = -gradvzdx
            dvzdy = -gradvzdy
            dvzdz = -gradvzdz
        else:
            div_a = -termnorm * (d_cd_axx[d_idx] + d_cd_ayy[d_idx] + d_cd_azz[d_idx])
            dvxdx = -termnorm * d_cd_vxx[d_idx]
            dvxdy = -termnorm * d_cd_vxy[d_idx]
            dvxdz = -termnorm * d_cd_vxz[d_idx]
            dvydx = -termnorm * d_cd_vyx[d_idx]
            dvydy = -termnorm * d_cd_vyy[d_idx]
            dvydz = -termnorm * d_cd_vyz[d_idx]
            dvzdx = -termnorm * d_cd_vzx[d_idx]
            dvzdy = -termnorm * d_cd_vzy[d_idx]
            dvzdz = -termnorm * d_cd_vzz[d_idx]

        divvdt = div_a - (dvxdx * dvxdx + dvydy * dvydy + dvzdz * dvzdz + 2.0 * (dvxdy * dvydx + dvxdz * dvzdx + dvydz * dvzdy))

        divv = dvxdx + dvydy + dvzdz
        curlvx = dvzdy - dvydz
        curlvy = dvxdz - dvzdx
        curlvz = dvydx - dvxdy
        compression = max(-divv, 0.0)
        fac = compression * compression
        trace_s = curlvx * curlvx + curlvy * curlvy + curlvz * curlvz
        denominator = fac + trace_s
        xilimiter = fac / denominator if denominator > 0.0 else 1.0

        d_divvdt[d_idx] = divvdt
        d_xilimiter[d_idx] = xilimiter
        source = 10.0 * d_h[d_idx] * d_h[d_idx] * d_xilimiter[d_idx] * max(-d_divvdt[d_idx], 0.0)
        temp = d_cs[d_idx] * d_cs[d_idx]
        alphaloc = max(min(source / temp, 1.0), 0.0)
        d_alphaloc[d_idx] = alphaloc


class StrainDiagnostics(Equation):
    @override
    def initialize(self, d_idx, d_rho, d_omega, d_cd_rxx, d_cd_rxy, d_cd_rxz, d_cd_ryy, d_cd_ryz, d_cd_rzz, d_cd_vxx, d_cd_vxy, d_cd_vxz, d_cd_vyx, d_cd_vyy, d_cd_vyz, d_cd_vzx, d_cd_vzy, d_cd_vzz, d_dvdx_xx, d_dvdx_xy, d_dvdx_xz, d_dvdx_yx, d_dvdx_yy, d_dvdx_yz, d_dvdx_zx, d_dvdx_zy, d_dvdx_zz):
        rxx = d_cd_rxx[d_idx]
        rxy = d_cd_rxy[d_idx]
        rxz = d_cd_rxz[d_idx]
        ryy = d_cd_ryy[d_idx]
        ryz = d_cd_ryz[d_idx]
        rzz = d_cd_rzz[d_idx]
        denom = rxx * ryy * rzz + 2.0 * rxy * rxz * ryz - rxx * ryz * ryz - ryy * rxz * rxz - rzz * rxy * rxy

        if abs(denom) > FLOAT_TINY:
            ddenom = 1.0 / denom
            term_xx = ryy * rzz - ryz * ryz
            term_xy = rxz * ryz - rzz * rxy
            term_xz = rxy * ryz - rxz * ryy
            term_yy = rzz * rxx - rxz * rxz
            term_yz = rxy * rxz - rxx * ryz
            term_zz = rxx * ryy - rxy * rxy

            d_dvdx_xx[d_idx] = -(d_cd_vxx[d_idx] * term_xx + d_cd_vxy[d_idx] * term_xy + d_cd_vxz[d_idx] * term_xz) * ddenom
            d_dvdx_xy[d_idx] = -(d_cd_vxx[d_idx] * term_xy + d_cd_vxy[d_idx] * term_yy + d_cd_vxz[d_idx] * term_yz) * ddenom
            d_dvdx_xz[d_idx] = -(d_cd_vxx[d_idx] * term_xz + d_cd_vxy[d_idx] * term_yz + d_cd_vxz[d_idx] * term_zz) * ddenom
            d_dvdx_yx[d_idx] = -(d_cd_vyx[d_idx] * term_xx + d_cd_vyy[d_idx] * term_xy + d_cd_vyz[d_idx] * term_xz) * ddenom
            d_dvdx_yy[d_idx] = -(d_cd_vyx[d_idx] * term_xy + d_cd_vyy[d_idx] * term_yy + d_cd_vyz[d_idx] * term_yz) * ddenom
            d_dvdx_yz[d_idx] = -(d_cd_vyx[d_idx] * term_xz + d_cd_vyy[d_idx] * term_yz + d_cd_vyz[d_idx] * term_zz) * ddenom
            d_dvdx_zx[d_idx] = -(d_cd_vzx[d_idx] * term_xx + d_cd_vzy[d_idx] * term_xy + d_cd_vzz[d_idx] * term_xz) * ddenom
            d_dvdx_zy[d_idx] = -(d_cd_vzx[d_idx] * term_xy + d_cd_vzy[d_idx] * term_yy + d_cd_vzz[d_idx] * term_yz) * ddenom
            d_dvdx_zz[d_idx] = -(d_cd_vzx[d_idx] * term_xz + d_cd_vzy[d_idx] * term_yz + d_cd_vzz[d_idx] * term_zz) * ddenom
        else:
            termnorm = d_omega[d_idx] / d_rho[d_idx]
            d_dvdx_xx[d_idx] = -termnorm * d_cd_vxx[d_idx]
            d_dvdx_xy[d_idx] = -termnorm * d_cd_vxy[d_idx]
            d_dvdx_xz[d_idx] = -termnorm * d_cd_vxz[d_idx]
            d_dvdx_yx[d_idx] = -termnorm * d_cd_vyx[d_idx]
            d_dvdx_yy[d_idx] = -termnorm * d_cd_vyy[d_idx]
            d_dvdx_yz[d_idx] = -termnorm * d_cd_vyz[d_idx]
            d_dvdx_zx[d_idx] = -termnorm * d_cd_vzx[d_idx]
            d_dvdx_zy[d_idx] = -termnorm * d_cd_vzy[d_idx]
            d_dvdx_zz[d_idx] = -termnorm * d_cd_vzz[d_idx]


class MagneticStateReconstruction(Equation):
    @override
    def initialize(self, d_idx, d_rho, d_Bevolx, d_Bevoly, d_Bevolz, d_Bevolxpred, d_Bevolypred, d_Bevolzpred, d_Bx, d_By, d_Bz, d_Bxpred, d_Bypred, d_Bzpred):
        bxpred = d_Bevolxpred[d_idx] * d_rho[d_idx]
        bypred = d_Bevolypred[d_idx] * d_rho[d_idx]
        bzpred = d_Bevolzpred[d_idx] * d_rho[d_idx]
        d_Bx[d_idx] = bxpred
        d_By[d_idx] = bypred
        d_Bz[d_idx] = bzpred
        d_Bxpred[d_idx] = bxpred
        d_Bypred[d_idx] = bypred
        d_Bzpred[d_idx] = bzpred


class MagneticStateRates(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)
        self.mu0 = MU0

    @override
    def initialize(self, d_idx, d_aBevolx, d_aBevoly, d_aBevolz, d_apsi, d_divBsymm, d_divBdiff):
        d_aBevolx[d_idx] = 0.0
        d_aBevoly[d_idx] = 0.0
        d_aBevolz[d_idx] = 0.0
        d_apsi[d_idx] = 0.0
        d_divBsymm[d_idx] = 0.0
        d_divBdiff[d_idx] = 0.0

    @override
    def loop(self, d_idx, s_idx, d_m, s_m, d_rho, s_rho, d_upred, d_vpred, d_wpred, s_upred, s_vpred, s_wpred, d_Bxpred, d_Bypred, d_Bzpred, s_Bxpred, s_Bypred, s_Bzpred, d_psipred, s_psipred, d_cs, s_cs, d_omega, s_omega, d_aBevolx, d_aBevoly, d_aBevolz, d_ae, d_divBsymm, d_divBdiff, XIJ, DWI, DWJ, RIJ):
        if RIJ > PAIR_DISTANCE_EPSILON:
            runix = XIJ[0] / RIJ
            runiy = XIJ[1] / RIJ
            runiz = XIJ[2] / RIJ
        else:
            runix = 0.0
            runiy = 0.0
            runiz = 0.0

        grad_i = d_omega[d_idx] * (runix * DWI[0] + runiy * DWI[1] + runiz * DWI[2])
        grad_j = s_omega[s_idx] * (runix * DWJ[0] + runiy * DWJ[1] + runiz * DWJ[2])
        rho_i = d_rho[d_idx]
        rho_j = s_rho[s_idx]
        bx_i = d_Bxpred[d_idx]
        by_i = d_Bypred[d_idx]
        bz_i = d_Bzpred[d_idx]
        bx_j = s_Bxpred[s_idx]
        by_j = s_Bypred[s_idx]
        bz_j = s_Bzpred[s_idx]
        proj_bi = bx_i * runix + by_i * runiy + bz_i * runiz
        proj_bj = bx_j * runix + by_j * runiy + bz_j * runiz
        dbx = bx_i - bx_j
        dby = by_i - by_j
        dbz = bz_i - bz_j
        proj_db = dbx * runix + dby * runiy + dbz * runiz
        delta_b2 = dbx * dbx + dby * dby + dbz * dbz
        rho_i2_inv = 1.0 / (rho_i * rho_i)
        rho_j2_inv = 1.0 / (rho_j * rho_j)
        pmj_rho21_grad_i = s_m[s_idx] * rho_i2_inv * grad_i
        pmj_rho21_grad_j = s_m[s_idx] * rho_j2_inv * grad_j
        d_brho_term = -(pmj_rho21_grad_i * proj_bi)
        dvx = d_upred[d_idx] - s_upred[s_idx]
        dvy = d_vpred[d_idx] - s_vpred[s_idx]
        dvz = d_wpred[d_idx] - s_wpred[s_idx]
        projv = dvx * runix + dvy * runiy + dvz * runiz
        dvxt = dvx - projv * runix
        dvyt = dvy - projv * runiy
        dvzt = dvz - projv * runiz
        vsig_b = sqrt(dvxt * dvxt + dvyt * dvyt + dvzt * dvzt)
        d_b_diss_term = 0.5 * (d_m[d_idx] * rho_i2_inv * grad_i + s_m[s_idx] * rho_j2_inv * grad_j) * vsig_b
        vwave_i = sqrt(d_cs[d_idx] * d_cs[d_idx] + (bx_i * bx_i + by_i * by_i + bz_i * bz_i) / (self.mu0 * rho_i))
        vwave_j = sqrt(s_cs[s_idx] * s_cs[s_idx] + (bx_j * bx_j + by_j * by_j + bz_j * bz_j) / (self.mu0 * rho_j))
        dpsi_term = pmj_rho21_grad_i * d_psipred[d_idx] * vwave_i + pmj_rho21_grad_j * s_psipred[s_idx] * vwave_j

        d_aBevolx[d_idx] += d_brho_term * dvx + d_b_diss_term * dbx - dpsi_term * runix
        d_aBevoly[d_idx] += d_brho_term * dvy + d_b_diss_term * dby - dpsi_term * runiy
        d_aBevolz[d_idx] += d_brho_term * dvz + d_b_diss_term * dbz - dpsi_term * runiz
        d_ae[d_idx] += -0.5 * delta_b2 * d_b_diss_term / self.mu0
        d_divBsymm[d_idx] += pmj_rho21_grad_i * proj_bi + pmj_rho21_grad_j * proj_bj
        d_divBdiff[d_idx] += -s_m[s_idx] * proj_db * grad_i

    @override
    def post_loop(self, d_idx, d_rho, d_h, d_psipred, d_cs, d_div, d_div_for_psi, d_Bxpred, d_Bypred, d_Bzpred, d_apsi, d_divBdiff, d_dt_cfl):
        vwave = sqrt(d_cs[d_idx] * d_cs[d_idx] + (d_Bxpred[d_idx] * d_Bxpred[d_idx] + d_Bypred[d_idx] * d_Bypred[d_idx] + d_Bzpred[d_idx] * d_Bzpred[d_idx]) / (self.mu0 * d_rho[d_idx]))
        d_apsi[d_idx] = -vwave * d_divBdiff[d_idx] / d_rho[d_idx] - d_psipred[d_idx] * vwave / d_h[d_idx] - 0.5 * d_psipred[d_idx] * d_div_for_psi[d_idx]
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], vwave)


class AdaptiveTimestep(Equation):
    def __init__(self, dest, sources):
        self.c_cour = COURANT_FACTOR
        self.c_force = FORCE_FACTOR
        self.bignumber = 1.0e29
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_h, d_dt_cfl, d_au, d_av, d_aw, d_dt_adapt):
        dt_adapt = self.bignumber
        if d_dt_cfl[d_idx] > 0.0:
            dt_adapt = self.c_cour * d_h[d_idx] / d_dt_cfl[d_idx]
        f2 = d_au[d_idx] * d_au[d_idx] + d_av[d_idx] * d_av[d_idx] + d_aw[d_idx] * d_aw[d_idx]
        if f2 > 0.0:
            dt_adapt = min(dt_adapt, self.c_force * sqrt(d_h[d_idx] / sqrt(f2)))
        d_dt_adapt[d_idx] = dt_adapt


class EnergyLimiter(Equation):
    def __init__(self, dest, sources):
        self.c_cour = COURANT_FACTOR
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_epred, d_ae, d_h, d_dt_cfl):
        if d_dt_cfl[d_idx] > 0.0:
            dtc = self.c_cour * d_h[d_idx] / d_dt_cfl[d_idx]
            if d_epred[d_idx] + dtc * d_ae[d_idx] < 0.0 and d_epred[d_idx] > 0.0:
                d_ae[d_idx] /= 1.0 - dtc * d_ae[d_idx] / d_epred[d_idx]


class DivBCorrection(Equation):
    def __init__(self, dest, sources):
        self.mu0 = MU0
        super().__init__(dest, sources)

    @override
    def initialize(self, d_idx, d_rho, d_p, d_Bxpred, d_Bypred, d_Bzpred, d_divBsymm, d_au, d_av, d_aw):
        bx = d_Bxpred[d_idx]
        by = d_Bypred[d_idx]
        bz = d_Bzpred[d_idx]
        b2 = bx * bx + by * by + bz * bz
        divb = d_divBsymm[d_idx]
        frac_divb = 0.0
        if b2 > 0.0:
            beta = 2.0 * self.mu0 * d_p[d_idx] / b2
            frac_divb = max(0.0, min(1.0, (10.0 - beta) * 0.125))

        ax = -bx * divb * frac_divb / self.mu0
        ay = -by * divb * frac_divb / self.mu0
        az = -bz * divb * frac_divb / self.mu0
        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az
        d_divBsymm[d_idx] = d_rho[d_idx] * divb
