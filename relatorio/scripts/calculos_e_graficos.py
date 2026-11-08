"""Grupo I (Pião) - cálculos, validação numérica e figuras do relatório.

Uso:  python calculos_e_graficos.py
Gera os PDFs em ../figuras e imprime todos os valores usados no relatório.

A validação integra as equações de Euler em eixos do corpo com quatérnios;
ela NÃO usa a equação de precessão permanente nem os ângulos de Euler, de modo
que constitui uma verificação independente da solução analítica.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext

FIG = Path(__file__).resolve().parent.parent / "figuras"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- dados
m = 0.100        # kg
k_ax = 0.025     # m  (raio de giração axial)
k_tr = 0.050     # m  (raio de giração transversal em relação a O)
c = 0.040        # m
g = 9.81         # m/s²
RPM = 2 * np.pi / 60
psi_d = 1800 * RPM

I = m * k_ax**2
Ip = m * k_tr**2
W = m * g
Wc = W * c


def raizes(theta, psid=psi_d):
    """Raízes da equação de precessão permanente (lenta, rápida)."""
    a = (Ip - I) * np.cos(theta)
    b = I * psid
    d = np.sqrt(b**2 - 4 * a * Wc)
    return (b - d) / (2 * a), (b + d) / (2 * a)


phi_aprox = Wc / (I * psi_d)


def erro_pct(theta, psid=psi_d):
    lenta, _ = raizes(theta, psid)
    return (Wc / (I * psid) - lenta) / lenta * 100


def psi_min(theta):
    return 2 / I * np.sqrt((Ip - I) * np.cos(theta) * Wc)


# ---------------------------------------------------------------- simulação
def qmul(a, b):
    w1, v1 = a[0], a[1:]
    w2, v2 = b[0], b[1:]
    return np.r_[w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2)]


def qrot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


J = np.array([Ip, Ip, I])


def deriv(s):
    q, om = s[:4], s[4:]
    R = qrot(q)
    # peso aplicado em G = c k_corpo; momento em relação a O expresso no corpo
    M = np.cross([0, 0, c], R.T @ np.array([0, 0, -W]))
    domega = (M - np.cross(om, J * om)) / J
    dq = 0.5 * qmul(q, np.r_[0, om])
    return np.r_[dq, domega]


CACHE = Path(__file__).resolve().parent / "cache_simulacoes"


def simula(theta0, phid0, psid0, thetad0=0.0, T=1.0, dt=2e-5, every=25):
    """Devolve a simulação guardada em disco ou a calcula.

    A pasta cache_simulacoes pode ser apagada: ela é recriada. Apague-a se
    alterar as equações de deriv(); mudanças nos dados já geram novas entradas.
    """
    chave = repr((m, k_ax, k_tr, c, g, theta0, phid0, psid0, thetad0, T, dt, every))
    arq = CACHE / (hashlib.sha1(chave.encode()).hexdigest() + ".npy")
    if arq.exists():
        return tuple(np.load(arq))
    res = _integra(theta0, phid0, psid0, thetad0, T, dt, every)
    CACHE.mkdir(exist_ok=True)
    np.save(arq, np.array(res))
    return res


def _integra(theta0, phid0, psid0, thetad0, T, dt, every):
    """Integra (RK4) a partir de psi = 0, phi = 0."""
    q = np.array([np.cos(theta0 / 2), np.sin(theta0 / 2), 0.0, 0.0])  # giro em x
    # velocidade angular no corpo: phid K + psid k + thetad i
    R = qrot(q)
    om = R.T @ np.array([0, 0, phid0]) + np.array([0, 0, psid0]) + np.array([thetad0, 0, 0])
    s = np.r_[q, om]
    n = int(round(T / dt))
    out_t, out_th, out_phi, out_E = [], [], [], []
    last_phi, turn = None, 0.0
    for i in range(n + 1):
        if i % every == 0:
            R = qrot(s[:4])
            k = R[:, 2]
            ph = np.arctan2(k[0], -k[1])  # azimute do nó: k = (s*sin(phi), -s*cos(phi), c)
            if last_phi is not None:
                dphi = ph - last_phi
                turn += (dphi + np.pi) % (2 * np.pi) - np.pi
            last_phi = ph
            out_t.append(i * dt)
            out_th.append(np.arccos(np.clip(k[2], -1, 1)))
            out_phi.append(turn)
            om_ = s[4:]
            out_E.append(0.5 * np.sum(J * om_**2) + Wc * k[2])
        k1 = deriv(s)
        k2 = deriv(s + dt / 2 * k1)
        k3 = deriv(s + dt / 2 * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[:4] /= np.linalg.norm(s[:4])
    return tuple(map(np.array, (out_t, out_th, out_phi, out_E)))


# ---------------------------------------------------------------- estilo
AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10.5,
    "pdf.fonttype": 42,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "axes.linewidth": 0.7,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.bbox": "tight",
})


def fmt_dec(dec=None):
    """Separador decimal em vírgula; dec fixa o número de casas."""
    if dec is None:
        return FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    return FuncFormatter(lambda v, _: f"{v:.{dec}f}".replace(".", ","))


def virgula(f, ydec=None, xdec=None):
    """Aplica vírgula decimal; ydec/xdec = {índice do painel: casas decimais}."""
    ydec, xdec = ydec or {}, xdec or {}
    for i, a in enumerate(f.axes):
        a.yaxis.set_major_formatter(fmt_dec(ydec.get(i)))
        if a.get_xscale() == "linear":
            a.xaxis.set_major_formatter(fmt_dec(xdec.get(i)))


TH = np.deg2rad(np.linspace(30, 60, 121))
THD = np.rad2deg(TH)


def fig_taxas(fs=(6.6, 2.7), suf=""):
    lenta, rapida = raizes(TH)
    f, ax = plt.subplots(1, 2, figsize=fs)
    ax[0].plot(THD, lenta, color=AZUL)
    ax[0].set_title(r"(a) Precessão lenta $\dot{\phi}_1$", fontsize=10, loc="left")
    ax[1].plot(THD, rapida, color=LARANJA, ls="--")
    ax[1].set_title(r"(b) Precessão rápida $\dot{\phi}_2$", fontsize=10, loc="left")
    for a, y in zip(ax, (lenta, rapida)):
        a.set_xlabel(r"$\theta$ (°)")
        a.set_xticks(range(30, 61, 5))
        for i in (0, -1):
            a.plot(THD[i], y[i], "o", ms=5, color=a.lines[0].get_color(),
                   mec="white", mew=1, clip_on=False, zorder=5)
    ax[0].set_ylim(3.41, 3.51)
    ax[0].set_ylabel(r"$\dot{\phi}_1$ (rad/s)")
    ax[1].set_ylabel(r"$\dot{\phi}_2$ (rad/s)")
    ax[0].annotate("3,50", (30, lenta[0]), xytext=(7, 0), textcoords="offset points", fontsize=9, color=TINTA)
    ax[0].annotate("3,42", (60, lenta[-1]), xytext=(-7, -3), textcoords="offset points", fontsize=9, color=TINTA, ha="right")
    ax[1].annotate("69,1", (30, rapida[0]), xytext=(8, -3), textcoords="offset points", fontsize=9, color=TINTA)
    ax[1].annotate("122,2", (60, rapida[-1]), xytext=(-8, -2), textcoords="offset points", fontsize=9, color=TINTA, ha="right")
    virgula(f, ydec={0: 2})
    f.tight_layout()
    f.savefig(FIG / f"taxas_precessao{suf}.pdf")


def fig_erro(fs=(6.6, 2.7), suf=""):
    lenta, _ = raizes(TH)
    f, ax = plt.subplots(1, 2, figsize=fs)
    ax[0].plot(THD, lenta, color=AZUL, label=r"exata, $\dot{\phi}_1$")
    ax[0].plot(THD, np.full_like(THD, phi_aprox), color=LARANJA, ls="--",
               label=r"aproximada, $\dot{\phi}_a$")
    ax[0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
    ax[0].set_ylim(3.25, 3.60)
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title("(a) Taxa de precessão lenta", fontsize=10, loc="left")
    err = np.abs(erro_pct(TH))
    ax[1].plot(THD, err, color=AZUL)
    ax[1].set_ylabel(r"$|e|$ (%)")
    ax[1].set_ylim(0, 6)
    ax[1].set_title("(b) Erro percentual da aproximação", fontsize=10, loc="left")
    for i, off, ha in ((0, (6, 6), "left"), (-1, (-5, 13), "right")):
        ax[1].plot(THD[i], err[i], "o", ms=5, color=AZUL, mec="white", mew=1, clip_on=False, zorder=5)
        ax[1].annotate(f"{err[i]:.2f}%".replace(".", ","), (THD[i], err[i]), xytext=off,
                       textcoords="offset points", fontsize=9, color=TINTA, ha=ha)
    for a in ax:
        a.set_xlabel(r"$\theta$ (°)")
        a.set_xticks(range(30, 61, 5))
    virgula(f, ydec={0: 2})
    f.tight_layout()
    f.savefig(FIG / f"erro_aproximacao{suf}.pdf")


def fig_spin(fs=(6.6, 2.8), suf=""):
    th = np.deg2rad(30)
    pm = psi_min(th)
    psid = np.linspace(pm, 400, 600)
    l, r = raizes(th, psid)
    f, ax = plt.subplots(1, 2, figsize=fs)
    ax[0].semilogy(psid / RPM, l, color=AZUL, label=r"$\dot{\phi}_1$ (lenta)")
    ax[0].semilogy(psid / RPM, r, color=LARANJA, ls="--", label=r"$\dot{\phi}_2$ (rápida)")
    ax[0].axvline(1800, color=TINTA2, lw=0.8, ls=":")
    ax[0].text(1840, 1.25, "1800 rpm", fontsize=8.5, color=TINTA2, va="bottom")
    ax[0].set_ylim(1, 300)
    ax[0].plot(pm / RPM, raizes(th, pm)[0], "o", ms=5, color=TINTA, mec="white", clip_on=False, zorder=5)
    ax[0].annotate(r"$\dot{\psi}_{\mathrm{min}}$", (pm / RPM, raizes(th, pm)[0]),
                   xytext=(7, 5), textcoords="offset points", fontsize=9.5, color=TINTA)
    ax[0].set_xlabel(r"$\dot{\psi}$ (rpm)")
    ax[0].set_ylabel(r"$\dot{\phi}$ (rad/s)")
    ax[0].legend(loc="center right", fontsize=9)
    ax[0].set_title(r"(a) Raízes para $\theta$ = 30°", fontsize=10, loc="left")
    p2 = np.linspace(1000, 6000, 300) * RPM
    for td, col, ls in ((30, AZUL, "-"), (60, LARANJA, "--")):
        t = np.deg2rad(td)
        ax[1].loglog(p2 / RPM, np.abs(erro_pct(t, p2)), color=col, ls=ls, label=rf"$\theta$ = {td}°")
    ax[1].axvline(1800, color=TINTA2, lw=0.8, ls=":")
    ax[1].set_xlabel(r"$\dot{\psi}$ (rpm)")
    ax[1].set_ylabel(r"$|e|$ (%)")
    ax[1].set_xticks([1000, 2000, 4000, 6000])
    ax[1].set_xticklabels(["1000", "2000", "4000", "6000"])
    ax[1].minorticks_off()
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].set_title("(b) Erro da aproximação", fontsize=10, loc="left")
    virgula(f)
    ax[1].set_xticks([1000, 2000, 4000, 6000])
    ax[1].set_xticklabels(["1000", "2000", "4000", "6000"])
    ax[1].minorticks_off()
    f.tight_layout()
    f.savefig(FIG / f"efeito_spin{suf}.pdf")


def fig_simulacao(fs=(3.6, 2.8), suf=""):
    th = np.deg2rad(30)
    l, r = raizes(th)
    t, _, pl, _ = simula(th, l, psi_d, T=1.0)
    f, ax = plt.subplots(1, 1, figsize=fs)
    ax.plot(t, pl, color=AZUL, label=r"$\dot{\phi}_1$")
    ax.set_ylabel(r"$\phi$ (rad)")
    ax.set_xlabel(r"$t$ (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Ângulo de precessão", fontsize=10, loc="left")
    virgula(f, xdec={0: 1})
    f.tight_layout()
    f.savefig(FIG / f"simulacao{suf}.pdf")


def fig_apendice():
    """Grandezas da simulação para as duas raízes, em theta = 30 graus."""
    th = np.deg2rad(30)
    lenta, rapida = raizes(th)
    t, th1, ph1, E1 = simula(th, lenta, psi_d, T=1.0)
    _, th2, ph2, E2 = simula(th, rapida, psi_d, T=1.0)
    f, ax = plt.subplots(2, 2, figsize=(6.6, 4.8))
    ax[0, 0].plot(t, ph1, color=AZUL)
    ax[0, 0].set_ylabel(r"$\phi$ (rad)")
    ax[0, 0].set_title(r"(a) Raiz lenta $\dot{\phi}_1$", fontsize=10, loc="left")
    ax[0, 1].plot(t, ph2, color=LARANJA, ls="--")
    ax[0, 1].set_ylabel(r"$\phi$ (rad)")
    ax[0, 1].set_title(r"(b) Raiz rápida $\dot{\phi}_2$", fontsize=10, loc="left")
    ax[1, 0].plot(t, np.rad2deg(th1), color=AZUL, label="lenta")
    ax[1, 0].plot(t, np.rad2deg(th2), color=LARANJA, ls="--", label="rápida")
    ax[1, 0].set_ylim(29, 31)
    ax[1, 0].set_ylabel(r"$\theta$ (°)")
    ax[1, 0].legend(loc="upper right", fontsize=9)
    ax[1, 0].set_title("(c) Ângulo de inclinação", fontsize=10, loc="left")
    piso = 1e-17  # evita log(0) nos instantes em que a variação é exatamente nula
    ax[1, 1].semilogy(t, np.abs(E1 - E1[0]) / abs(E1[0]) + piso, color=AZUL, lw=1, label="lenta")
    ax[1, 1].semilogy(t, np.abs(E2 - E2[0]) / abs(E2[0]) + piso, color=LARANJA, lw=1, label="rápida")
    ax[1, 1].set_ylabel(r"$|E-E_0|/E_0$")
    ax[1, 1].legend(loc="lower right", fontsize=9)
    ax[1, 1].set_title("(d) Variação relativa da energia", fontsize=10, loc="left")
    virgula(f, xdec={0: 1, 1: 1, 2: 1, 3: 1})
    ax[1, 1].yaxis.set_major_formatter(LogFormatterMathtext())
    for a in ax[1]:
        a.set_xlabel(r"$t$ (s)")
    f.tight_layout()
    f.savefig(FIG / "simulacao_apendice.pdf")


# ---------------------------------------------------------------- relatório
def relatorio_numerico():
    print(f"I={I:.4e}  I'={Ip:.4e}  W={W:.5f}  Wc={Wc:.5f}  psi_d={psi_d:.4f} rad/s")
    print(f"b=I psi_d={I*psi_d:.6f}  phi_a={phi_aprox:.4f} rad/s = {phi_aprox/RPM:.3f} rpm")
    print("\ntheta  a(=(I'-I)cos)   phi1(rad/s) rpm  phi2(rad/s) rpm  eps  err%  psi_min(rpm)")
    for td in range(30, 61, 5):
        t = np.deg2rad(td)
        l, r = raizes(t)
        a = (Ip - I) * np.cos(t)
        print(f"{td:3d} {a:.5e} {l:8.4f} {l/RPM:8.3f} {r:9.3f} {r/RPM:9.2f} "
              f"{a*Wc/(I*psi_d)**2:.5f} {erro_pct(t):7.3f} {psi_min(t)/RPM:7.1f}")
    print(f"psi_min(30)={psi_min(np.deg2rad(30)):.3f} rad/s")

    print("\ntheta ramo  wz     H_z     Hx      T     E      Rh    Rh/W  phi/psi")
    for td in (30, 45, 60):
        t = np.deg2rad(td)
        for nome, p in zip(("lenta", "rapida"), raizes(t)):
            wz = psi_d + p * np.cos(t)
            T = 0.5 * I * wz**2 + 0.5 * Ip * (p * np.sin(t))**2
            E = T + Wc * np.cos(t)
            Rh = m * c * np.sin(t) * p**2
            print(f"{td} {nome:6s} {wz:7.2f} {I*wz:.5f} {Ip*p*np.sin(t):.5f} {T:6.3f} {E:6.3f} "
                  f"{Rh:7.3f} {Rh/W:6.3f} {p/psi_d:.4f}")

    print("\nValidação numérica (Euler + quatérnios, RK4, dt=2e-5 s, T=1 s)")
    for td in (30, 45, 60):
        t = np.deg2rad(td)
        for nome, p in zip(("lenta", "rapida"), raizes(t)):
            tt, th, ph, E = simula(t, p, psi_d, T=1.0)
            slope = np.polyfit(tt, ph, 1)[0]
            print(f"{td} {nome:6s} phi_ana={p:9.4f} phi_num={slope:9.4f} "
                  f"desv={abs(slope-p)/p*100:.2e}% max|dtheta|={np.rad2deg(np.abs(th-t).max()):.2e} graus "
                  f"dE/E={np.abs(E-E[0]).max()/abs(E[0]):.1e}")


if __name__ == "__main__":
    relatorio_numerico()
    # versões do relatório e versões para projeção (letras proporcionalmente maiores)
    fig_taxas()
    fig_erro()
    fig_spin()
    fig_simulacao()
    fig_apendice()
    fig_taxas((5.0, 2.9), "_slide")
    fig_erro((5.0, 2.9), "_slide")
    fig_spin((5.2, 2.5), "_slide")
    fig_simulacao((5.0, 2.9), "_slide")
    print("\nFiguras geradas em", FIG)
