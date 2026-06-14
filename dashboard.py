"""Interactive fault-explorer dashboard for the Arteche DDB-123 CVT model.

    streamlit run dashboard.py

Layout
------
* Left sidebar: Type A pollution severity, Type B streamer length, burden, and a
  Run-FEA button.
* Metrics: the diagnostic tuple (phase shift, tan delta, leakage current).
* "The Map": a bubble scatter of phase shift vs ratio error, with leakage
  current as the bubble colour AND size.  The burden slider recomputes it
  INSTANTLY from cached FEA observables (no re-solve).
* "The Physics": the 3-D phi_phase_mrad field of your last run, embedded
  interactively (stpyvista) with a static snapshot fallback.

Everything solves in 3-D on one cached scikit-fem mesh, so the baseline library
and your runs are consistent.  First load solves the library (~20-30 s); after
that the burden slider is instant and each run is a single ~5 s solve.
"""
from __future__ import annotations

import math
import os

import numpy as np
import streamlit as st

from ctfem.config import CVTParams, OperatingParams
from ctfem.util import results_dir
from ctfem import dashboard_data as dd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

st.set_page_config(page_title="CVT Fault Explorer", layout="wide")

# category -> (colour, marker, size) for the matplotlib fallback scatter
_CAT_STYLE = {
    "healthy": ("black", "*", 320),
    "internal": ("tab:red", "o", 130),
    "surface": ("tab:green", "s", 110),
    "user": ("tab:blue", "D", 200),
}


@st.cache_resource(show_spinner=False)
def _setup(refine: float):
    cvt = CVTParams(mesh_refinement=refine)
    op = OperatingParams(um_kv=123.0)
    matdb = cvt.material_db()
    out = results_dir("dashboard")
    msh = os.path.join(out, f"cvt3d_r{refine}.msh")
    if not os.path.exists(msh):
        dd.build_mesh_3d(cvt, msh)
    return cvt, op, matdb, msh, out


@st.cache_data(show_spinner=False)
def _library(refine: float):
    cvt, op, matdb, msh, _ = _setup(refine)
    return dd.baseline_library(msh, op, matdb, cvt)


@st.cache_data(show_spinner=False)
def _user_run(refine: float, sigma_exp: float, sheds: int):
    cvt, op, matdb, msh, out = _setup(refine)
    png = os.path.join(out, f"run_e{sigma_exp:+.2f}_s{sheds}.png")
    return dd.user_run(msh, op, matdb, cvt, 10.0 ** sigma_exp, sheds, png_path=png)


def _log_leak(mA: float) -> float:
    return math.log10(max(mA, 1e-3))


def _scatter_plotly(cpoints, op_cp=None, envelope=None):
    data = {
        "ratio_err": [cp.ratio_err_pct for cp in cpoints],
        "phase_disp": [cp.phase_disp_min for cp in cpoints],
        "leakage_mA": [cp.leakage_mA for cp in cpoints],
        "log_leak": [_log_leak(cp.leakage_mA) for cp in cpoints],
        "tan_delta": [cp.tan_delta for cp in cpoints],
        "name": [cp.name for cp in cpoints],
        "category": [cp.category for cp in cpoints],
        # bubble size: floor so near-zero-leakage points stay visible
        "bubble": [6.0 + 5.0 * max(0.0, _log_leak(cp.leakage_mA) + 3.0)
                   for cp in cpoints],
    }
    fig = px.scatter(
        data, x="ratio_err", y="phase_disp",
        color="log_leak", size="bubble", symbol="category", hover_name="name",
        hover_data={"leakage_mA": ":.3f", "tan_delta": ":.4f",
                    "ratio_err": ":.2f", "phase_disp": ":.2f",
                    "log_leak": False, "bubble": False, "category": False},
        color_continuous_scale="Turbo", size_max=34,
        labels={"ratio_err": "ratio error [%]",
                "phase_disp": "phase displacement [arc-min]",
                "log_leak": "log10 leak [mA]"},
    )
    if envelope is not None:                 # weather-drift convex hull
        path = ("M " + " L ".join(f"{x},{y}" for x, y in envelope) + " Z")
        fig.add_shape(type="path", path=path, layer="below",
                      fillcolor="rgba(140,140,140,0.18)",
                      line=dict(color="gray", dash="dot", width=1))
        ymax = float(max(p[1] for p in envelope))
        xat = float(envelope[int(np.argmax([p[1] for p in envelope]))][0])
        fig.add_annotation(x=xat, y=ymax, text="weather drift", showarrow=False,
                           font=dict(size=10, color="gray"), yshift=8)
    if op_cp is not None:                    # live operating point
        fig.add_trace(go.Scatter(
            x=[op_cp.ratio_err_pct], y=[op_cp.phase_disp_min],
            mode="markers+text", text=["operating"], textposition="top center",
            marker=dict(symbol="star", size=20, color="black",
                        line=dict(color="white", width=1)),
            name="operating point", hoverinfo="x+y"))
    fig.add_hline(y=0.0, line_color="lightgray")
    fig.add_vline(x=0.0, line_color="lightgray")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10),
                      legend_title_text="fault type")
    return fig


def _scatter_mpl(cpoints, op_cp=None, envelope=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if envelope is not None:
        from matplotlib.patches import Polygon as _Poly
        ax.add_patch(_Poly(envelope, closed=True, facecolor="0.6", alpha=0.18,
                           edgecolor="0.5", ls=":", zorder=1,
                           label="weather drift"))
    xs = [cp.ratio_err_pct for cp in cpoints]
    ys = [cp.phase_disp_min for cp in cpoints]
    cs = [_log_leak(cp.leakage_mA) for cp in cpoints]
    ss = [40 + 60 * max(0.0, _log_leak(cp.leakage_mA) + 3.0) for cp in cpoints]
    sc = ax.scatter(xs, ys, c=cs, s=ss, cmap="turbo", edgecolors="k",
                    linewidths=0.5, zorder=3)
    for cp in cpoints:
        ax.annotate(cp.name, (cp.ratio_err_pct, cp.phase_disp_min),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    if op_cp is not None:
        ax.scatter([op_cp.ratio_err_pct], [op_cp.phase_disp_min], marker="*",
                   s=420, c="k", edgecolors="w", linewidths=1.0, zorder=4)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("ratio error  [%]")
    ax.set_ylabel("phase displacement vs healthy  [arc-min]")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="log10 leakage [mA]")
    fig.tight_layout()
    return fig


def _show_3d(vtu, png):
    """Interactive phi_phase_mrad view (stpyvista); static snapshot fallback."""
    if vtu and os.path.exists(vtu):
        try:
            import pyvista as pv
            from stpyvista import stpyvista
            grid = pv.read(vtu)
            dev = grid.threshold(0.5, scalars="device")     # strip far-field air
            clipped = dev.clip(normal=[0.0, 1.0, 0.0])       # cut to see inside
            pl = pv.Plotter(window_size=[720, 760])
            pl.add_mesh(clipped, scalars="phi_phase_mrad", cmap="turbo",
                        scalar_bar_args={"title": "phase [mrad]"})
            pl.add_axes()
            pl.view_isometric()
            stpyvista(pl, key="phase3d")
            return
        except Exception as exc:    # WebGL/trame hiccup -> fall back to snapshot
            st.caption(f"(interactive 3-D unavailable: {type(exc).__name__}; "
                       "showing snapshot)")
    if png and os.path.exists(png):
        st.image(png, use_container_width=True,
                 caption="phi_phase_mrad [mrad] - where the phase vector twists")
    else:
        st.warning("3-D phase render unavailable (the scatter point is still "
                   "valid).")


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Controls")
refine = st.sidebar.select_slider(
    "3-D mesh refine (speed <-> accuracy)", options=[0.15, 0.20, 0.30],
    value=0.15)

st.sidebar.subheader("Circuit  (instant)")
burden = st.sidebar.slider("Secondary burden  [VA]", 5, 100, 25, 5)

st.sidebar.subheader("Internal faults  (instant)")
c1_short = st.sidebar.slider("C1 element shorting  [%]", 0.0, 10.0, 0.0, 0.5,
                             help="fraction of the C1 (HV) section shorted")
c2_short = st.sidebar.slider("C2 element shorting  [%]", 0.0, 10.0, 0.0, 0.5,
                             help="fraction of the C2 (tap) section shorted")
moisture = st.sidebar.slider("Internal moisture (base C)  [%]", 0.0, 10.0, 0.0,
                             0.5, help="uniform capacitance rise from moisture")

st.sidebar.subheader("Grid conditions  (noise)")
temp_c = st.sidebar.slider("Ambient temperature  [degC]", -20, 40, 20, 1,
                           help="shifts base C via the oil-paper temp. coeff.")
grid_freq = st.sidebar.slider("Grid frequency  [Hz]", 49.80, 50.20, 50.00, 0.01,
                              help="off-nominal frequency detunes the reactor")

st.sidebar.subheader("Physical fault  (needs FEA)")
sigma_exp = st.sidebar.slider("Type A pollution  log10(sigma_s [S])",
                              -9.0, -5.0, -7.0, 0.5,
                              help="uniform surface layer; 1e-7 ~ IEC 'Medium'")
sheds = st.sidebar.slider("Type B streamer length  [sheds, 0 = off]",
                          0, 24, 0, 1,
                          help="vertical streak bridging the upper N sheds; "
                               "~4 = top only (invisible), ~20 = bridges to gnd")
run = st.sidebar.button("Run FEA Simulation", type="primary",
                        use_container_width=True)

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
st.title("DDB-123 CVT - Fault Signature Explorer")

cvt, op, matdb, msh, out = _setup(refine)
with st.spinner("Solving the baseline fault library (first load, ~20-30 s)..."):
    lib = _library(refine)
healthy = lib[0]

if run:
    with st.spinner("Running FEA: 3-D solve + phase render..."):
        st.session_state["run"] = _user_run(refine, sigma_exp, sheds)

points = list(lib)
run_state = st.session_state.get("run")
if run_state is not None:
    points = points + [run_state[0]]

cpoints = dd.circuit_points(points, healthy, burden, op.u0)

# instant operating point: internal faults + grid conditions (no FEA) ----------
op_pt = dd.operating_point(healthy, c1_short=c1_short / 100.0,
                           c2_short=c2_short / 100.0, moisture=moisture / 100.0,
                           temperature_c=temp_c)
op_cp = dd.secondary_signature(op_pt, healthy, burden, op.u0, grid_freq=grid_freq)
envelope = dd.noise_envelope(healthy, burden, op.u0)        # convex-hull vertices
phase_noise = float(np.abs(envelope[:, 1]).max())
ratio_noise = float(np.abs(envelope[:, 0]).max())
detectable = not dd.inside_envelope(envelope, op_cp.ratio_err_pct,
                                    op_cp.phase_disp_min)
# tan delta thermal drift band over the temperature range (summer false-positive)
tand_cold = dd.operating_point(healthy, temperature_c=-20.0).tan_delta
tand_hot = dd.operating_point(healthy, temperature_c=40.0).tan_delta

# --- metrics: the live operating point (internal faults + grid) --------------
st.subheader("Diagnostic readouts  (live operating point)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Phase shift  [arc-min]", f"{op_cp.phase_disp_min:+.3f}",
          help="secondary phase displacement vs healthy @ 50 Hz / 20 degC")
m2.metric("tan delta", f"{op_cp.tan_delta:.4f}",
          delta=f"{op_cp.tan_delta - healthy.tan_delta:+.4f} vs 20 degC")
m3.metric("Leakage  [mA]", f"{op_cp.leakage_mA:.3f}",
          help="internal faults carry ~0 surface leakage; run FEA for pollution")
m4.metric("Ratio error  [%]", f"{op_cp.ratio_err_pct:+.3f}")
if detectable:
    st.success("**Detectable** - the operating point is OUTSIDE the weather-drift "
               f"hull (phase up to +/-{phase_noise:.3f} arc-min, ratio "
               f"+/-{ratio_noise:.3f} %).")
else:
    st.warning("**Below the noise floor** - the operating point is INSIDE the "
               f"noise-floor hull (phase +/-{phase_noise:.3f} arc-min, ratio "
               f"+/-{ratio_noise:.3f} %). Crank a fault until the star leaves the "
               "grey hull.")
st.caption(f"tan delta thermal drift over -20..40 degC: "
           f"{min(tand_cold, tand_hot):.4f} -> {max(tand_cold, tand_hot):.4f} "
           f"(summer heat raises loss to {tand_hot:.4f}) - a pollution/moisture "
           "tan delta rise must clear this band to avoid summer false positives.")

# --- the map: phase shift vs ratio error, leakage = colour + size ------------
st.subheader("The Map - fault signatures (bubble colour & size = leakage mA)")
if _HAS_PLOTLY:
    st.plotly_chart(_scatter_plotly(cpoints, op_cp, envelope),
                    use_container_width=True)
else:
    st.pyplot(_scatter_mpl(cpoints, op_cp, envelope))
st.caption(f"Black star = your live operating point; grey hull = total noise "
           f"floor (T -20..40 degC, f 49.8..50.2 Hz, burden {burden} VA +/-5%, "
           f"+ instrument resolution). Red = internal "
           "faults (big ratio/phase, ~no leakage); leakage lights up the bubble "
           "colour/size; the top-shed stork streamer stays dark near the origin. "
           "A fault is detectable once the star clears the grey box.")

# --- the physics: interactive 3-D phase field --------------------------------
st.subheader("The Physics - 3-D phase-angle field of your run")
if run_state is not None:
    _show_3d(run_state[2], run_state[1])
else:
    st.info("Press **Run FEA Simulation** to render the 3-D phase field here.")
