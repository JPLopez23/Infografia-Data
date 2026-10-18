"""
Laboratorio 09 - Visualizacion Estatica
CC3084 Data Science, UVG, Semestre II 2026

Genera la infografia de una pagina a partir de la Base de Migracion 2009-2026
(INGUAT/IGM) y de los resultados del Laboratorio 1 de Series de Tiempo.

Salidas:
    output/infografia.png  3300 x 5100 px (11 x 17 pulgadas a 300 dpi)
    output/infografia.pdf  11 x 17 pulgadas, una pagina

Uso:
    python infografia.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Base_Migracion_2009-2026jun.xlsx"
OUT = BASE / "infografia"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Tipografia: Source Sans 3 en tres pesos
# ---------------------------------------------------------------------------
for ttf in (BASE / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))
plt.rcParams["font.family"] = "Source Sans 3"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Paleta (ver documento de explicacion)
# ---------------------------------------------------------------------------
PET = "#0E4D5C"      # principal: serie historica, titulares, cifras KPI
TEAL = "#2F8F8B"     # secundario: segunda serie, barras normales
TEAL_L = "#A9D3CF"   # secundario claro: abanico de Estimacion, fondos
GRAY = "#6B7378"     # contexto: ejes, notas
GRAY_L = "#D3D6D8"   # contexto claro: banda pandemia, grilla
ORANGE = "#D9631E"   # acento unico
INK = "#1F2A30"      # texto
BG = "#FAF8F5"       # fondo blanco hueso

W, H = 11.0, 17.0    # pulgadas (lienzo 11 x 17)
ML = 0.75            # margen lateral
CW = W - 2 * ML      # ancho util
SOFT = "#EEF3F2"     # fondo suave para tarjetas y pie (derivado de TEAL_L)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_series():
    df = pd.read_excel(DATA, sheet_name="Datos")
    idx = lambda g: pd.to_datetime([f"{a}-{m:02d}-01" for a, m in g.index])
    total = df.groupby(["Año", "Mes cod"]).Viajero.sum()
    total.index = idx(total)
    comp = df[df["Tipo de Viajero"].isin(["Turista", "Excursionista"])]
    comp = comp.groupby(["Año", "Mes cod"]).Viajero.sum()
    comp.index = idx(comp)
    return total.sort_index(), comp.sort_index()


def indice_estacional(serie):
    """Indice de cada mes respecto del mes promedio, 2009-2019 (base 100)."""
    pre = serie["2009":"2019"]
    tabla = pre.groupby([pre.index.year, pre.index.month]).sum().unstack()
    return (tabla.div(tabla.mean(axis=1), axis=0).mean() * 100)


def pronostico_prophet(serie, n_train=147):
    """Reproduce el Prophet del Laboratorio 1 (corte 70/30, intervalo 80%)."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    train, test = serie[:n_train], serie[n_train:]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="multiplicative",
                interval_width=0.80)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    f = m.predict(pd.DataFrame({"ds": test.index})).set_index("ds")
    mae = np.mean(np.abs(test.values - f.yhat.values))
    rmse = np.sqrt(np.mean((test.values - f.yhat.values) ** 2))
    return train, test, f, mae, rmse


# Resultados del Laboratorio 1 (seccion 5, analisis comparativo, e incisos k)
CUADRANTE = pd.DataFrame({
    "serie": ["Aérea", "Terrestre", "Marítima", "El Salvador", "Honduras", "EE. UU."],
    "grupo": ["Vía", "Vía", "Vía", "País", "País", "País"],
    "estacionalidad": [0.86, 0.76, 0.75, 0.68, 0.57, 0.92],
    "crecimiento": [4.9, 12.2, 5.7, 11.9, 6.7, 4.6],
    "volatilidad": [17.1, 23.5, 117.6, 26.6, 22.2, 31.6],
    "caida": [-72.3, -73.6, -67.9, -74.9, -75.3, -73.7],
})
MODELOS_TOTAL = pd.DataFrame({
    "modelo": ["Prophet", "Suavizamiento exponencial simple", "Holt-Winters",
               "Seasonal naive", "SARIMA(2,1,2)(1,1,1)[12]"],
    "mae": [136571, 175088, 196956, 235731, 249738],
    "rmse": [144939, 194791, 216457, 253203, 270930],
})


# ---------------------------------------------------------------------------
# Utilidades de maquetacion (coordenadas en pulgadas desde arriba a la izquierda)
# ---------------------------------------------------------------------------
import textwrap


def txt(fig, x, y, s, size=11, weight="regular", color=INK, **kw):
    return fig.text(x / W, 1 - y / H, s, fontsize=size, fontweight=weight,
                    color=color, va=kw.pop("va", "top"), **kw)


def ejes(fig, x, y, w, h):
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=GRAY, labelsize=9.5, length=0)
    return ax


def rect(fig, x, y, w, h, fc, ec=None, lw=0, r=0.0):
    p = FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       transform=fig.transFigure, fc=fc, ec=ec or fc, lw=lw, zorder=0)
    fig.patches.append(p)


def encabezado(fig, y, num, etiqueta, titular):
    """Encabezado uniforme de cada bloque: numero + etiqueta corta + titular hallazgo."""
    txt(fig, ML, y, f"{num}  ·  {etiqueta}", size=10.5, weight="semibold", color=TEAL)
    txt(fig, ML, y + 0.24, titular, size=19, weight="semibold", color=PET)


def miles(v, _=None):
    return "0" if v == 0 else f"{v/1000:,.0f} mil"


# ---------------------------------------------------------------------------
# Bloques
# ---------------------------------------------------------------------------
def bloque_titular(fig, total, comp, idx):
    rect(fig, 0, 0, W, 3.05, fc=PET)
    txt(fig, ML, 0.5, "TURISMO INTERNACIONAL EN GUATEMALA  ·  2009–2026",
        size=11, weight="semibold", color=TEAL_L)
    txt(fig, ML, 0.76,
        "Guatemala recibe 8 de cada 10 visitantes\nque tenía antes de la pandemia,\ny diciembre sigue mandando",
        size=32, weight="bold", color="white", linespacing=0.98)
    txt(fig, ML, 2.6, "Radiografía de 17 años de ingresos internacionales y lo que los modelos anticipan",
        size=14, color=TEAL_L)

    anual = comp.groupby(comp.index.year).sum()
    anual_t = total.groupby(total.index.year).sum()
    kpis = [
        (f"{total.sum()/1e6:.1f} M", "ingresos internacionales\nregistrados, 2009–2026", False),
        (f"−{(1-anual_t[2020]/anual_t[2019])*100:.0f}%", "caída en 2020 frente a 2019\npor la pandemia", True),
        (f"{anual[2025]/anual[2019]*100:.0f}%", "del nivel de 2019 recuperado\nen 2025", False),
        (f"+{idx[12]-100:.0f}%", "visitantes en diciembre\nsobre un mes promedio", False),
    ]
    cw = CW / 4
    y0 = 3.35
    for i, (num, lab, acento) in enumerate(kpis):
        x = ML + i * cw
        if i > 0:
            fig.add_artist(plt.Line2D([x / W] * 2, [1 - (y0 + 0.05) / H, 1 - (y0 + 1.1) / H], color=GRAY_L, lw=1))
        xi = x + (0.0 if i == 0 else 0.25)
        txt(fig, xi, y0, num, size=42, weight="bold", color=ORANGE if acento else PET)
        txt(fig, xi, y0 + 0.7, lab, size=10.5, color=INK, linespacing=1.2)


def bloque_panorama(fig, total, comp):
    y = 4.75
    encabezado(fig, y, "01", "PANORAMA 2009–2026",
               "Diez años de crecimiento, un colapso y una recuperación parcial")
    ch, ymax = 1.85, 600000
    cx, cwid = ML + 0.55, CW - 0.55 - 1.3
    cy = y + 0.8
    ax = ejes(fig, cx, cy, cwid, ch)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRAY_L, alpha=0.5, lw=0)
    ax.plot(comp.index, comp.values, color=TEAL_L, lw=1.5, zorder=2)
    ax.plot(total.index, total.values, color=PET, lw=1.5, zorder=3)
    corte = pd.Timestamp("2023-01-01")
    ax.axvline(corte, color=GRAY, lw=0.9, ls=(0, (3, 3)))
    ax.set_ylim(0, ymax)
    ax.set_xlim(pd.Timestamp("2009-01-01"), pd.Timestamp("2026-06-30"))
    ax.yaxis.set_major_formatter(FuncFormatter(miles))
    ax.set_yticks([0, 200000, 400000, 600000])
    ax.grid(axis="y", color=GRAY_L, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in range(2010, 2027, 2)])
    ax.set_xticklabels([str(a) for a in range(2010, 2027, 2)])

    nota = dict(fontsize=9.5, color=INK, linespacing=1.15, va="top")
    linea = dict(arrowstyle="-", color=GRAY, lw=0.7)
    ax.annotate("2019: récord anual,\n4.69 millones", xy=(pd.Timestamp("2019-12-01"), total["2019-12-01"]),
                xytext=(pd.Timestamp("2016-03-01"), 590000), arrowprops=linea, **nota)
    ax.annotate("Mayo 2020: mínimo\nde 9,779", xy=(pd.Timestamp("2020-05-01"), 9779),
                xytext=(pd.Timestamp("2016-09-01"), 110000), arrowprops=linea, **nota)
    ax.text(pd.Timestamp("2020-12-15"), 590000, "Pandemia", ha="center", va="top", fontsize=9.5, color=GRAY)
    ax.text(corte + pd.Timedelta(days=50), 590000, "2023: cambio de\nmetodología",
            fontsize=9.5, color=GRAY, linespacing=1.15, va="top")

    yax = lambda v: cy + ch * (1 - v / ymax)
    xe = cx + cwid + 0.1
    txt(fig, xe, yax(total.iloc[-1]) - 0.3, "Total", size=10.5, weight="semibold", color=PET)
    txt(fig, xe, yax(comp.iloc[-1]) + 0.04, "Turistas y\nexcursionistas", size=10.5, weight="semibold",
        color=TEAL, linespacing=1.05)


def bloque_estacionalidad(fig, idx):
    y = 7.9
    encabezado(fig, y, "02", "ESTACIONALIDAD, VÍAS Y MERCADOS",
               "Diciembre concentra un 36% más de visitantes que un mes promedio")

    # calendario mensual
    cy = y + 0.8
    txt(fig, ML, cy, "Índice mensual 2009–2019 (100 = mes promedio)", size=10, color=GRAY)
    ax = ejes(fig, ML, cy + 0.3, 4.35, 1.85)
    meses = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    col = [ORANGE if m == 12 else (GRAY if m == 9 else TEAL_L) for m in range(1, 13)]
    ax.bar(range(12), idx.values, color=col, width=0.7, zorder=2)
    ax.axhline(100, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=3)
    for i in (2, 6, 8, 11):
        v = idx.values[i]
        ax.text(i, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                color=ORANGE if i == 11 else INK, fontweight="semibold")
    ax.set_xticks(range(12))
    ax.set_xticklabels(meses, fontsize=10, color=INK)
    ax.set_yticks([])
    ax.set_ylim(0, 205)
    ax.set_xlim(-0.6, 11.6)
    for xx, t in [(2.5, "Semana Santa"), (6.5, "Vacaciones\ny Bono 14"), (8, "Valle"), (11, "Aguinaldo\ny fiestas")]:
        ax.text(xx, 203, t, ha="center", va="top", fontsize=9, color=GRAY, linespacing=1.05)

    # cuadrante: cuatro tarjetas
    x0 = ML + 4.75
    txt(fig, x0, cy, "Ganadoras por pregunta, según el Laboratorio 1", size=10, color=GRAY)
    tarjetas = [
        ("Mayor estacionalidad", ("Aérea", "0.86"), ("EE. UU.", "0.92"), "fuerza estacional, 0 a 1"),
        ("Mayor crecimiento", ("Terrestre", "+12.2%"), ("El Salvador", "+11.9%"), "anual, 2009–2019"),
        ("Mayor volatilidad", ("Marítima", "118%"), ("EE. UU.", "32%"), "cambio mensual típico"),
        ("Más golpeada en 2020", ("Terrestre", "−73.6%"), ("Honduras", "−75.3%"), "caída frente a 2019"),
    ]
    tw, th, g = (CW - 4.75 - 0.2) / 2, 0.92, 0.18
    for k, (q, via, pais, unidad) in enumerate(tarjetas):
        tx = x0 + (k % 2) * (tw + g)
        ty = cy + 0.3 + (k // 2) * (th + g)
        rect(fig, tx, ty, tw, th, fc=SOFT, r=0.008)
        txt(fig, tx + 0.15, ty + 0.1, q, size=10.5, weight="semibold", color=PET)
        txt(fig, tx + 0.15, ty + 0.28, unidad, size=9, color=GRAY)
        for j, (grupo, (nom, val)) in enumerate([("Vía", via), ("País", pais)]):
            yy = ty + 0.5 + j * 0.2
            txt(fig, tx + 0.15, yy, grupo, size=9, color=GRAY)
            txt(fig, tx + 0.55, yy - 0.01, nom, size=10.5, weight="semibold", color=INK)
            txt(fig, tx + tw - 0.15, yy - 0.02, val, size=11, weight="bold", color=ORANGE, ha="right")


def bloque_modelos(fig, train, test, f, mae, rmse):
    y = 11.35
    encabezado(fig, y, "03", "MODELOS Y PRONÓSTICO",
               "Prophet capta el calendario, pero nadie anticipó la reapertura")
    cy = y + 0.8
    # desempeno
    txt(fig, ML, cy, "Error promedio mensual (MAE), serie total", size=10, color=GRAY)
    ax = ejes(fig, ML + 1.25, cy + 0.3, 1.95, 1.4)
    d = MODELOS_TOTAL.iloc[::-1].reset_index(drop=True)
    ax.barh(range(5), d.mae, color=[ORANGE if m == "Prophet" else GRAY_L for m in d.modelo], height=0.6)
    for i, (mm, v) in enumerate(zip(d.modelo, d.mae)):
        ax.text(v + 6000, i, f"{v/1000:.0f} mil", va="center", fontsize=9.5, color=INK,
                fontweight="semibold" if mm == "Prophet" else "regular")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["SARIMA", "Seasonal naive", "Holt-Winters", "Suaviz. simple", "Prophet"],
                       fontsize=10, color=INK)
    ax.set_xlim(0, 330000)
    ax.set_xticks([])
    txt(fig, ML, cy + 1.9, "Prophet ganó en 4 de las 7 series analizadas.", size=9, color=GRAY)

    # pronostico
    x0 = ML + 3.75
    txt(fig, x0, cy, "Real frente a Prophet en la prueba, abr 2021–jun 2026", size=10, color=GRAY)
    cx, cwid, ch, ymax = x0 + 0.5, CW - 3.75 - 0.5 - 1.05, 1.4, 600000
    ax = ejes(fig, cx, cy + 0.3, cwid, ch)
    hist = train["2019-01-01":]
    ax.plot(hist.index, hist.values, color=GRAY_L, lw=1.3)
    ax.fill_between(f.index, f.yhat_lower.clip(lower=0), f.yhat_upper, color=TEAL_L, alpha=0.6, lw=0)
    ax.plot(test.index, test.values, color=PET, lw=1.6)
    ax.plot(f.index, f.yhat, color=ORANGE, lw=1.7, ls=(0, (4, 2)))
    ax.axvline(test.index[0], color=GRAY, lw=0.7, ls=(0, (2, 2)))
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 200000, 400000, 600000])
    ax.yaxis.set_major_formatter(FuncFormatter(miles))
    ax.grid(axis="y", color=GRAY_L, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2026-06-30"))
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in range(2019, 2027, 1)])
    ax.set_xticklabels([f"{a}" if a % 2 == 1 else "" for a in range(2019, 2027)])
    ax.text(test.index[0] + pd.Timedelta(days=30), 585000, "Prueba", va="top", fontsize=9, color=GRAY)
    yax = lambda v: cy + 0.3 + ch * (1 - v / ymax)
    xe = cx + cwid + 0.1
    txt(fig, xe, yax(test.iloc[-1]) - 0.2, "Real", size=10.5, weight="semibold", color=PET)
    txt(fig, xe, yax(f.yhat.iloc[-1]) - 0.2, "Estimación", size=10.5, weight="semibold", color=ORANGE)
    txt(fig, xe, yax(f.yhat.iloc[-1]) + 0.0, "rango 80%", size=9, color=GRAY)
    txt(fig, x0, cy + 1.95,
        f"Entrenado hasta mar 2021, el punto más bajo: subestima el nivel. MAE {mae/1000:.0f} mil, RMSE {rmse/1000:.0f} mil.",
        size=9, color=GRAY)


def bloque_recomendaciones(fig):
    y = 14.35
    encabezado(fig, y, "04", "RECOMENDACIONES PARA INGUAT", "Cinco decisiones que salen de la evidencia")
    recs = [
        ("Planificar por calendario", "Reforzar fronteras y campañas antes de diciembre y julio; promover septiembre."),
        ("Separar vías", "Metas propias: la terrestre crecía 12.2% al año y la aérea 4.9%."),
        ("EE. UU. en julio", "Mercado más estacional, ya en 146% de 2019; promover antes de julio."),
        ("Pausar cruceros", "Registro marítimo en 5.3% de 2019; corregirlo antes de fijar metas."),
        ("Medir mejor", "Usar turistas y excursionistas y reentrenar los modelos cada mes."),
    ]
    g = 0.18
    cw = (CW - 4 * g) / 5
    cy = y + 0.8
    for i, (lead, body) in enumerate(recs):
        x = ML + i * (cw + g)
        rect(fig, x, cy, cw, 1.25, fc=SOFT, r=0.008)
        txt(fig, x + 0.14, cy + 0.06, f"{i+1}", size=20, weight="bold", color=ORANGE)
        txt(fig, x + 0.14, cy + 0.42, lead, size=10.5, weight="semibold", color=PET)
        txt(fig, x + 0.14, cy + 0.62, textwrap.fill(body, 30), size=9, color=INK, linespacing=1.2)


def pie(fig):
    rect(fig, 0, 16.5, W, 0.5, fc=SOFT)
    txt(fig, ML, 16.6,
        "Fuente: Base de Migración 2009–2026, INGUAT/IGM (ene 2009–jun 2026, 161,036 registros). "
        "Modelos: 70% entrenamiento, 30% prueba. Uso académico; no son cifras oficiales.",
        size=9, color=GRAY)
    txt(fig, ML, 16.78,
        "José López (23773) · Hugo Barillas (23306) · Luis Palacios (23933)  —  CC3084 Data Science, "
        "Universidad del Valle de Guatemala, Semestre II 2026",
        size=9, color=INK)


def main():
    total, comp = cargar_series()
    idx = indice_estacional(total)
    train, test, f, mae, rmse = pronostico_prophet(total)
    print(f"Prophet total: MAE {mae:,.0f}  RMSE {rmse:,.0f}")

    fig = plt.figure(figsize=(W, H), facecolor=BG)
    bloque_titular(fig, total, comp, idx)
    bloque_panorama(fig, total, comp)
    bloque_estacionalidad(fig, idx)
    bloque_modelos(fig, train, test, f, mae, rmse)
    bloque_recomendaciones(fig)
    pie(fig)

    fig.savefig(OUT / "infografia.png", dpi=300, facecolor=BG)
    fig.savefig(OUT / "infografia.pdf", facecolor=BG)
    print("Listo:", OUT / "infografia.png", OUT / "infografia.pdf")


if __name__ == "__main__":
    main()
