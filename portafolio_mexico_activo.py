"""
================================================================
 ESTRATEGIA ACTIVA – PORTAFOLIO 10 ACCIONES MEXICANAS
 Benchmark  : MSCI Mexico Top 40 (proxy ETF EWW)
 Rebalanceo : Mensual
 Regla      : Reponderacion proporcional al Alpha CAPM individual.
              Las 10 acciones se mantienen siempre en el portafolio.
              Cada accion recibe un peso minimo garantizado (floor).
              El capital restante se distribuye entre las acciones
              con alpha > 0, en proporcion a su alpha positivo.
              Acciones con alpha < 0 solo conservan el floor.
 Alpha      : Calculado correctamente sobre excesos de retorno
              respecto a la tasa libre de riesgo (Jensen 1968).
              (R_i - Rf) = alpha + beta*(R_bm - Rf) + eps
 Backtest   : 2024 completo (out-of-sample respecto al YTD actual)
 Metricas   : Alpha Jensen, Sharpe, Drawdown maximo, Tracking Error,
              Information Ratio – tanto en backtest como en YTD.

 Fuentes metodologicas:
   - Jensen (1968), J. of Finance 23(2): alpha CAPM
   - Sharpe (1964), Lintner (1965): CAPM base
   - Grinold & Kahn (2000), Active Portfolio Management: IR y TE
================================================================
Requisitos:
    pip install yfinance pandas numpy scipy tabulate matplotlib python-dateutil
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from datetime import date
from dateutil.relativedelta import relativedelta
import sys

# ════════════════════════════════════════════════════════════
# 1. CONFIGURACION
# ════════════════════════════════════════════════════════════

UNIVERSO = {
    "AMXB.MX":     "America Movil",
    "WALMEX.MX":   "Walmart MX",
    "FEMSAUBD.MX": "FEMSA",
    "GFNORTEO.MX": "GFNorte",
    "GMEXICOB.MX": "Grupo Mexico",
    "BIMBOA.MX":   "Grupo Bimbo",
    "CEMEXCPO.MX": "CEMEX",
    "ALSEA.MX":    "Alsea",
    "KIMBERA.MX":  "Kimberly-Clark MX",
    "MEGACPO.MX":  "Megacable",
}

BENCHMARK_TICKER = "EWW"
CAPITAL_INICIAL  = 1_000_000     # MXN
COMISION         = 0.0025        # 0.25 % por operacion (compra o venta)
IVA              = 0.16
TASA_RF_ANUAL    = 0.105         # CETES referencia ~10.5 % anual
RF_DIARIO        = (1 + TASA_RF_ANUAL) ** (1 / 252) - 1
VENTANA_ALPHA    = 21*3           # dias habiles (~1 mes) para la regresion
PESO_MINIMO      = 0.08        # floor garantizado por accion

# Periodos
HOY            = date.today()
INICIO_YTD     = date(HOY.year, 1, 1)
DIAS_YTD       = (HOY - INICIO_YTD).days
RF_YTD         = (1 + TASA_RF_ANUAL) ** (DIAS_YTD / 365) - 1

INICIO_BT      = date(2024, 1, 1)
FIN_BT         = date(2024, 12, 31)
DIAS_BT        = (FIN_BT - INICIO_BT).days
RF_BT          = (1 + TASA_RF_ANUAL) ** (DIAS_BT / 365) - 1

TICKERS = list(UNIVERSO.keys())
N       = len(TICKERS)

print("=" * 68)
print("  PORTAFOLIO ACTIVO MEXICANO – REPONDERACION MENSUAL POR ALPHA CAPM")
print(f"  Backtest   : {INICIO_BT}  ->  {FIN_BT}")
print(f"  YTD actual : {INICIO_YTD}  ->  {HOY}  ({DIAS_YTD} dias)")
print(f"  Acciones   : {N} (fijas)  |  Floor: {PESO_MINIMO*100:.0f}%  |  Ventana alpha: {VENTANA_ALPHA}d")
print("=" * 68)

# ════════════════════════════════════════════════════════════
# 2. DESCARGA DE DATOS
# ════════════════════════════════════════════════════════════

# Descargamos desde 3 meses antes del inicio del backtest para
# tener ventana de calentamiento disponible desde el primer rebalanceo.
INICIO_DATOS = INICIO_BT - relativedelta(months=3)
tickers_all  = TICKERS + [BENCHMARK_TICKER]

print("\nDescargando datos de Yahoo Finance...")
raw = yf.download(
    tickers_all,
    start=str(INICIO_DATOS),
    end=str(HOY),
    auto_adjust=True,   # incorpora dividendos y splits
    progress=False,
)

if raw.empty:
    sys.exit("ERROR: No se obtuvieron datos. Verifica tu conexion a internet.")

# Extraer Close y normalizar MultiIndex que devuelve yfinance
close_raw = raw["Close"] if "Close" in raw.columns else raw
if isinstance(close_raw.columns, pd.MultiIndex):
    close_raw = close_raw.droplevel(1, axis=1)
precios_full = close_raw.ffill().dropna(how="all")

# Detectar tickers que fallaron en la descarga batch y reintentar uno por uno
tickers_batch_fallidos = [
    t for t in tickers_all
    if t not in precios_full.columns or precios_full[t].notna().sum() <= 5
]
if tickers_batch_fallidos:
    print(f"\nReintenando descarga individual para {len(tickers_batch_fallidos)} ticker(s)...")
    for t in tickers_batch_fallidos:
        try:
            df_t = yf.download(t, start=str(INICIO_DATOS), end=str(HOY),
                               auto_adjust=True, progress=False)
            if df_t.empty:
                print(f"  {t}: sin datos")
                continue
            col = df_t["Close"] if "Close" in df_t.columns else df_t.iloc[:, 0]
            if isinstance(col, pd.DataFrame):
                col = col.squeeze()
            col = col.ffill().dropna()
            if len(col) > 5:
                precios_full[t] = col
                print(f"  {t}: OK  ({len(col)} sesiones recuperadas)")
            else:
                print(f"  {t}: insuficientes datos ({len(col)} sesiones)")
        except Exception as e:
            print(f"  {t}: error al descargar – {e}")

# Validacion final
aun_fallidos = [
    t for t in TICKERS
    if t not in precios_full.columns or precios_full[t].notna().sum() <= 5
]
if aun_fallidos:
    print("\nERROR – Sin datos suficientes para:")
    for t in aun_fallidos:
        print(f"  {t}  ({UNIVERSO.get(t, '?')})")
    print("Verifica los simbolos en finance.yahoo.com y vuelve a intentar.")
    sys.exit(1)

bm_full  = precios_full[[BENCHMARK_TICKER]]
acc_full = precios_full[TICKERS]

# Subsets por periodo
acc_bt  = acc_full[(acc_full.index >= str(INICIO_BT))  & (acc_full.index <= str(FIN_BT))]
bm_bt   = bm_full[ (bm_full.index  >= str(INICIO_BT))  & (bm_full.index  <= str(FIN_BT))]
acc_ytd = acc_full[acc_full.index  >= str(INICIO_YTD)]
bm_ytd  = bm_full[ bm_full.index   >= str(INICIO_YTD)]

print(f"Sesiones backtest (2024) : {len(acc_bt)}")
print(f"Sesiones YTD {HOY.year}        : {len(acc_ytd)}")

# ════════════════════════════════════════════════════════════
# 3. FUNCIONES CENTRALES
# ════════════════════════════════════════════════════════════

def calcular_alphas(fecha_eval, ventana=VENTANA_ALPHA):
    """
    Regresion CAPM sobre excesos de retorno (Jensen 1968):
        (R_i - Rf) = alpha + beta * (R_bm - Rf) + eps

    Usa los `ventana` dias habiles ANTERIORES a fecha_eval
    (sin incluirla), garantizando que no hay look-ahead bias.
    Retorna dict {ticker: alpha_diario}.
    """
    idx_all = precios_full.index
    pos = idx_all.searchsorted(str(fecha_eval))
    if pos < ventana + 1:
        return {t: 0.0 for t in TICKERS}

    ventana_idx = idx_all[pos - ventana: pos]
    ret_bm_raw  = bm_full.loc[ventana_idx, BENCHMARK_TICKER].pct_change().dropna()
    exc_bm      = ret_bm_raw - RF_DIARIO

    alphas = {}
    for t in TICKERS:
        ret_acc_raw = acc_full.loc[ventana_idx, t].pct_change().dropna()
        comun       = exc_bm.index.intersection(ret_acc_raw.index)
        if len(comun) < 8:
            alphas[t] = 0.0
            continue
        exc_acc = ret_acc_raw[comun] - RF_DIARIO
        _, intercepto, *_ = stats.linregress(exc_bm[comun], exc_acc)
        alphas[t] = intercepto   # alpha diario sobre exceso de retorno
    return alphas


def pesos_por_alpha(alphas):
    """
    1. Floor PESO_MINIMO garantizado a todas las acciones.
    2. El sobrante (1 - N*PESO_MINIMO) va a las acciones con
       alpha > 0, ponderado por su alpha.
    3. Fallback: si ninguna tiene alpha > 0, sobrante uniforme.
    """
    pesos    = {t: PESO_MINIMO for t in TICKERS}
    sobrante = 1.0 - N * PESO_MINIMO
    pos      = {t: alphas[t] for t in TICKERS if alphas[t] > 0}

    if pos:
        total = sum(pos.values())
        for t, a in pos.items():
            pesos[t] += sobrante * (a / total)
    else:
        for t in TICKERS:
            pesos[t] += sobrante / N

    assert abs(sum(pesos.values()) - 1.0) < 1e-9
    return pesos


def costo_tx(monto):
    return abs(monto) * COMISION * (1 + IVA)


def primer_dia_habil_mes(year, month, sesiones):
    candidatos = [d for d in sesiones if d.year == year and d.month == month]
    return candidatos[0] if candidatos else None


# ════════════════════════════════════════════════════════════
# 4. MOTOR DE SIMULACION (reutilizable para backtest y YTD)
# ════════════════════════════════════════════════════════════

def simular(acc_prices, bm_prices, rf_periodo, label=""):
    """
    Corre la estrategia de reponderacion mensual sobre acc_prices.
    Devuelve:
      serie_port   : valor diario del portafolio activo
      serie_bh     : valor diario del buy-and-hold
      fechas_rb    : conjunto de fechas de rebalanceo ejecutadas
      log_rb       : lista de dicts con detalle de cada rebalanceo
      hist_pesos   : {fecha: {ticker: peso}} en cada rebalanceo
      tenencias_fin: {ticker: valor_MXN} al cierre del periodo
    """
    sesiones = acc_prices.index.tolist()
    meses    = sorted({(d.year, d.month) for d in sesiones})

    fechas_rb = set()
    for y, m in meses[1:]:   # saltar el primer mes (ya se compra al inicio)
        fd = primer_dia_habil_mes(y, m, sesiones)
        if fd:
            fechas_rb.add(fd)

    # Estado inicial
    costo_entrada = CAPITAL_INICIAL * COMISION * (1 + IVA)
    capital_neto  = CAPITAL_INICIAL - costo_entrada
    tenencias     = {t: capital_neto / N for t in TICKERS}

    log_rb      = []
    log_diario  = []
    hist_pesos  = {}
    prev_prices = {t: acc_prices[t].iloc[0] for t in TICKERS}

    for fecha in sesiones:
        cur_prices = {
            t: acc_prices.loc[fecha, t]
            for t in TICKERS
            if not pd.isna(acc_prices.loc[fecha, t])
        }

        # Actualizar valor por retorno diario
        for t in TICKERS:
            if t in cur_prices and t in prev_prices and prev_prices[t] > 0:
                tenencias[t] *= (1 + (cur_prices[t] / prev_prices[t]) - 1)

        valor_total = sum(tenencias.values())
        log_diario.append({"fecha": fecha, "valor": valor_total})

        # Rebalanceo mensual
        if fecha in fechas_rb:
            alphas      = calcular_alphas(fecha)
            pesos_nuevo = pesos_por_alpha(alphas)
            pesos_real  = {t: tenencias[t] / valor_total for t in TICKERS}

            for t in TICKERS:
                diff  = (pesos_nuevo[t] - pesos_real[t]) * valor_total
                costo = costo_tx(diff)
                tenencias[t] += diff - (costo if diff >= 0 else -costo)

                log_rb.append({
                    "Fecha":         str(fecha.date()),
                    "Ticker":        t.replace(".MX", ""),
                    "Nombre":        UNIVERSO[t],
                    "Alpha (pb/d)":  f"{alphas[t]*10000:+.2f}",
                    "Peso ant. %":   f"{pesos_real[t]*100:.1f}",
                    "Peso nuevo %":  f"{pesos_nuevo[t]*100:.1f}",
                    "Cambio (pp)":   f"{(pesos_nuevo[t]-pesos_real[t])*100:+.1f}",
                })

            hist_pesos[fecha] = dict(pesos_nuevo)

        prev_prices = cur_prices

    # Serie del portafolio activo
    df_log = pd.DataFrame(log_diario).set_index("fecha")
    df_log.index = pd.to_datetime(df_log.index)
    serie_port = df_log["valor"]

    # Buy-and-hold con pesos iguales
    ret_bh   = acc_prices.pct_change().dropna().dot(np.full(N, 1.0 / N))
    serie_bh = (1 + ret_bh).cumprod() * capital_neto

    return serie_port, serie_bh, fechas_rb, log_rb, hist_pesos, tenencias


# ════════════════════════════════════════════════════════════
# 5. FUNCION DE METRICAS
# ════════════════════════════════════════════════════════════

def calcular_metricas(serie_port, serie_bh, bm_prices, rf_periodo, dias):
    """
    Calcula todas las metricas de desempeno para un periodo dado.
    """
    ret_port = (serie_port.iloc[-1] / serie_port.iloc[0]) - 1
    ret_bh   = (serie_bh.iloc[-1]   / serie_bh.iloc[0])   - 1
    ret_bm   = (bm_prices.iloc[-1, 0] / bm_prices.iloc[0, 0]) - 1

    rp_d  = serie_port.pct_change().dropna()
    rb_d  = bm_prices.iloc[:, 0].pct_change().dropna()
    comun = rp_d.index.intersection(rb_d.index)
    rp, rb = rp_d[comun], rb_d[comun]

    # Excesos diarios sobre Rf (para regresion CAPM del portafolio completo)
    exc_port = rp - RF_DIARIO
    exc_bm   = rb - RF_DIARIO

    slope, intercept, r_val, *_ = stats.linregress(exc_bm, exc_port)
    beta         = slope
    alpha_jensen = rf_periodo + intercept * dias - rf_periodo
    # Forma directa: alpha_jensen = ret_port - [Rf + beta*(ret_bm - Rf)]
    alpha_jensen = ret_port - (rf_periodo + beta * (ret_bm - rf_periodo))

    vol      = rp.std() * np.sqrt(252)
    te       = (rp - rb).std() * np.sqrt(252)
    factor_t = np.sqrt(dias / 365)
    sharpe   = (ret_port - rf_periodo) / (vol * factor_t) if vol else np.nan
    ir       = (ret_port - ret_bm) / (te * factor_t) if te else np.nan

    acum    = (1 + rp).cumprod()
    max_dd  = ((acum - acum.cummax()) / acum.cummax()).min()

    return {
        "ret_port":    ret_port,
        "ret_bh":      ret_bh,
        "ret_bm":      ret_bm,
        "beta":        beta,
        "alpha_jensen":alpha_jensen,
        "r2":          r_val ** 2,
        "vol":         vol,
        "te":          te,
        "sharpe":      sharpe,
        "ir":          ir,
        "max_dd":      max_dd,
        "rp":          rp,
        "rb":          rb,
    }


# ════════════════════════════════════════════════════════════
# 6. EJECUTAR BACKTEST (2024) Y YTD
# ════════════════════════════════════════════════════════════

print("\nEjecutando backtest 2024...")
sp_bt, sbh_bt, frb_bt, log_bt, hp_bt, ten_bt = simular(
    acc_bt, bm_bt, RF_BT, label="BT"
)
m_bt = calcular_metricas(sp_bt, sbh_bt, bm_bt, RF_BT, DIAS_BT)

print(f"Ejecutando analisis YTD {HOY.year}...")
sp_ytd, sbh_ytd, frb_ytd, log_ytd, hp_ytd, ten_ytd = simular(
    acc_ytd, bm_ytd, RF_YTD, label="YTD"
)
m_ytd = calcular_metricas(sp_ytd, sbh_ytd, bm_ytd, RF_YTD, DIAS_YTD)


# ════════════════════════════════════════════════════════════
# 7. IMPRESION DE RESULTADOS
# ════════════════════════════════════════════════════════════

SEP = "─" * 68

def imprimir_log_rebalanceos(log, fechas_rb, titulo):
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(SEP)
    if not log:
        print("  (Sin rebalanceos en el periodo)")
        return
    df = pd.DataFrame(log)
    for fr in sorted(fechas_rb):
        bloque = df[df["Fecha"] == str(fr.date())]
        if not bloque.empty:
            print(f"\n  Rebalanceo: {fr.date()}")
            print(tabulate(
                bloque.drop(columns="Fecha"),
                headers="keys",
                tablefmt="rounded_outline",
                showindex=False,
            ))

def imprimir_pesos_finales(tenencias, titulo):
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(SEP)
    vt = sum(tenencias.values())
    tabla = [
        [t.replace(".MX",""), UNIVERSO[t],
         f"${tenencias[t]:,.0f}", f"{tenencias[t]/vt*100:.1f}%"]
        for t in TICKERS
    ]
    print(tabulate(tabla,
                   headers=["Ticker","Nombre","Valor MXN","Peso"],
                   tablefmt="rounded_outline"))

def imprimir_metricas(m, titulo, periodo_str, dias):
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(f"  Periodo: {periodo_str}  ({dias} dias)")
    print(SEP)
    tabla = [
        ["Retorno portafolio activo",   f"{m['ret_port']*100:+.2f}%"],
        ["Retorno buy-and-hold",        f"{m['ret_bh']*100:+.2f}%"],
        ["Retorno benchmark (EWW)",     f"{m['ret_bm']*100:+.2f}%"],
        ["Exceso vs benchmark",         f"{(m['ret_port']-m['ret_bm'])*100:+.2f} pp"],
        ["Exceso vs buy-and-hold",      f"{(m['ret_port']-m['ret_bh'])*100:+.2f} pp"],
        ["", ""],
        ["Beta",                        f"{m['beta']:.3f}"],
        ["Alpha Jensen (CAPM)",         f"{m['alpha_jensen']*100:+.2f}%"],
        ["R2 vs benchmark",             f"{m['r2']:.3f}"],
        ["", ""],
        ["Volatilidad anualizada",      f"{m['vol']*100:.2f}%"],
        ["Tracking error anualizado",   f"{m['te']*100:.2f}%"],
        ["Sharpe ratio",                f"{m['sharpe']:.3f}"],
        ["Information ratio",           f"{m['ir']:.3f}" if not np.isnan(m['ir']) else "N/D"],
        ["Maximo drawdown",             f"{m['max_dd']*100:.2f}%"],
    ]
    print(tabulate(tabla, headers=["Metrica","Valor"], tablefmt="rounded_outline"))

    dir_bm = "SUPERA" if m['alpha_jensen'] > 0 else "QUEDA POR DEBAJO DE"
    print(f"\n  Alpha Jensen = {m['alpha_jensen']*100:+.2f}%")
    print(f"  El portafolio {dir_bm} al benchmark ajustado por riesgo (beta={m['beta']:.2f}).")


# -- Logs de rebalanceo
imprimir_log_rebalanceos(log_bt,  frb_bt,  "LOG REBALANCEOS – BACKTEST 2024")
imprimir_log_rebalanceos(log_ytd, frb_ytd, f"LOG REBALANCEOS – YTD {HOY.year}")

# -- Pesos finales
imprimir_pesos_finales(ten_bt,  "PESOS FINALES – BACKTEST 2024")
imprimir_pesos_finales(ten_ytd, f"PESOS FINALES – YTD {HOY.year}")

# -- Metricas
imprimir_metricas(m_bt,  "METRICAS – BACKTEST 2024",
                  f"{INICIO_BT} / {FIN_BT}", DIAS_BT)
imprimir_metricas(m_ytd, f"METRICAS – YTD {HOY.year}",
                  f"{INICIO_YTD} / {HOY}", DIAS_YTD)

# -- Tabla comparativa rapida
print(f"\n{SEP}")
print("  COMPARACION RAPIDA: BACKTEST 2024  vs  YTD ACTUAL")
print(SEP)
comp = [
    ["Retorno portafolio activo",
     f"{m_bt['ret_port']*100:+.2f}%", f"{m_ytd['ret_port']*100:+.2f}%"],
    ["Retorno buy-and-hold",
     f"{m_bt['ret_bh']*100:+.2f}%",  f"{m_ytd['ret_bh']*100:+.2f}%"],
    ["Retorno benchmark EWW",
     f"{m_bt['ret_bm']*100:+.2f}%",  f"{m_ytd['ret_bm']*100:+.2f}%"],
    ["Alpha Jensen",
     f"{m_bt['alpha_jensen']*100:+.2f}%", f"{m_ytd['alpha_jensen']*100:+.2f}%"],
    ["Beta",
     f"{m_bt['beta']:.3f}", f"{m_ytd['beta']:.3f}"],
    ["Sharpe ratio",
     f"{m_bt['sharpe']:.3f}", f"{m_ytd['sharpe']:.3f}"],
    ["Maximo drawdown",
     f"{m_bt['max_dd']*100:.2f}%", f"{m_ytd['max_dd']*100:.2f}%"],
    ["Tracking error",
     f"{m_bt['te']*100:.2f}%", f"{m_ytd['te']*100:.2f}%"],
    ["Information ratio",
     f"{m_bt['ir']:.3f}" if not np.isnan(m_bt['ir']) else "N/D",
     f"{m_ytd['ir']:.3f}" if not np.isnan(m_ytd['ir']) else "N/D"],
]
print(tabulate(comp,
               headers=["Metrica", "Backtest 2024", f"YTD {HOY.year}"],
               tablefmt="rounded_outline"))


# ════════════════════════════════════════════════════════════
# 8. GRAFICAS
# ════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(17, 11))
fig.patch.set_facecolor("#0d1117")
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

C_ACTIVO = "#58a6ff"
C_BH     = "#3fb950"
C_BM     = "#f78166"
C_GRAY   = "#8b949e"
C_GOLD   = "#e3b341"

def estilo_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#f0f6fc")

def marcar_rebalanceos(ax, fechas, serie_idx):
    primero = True
    for fr in sorted(fechas):
        if fr in serie_idx:
            ax.axvline(fr, color=C_GOLD, lw=0.8, alpha=0.5,
                       label="Rebalanceo" if primero else "")
            primero = False

leyenda_base = [
    Line2D([0],[0], color=C_ACTIVO, lw=2.0, label="Activo"),
    Line2D([0],[0], color=C_BH,     lw=1.5, ls="--", label="Buy&Hold"),
    Line2D([0],[0], color=C_BM,     lw=1.5, ls=":",  label="EWW"),
    Line2D([0],[0], color=C_GOLD,   lw=0.8, alpha=0.7, label="Rebalanceo"),
]

# ── (0,0)+(0,1): Retorno acumulado backtest 2024 ─────────
ax_bt = fig.add_subplot(gs[0, :2])
estilo_ax(ax_bt)
n_bt_p = sp_bt  / sp_bt.iloc[0]
n_bt_b = sbh_bt / sbh_bt.iloc[0]
n_bt_m = bm_bt.iloc[:,0] / bm_bt.iloc[0,0]
ax_bt.plot(n_bt_p.index, n_bt_p.values, color=C_ACTIVO, lw=2.0,
           label=f"Activo ({m_bt['ret_port']*100:+.1f}%)")
ax_bt.plot(n_bt_b.index, n_bt_b.values, color=C_BH, lw=1.5, ls="--",
           label=f"Buy&Hold ({m_bt['ret_bh']*100:+.1f}%)")
ax_bt.plot(n_bt_m.index, n_bt_m.values, color=C_BM, lw=1.5, ls=":",
           label=f"EWW ({m_bt['ret_bm']*100:+.1f}%)")
ax_bt.axhline(1, color=C_GRAY, lw=0.6, ls=":")
marcar_rebalanceos(ax_bt, frb_bt, n_bt_p.index)
ax_bt.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=8)
ax_bt.set_title("Retorno Acumulado – Backtest 2024 (base 1)")
ax_bt.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

# ── (0,2): Drawdown backtest ──────────────────────────────
ax_dd_bt = fig.add_subplot(gs[0, 2])
estilo_ax(ax_dd_bt)
acum_bt = (1 + m_bt["rp"]).cumprod()
dd_bt   = ((acum_bt - acum_bt.cummax()) / acum_bt.cummax()) * 100
ax_dd_bt.fill_between(dd_bt.index, dd_bt.values, 0, alpha=0.6, color=C_BM)
ax_dd_bt.plot(dd_bt.index, dd_bt.values, color=C_BM, lw=1.2)
ax_dd_bt.axhline(0, color=C_GRAY, lw=0.6)
ax_dd_bt.set_title(f"Drawdown BT 2024 (max {m_bt['max_dd']*100:.1f}%)")
ax_dd_bt.yaxis.set_major_formatter(mtick.PercentFormatter())

# ── (1,0)+(1,1): Retorno acumulado YTD ───────────────────
ax_ytd = fig.add_subplot(gs[1, :2])
estilo_ax(ax_ytd)
n_ytd_p = sp_ytd  / sp_ytd.iloc[0]
n_ytd_b = sbh_ytd / sbh_ytd.iloc[0]
n_ytd_m = bm_ytd.iloc[:,0] / bm_ytd.iloc[0,0]
ax_ytd.plot(n_ytd_p.index, n_ytd_p.values, color=C_ACTIVO, lw=2.0,
            label=f"Activo ({m_ytd['ret_port']*100:+.1f}%)")
ax_ytd.plot(n_ytd_b.index, n_ytd_b.values, color=C_BH, lw=1.5, ls="--",
            label=f"Buy&Hold ({m_ytd['ret_bh']*100:+.1f}%)")
ax_ytd.plot(n_ytd_m.index, n_ytd_m.values, color=C_BM, lw=1.5, ls=":",
            label=f"EWW ({m_ytd['ret_bm']*100:+.1f}%)")
ax_ytd.axhline(1, color=C_GRAY, lw=0.6, ls=":")
marcar_rebalanceos(ax_ytd, frb_ytd, n_ytd_p.index)
ax_ytd.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=8)
ax_ytd.set_title(f"Retorno Acumulado – YTD {HOY.year} (base 1)")
ax_ytd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

# ── (1,2): Drawdown YTD ──────────────────────────────────
ax_dd_ytd = fig.add_subplot(gs[1, 2])
estilo_ax(ax_dd_ytd)
acum_ytd = (1 + m_ytd["rp"]).cumprod()
dd_ytd   = ((acum_ytd - acum_ytd.cummax()) / acum_ytd.cummax()) * 100
ax_dd_ytd.fill_between(dd_ytd.index, dd_ytd.values, 0, alpha=0.6, color=C_BM)
ax_dd_ytd.plot(dd_ytd.index, dd_ytd.values, color=C_BM, lw=1.2)
ax_dd_ytd.axhline(0, color=C_GRAY, lw=0.6)
ax_dd_ytd.set_title(f"Drawdown YTD {HOY.year} (max {m_ytd['max_dd']*100:.1f}%)")
ax_dd_ytd.yaxis.set_major_formatter(mtick.PercentFormatter())

# ── (2,0): Pesos finales YTD – pie ───────────────────────
ax_pie = fig.add_subplot(gs[2, 0])
estilo_ax(ax_pie)
vt_ytd  = sum(ten_ytd.values())
pf_ytd  = [ten_ytd[t] / vt_ytd for t in TICKERS]
nom_pie = [UNIVERSO[t].split()[0] for t in TICKERS]
cols_pie = plt.cm.Blues(np.linspace(0.35, 0.9, N))
_, _, autotexts = ax_pie.pie(pf_ytd, labels=nom_pie, autopct="%1.1f%%",
                              colors=cols_pie, startangle=140,
                              textprops={"color":"#c9d1d9","fontsize":6})
for at in autotexts:
    at.set_color("#f0f6fc"); at.set_fontsize(6)
ax_pie.set_title(f"Pesos Finales YTD {HOY.year}")

# ── (2,1): Evolucion de pesos YTD ────────────────────────
ax_pesos = fig.add_subplot(gs[2, 1])
estilo_ax(ax_pesos)
if hp_ytd:
    fechas_rb_s  = sorted(hp_ytd.keys())
    cols_lineas  = plt.cm.tab10(np.linspace(0, 0.95, N))
    for i, t in enumerate(TICKERS):
        ys = [hp_ytd[f][t] * 100 for f in fechas_rb_s]
        ax_pesos.plot(
            [f.to_pydatetime() for f in fechas_rb_s], ys,
            marker="o", ms=4, lw=1.4,
            color=cols_lineas[i],
            label=UNIVERSO[t].split()[0],
        )
    ax_pesos.axhline(PESO_MINIMO * 100, color=C_GRAY, lw=1.0, ls="--",
                     label=f"Floor {PESO_MINIMO*100:.0f}%")
    ax_pesos.set_title(f"Evolucion de Pesos YTD {HOY.year} (%)")
    ax_pesos.set_ylabel("%")
    ax_pesos.legend(facecolor="#21262d", labelcolor="#c9d1d9",
                    fontsize=6, ncol=2, loc="upper left")
else:
    ax_pesos.text(0.5, 0.5, "Sin rebalanceos", ha="center", va="center",
                  color=C_GRAY, transform=ax_pesos.transAxes)
    ax_pesos.set_title(f"Evolucion de Pesos YTD {HOY.year} (%)")

# ── (2,2): Alpha rolling 21d – ambos periodos superpuestos
ax_roll = fig.add_subplot(gs[2, 2])
estilo_ax(ax_roll)
exc_bt_roll  = (m_bt["rp"]  - m_bt["rb"]).rolling(21).mean()  * 252 * 100
exc_ytd_roll = (m_ytd["rp"] - m_ytd["rb"]).rolling(21).mean() * 252 * 100
ax_roll.plot(exc_bt_roll.index,  exc_bt_roll.values,
             color=C_GOLD, lw=1.5, label="BT 2024")
ax_roll.plot(exc_ytd_roll.index, exc_ytd_roll.values,
             color=C_ACTIVO, lw=1.5, label=f"YTD {HOY.year}")
ax_roll.axhline(0, color=C_BM, lw=0.8, ls="--")
ax_roll.fill_between(exc_ytd_roll.index, exc_ytd_roll.values, 0,
                     where=exc_ytd_roll.values >= 0, alpha=0.18, color=C_BH)
ax_roll.fill_between(exc_ytd_roll.index, exc_ytd_roll.values, 0,
                     where=exc_ytd_roll.values < 0,  alpha=0.18, color=C_BM)
ax_roll.set_title("Alpha Rolling 21d vs EWW (anualiz. %)")
ax_roll.yaxis.set_major_formatter(mtick.PercentFormatter())
ax_roll.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=7)

plt.suptitle(
    f"Portafolio Activo Mexicano  |  Reponderacion mensual por Alpha CAPM  |  "
    f"Backtest 2024: alpha={m_bt['alpha_jensen']*100:+.2f}%  |  "
    f"YTD {HOY.year}: alpha={m_ytd['alpha_jensen']*100:+.2f}%",
    color="#f0f6fc", fontsize=10, y=1.01,
)

output_png = r"portafolio_activo_mx.png"
plt.savefig(output_png, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nGrafica guardada: {output_png}")
plt.show()

print("\nAnalisis completado.")