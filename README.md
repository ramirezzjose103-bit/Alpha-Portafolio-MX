# Portafolio Activo Mexicano — Reponderación por Alpha CAPM

Estrategia activa sobre 10 acciones fijas de la BMV con reponderación trimestral basada
en alpha CAPM individual vs el ETF EWW (proxy MSCI México).

## Cómo funciona

En cada rebalanceo (enero, abril, julio, octubre) se estima el alpha de cada acción
mediante la regresión de Jensen (1968) sobre excesos de retorno respecto a CETES:

```
(R_i − Rf) = α + β·(R_bm − Rf) + ε
```

Las acciones con `α > 0` reciben mayor peso; las de `α ≤ 0` conservan solo el floor.
Cada operación descuenta comisión de 0.25% + IVA. No hay look-ahead bias: la regresión
usa únicamente los 63 días hábiles previos a cada fecha de rebalanceo.

## Parámetros clave

| Parámetro | Valor |
|---|---|
| Universo | 10 acciones BMV (fijo) |
| Rebalanceo | Trimestral |
| Ventana alpha | 63 días hábiles |
| Floor por acción | 8% |
| Comisión | 0.25% + IVA |
| Tasa libre de riesgo | CETES ~10.5% anual |
| Benchmark | EWW (iShares MSCI Mexico) |

## Resultados de referencia

| Métrica | Backtest 2024 | YTD 2026 |
|---|---|---|
| Retorno portafolio activo | −13.4% | +11.0% |
| Retorno buy-and-hold | −10.4% | +10.9% |
| Retorno benchmark (EWW) | −27.5% | +9.3% |
| Alpha Jensen | −4.31% | +4.21% |
| Máximo drawdown | −20.5% | −9.0% |

## Instalación y uso

```bash
pip install yfinance pandas numpy scipy tabulate matplotlib python-dateutil
python portafolio_mexico_activo.py
```

## Limitaciones

El universo de acciones fue seleccionado con conocimiento del período evaluado. El
benchmark EWW cotiza en USD y no se ajusta por tipo de cambio. Con 10 acciones y
rebalanceo trimestral, los estimadores tienen intervalos de confianza amplios.

## Fuentes

Jensen (1968) · Sharpe (1964) · Lintner (1965) · Grinold & Kahn, *Active Portfolio Management* (2000)
