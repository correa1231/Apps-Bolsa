import yfinance as yf
import pandas as pd
import numpy as np
import datetime
#from scipy.stats import norm # Importamos estadística normal (viene con numpy/pandas a veces, si falla usamos math)
import math

def calcular_rsi(serie, periodo):
    delta = serie.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_gain = ganancia.ewm(alpha=1/periodo, adjust=False).mean()
    avg_loss = perdida.ewm(alpha=1/periodo, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

window = 14
vol_short = 5
vol_long = 10
ticker = "AMD"
datos = yf.download(ticker, start="2020-01-01", progress=False)
if isinstance(datos.columns, pd.MultiIndex):
    datos.columns = datos.columns.droplevel(1)
datos['Prev_Close'] = datos['Close'].shift(1)
datos['TR1'] = datos['High'] - datos['Low']
datos['TR2'] = abs(datos['High'] - datos['Prev_Close'])
datos['TR3'] = abs(datos['Low'] - datos['Prev_Close'])
datos['TR'] = datos[['TR1', 'TR2', 'TR3']].max(axis=1)
datos.drop(columns=['Prev_Close', 'TR1', 'TR2', 'TR3'], inplace=True)
up_move = datos['High'] - datos['High'].shift(1)
down_move = datos['Low'].shift(1) - datos['Low']
datos['+DM'] = 0.0
datos['-DM'] = 0.0
datos['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
datos['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
datos['TR14'] = datos['TR'].ewm(alpha=1/window, adjust=False).mean()
datos['+DM14'] = datos['+DM'].ewm(alpha=1/window, adjust=False).mean()
datos['-DM14'] = datos['-DM'].ewm(alpha=1/window, adjust=False).mean()
datos['+DI'] = 100 * (datos['+DM14'] / datos['TR14'])
datos['-DI'] = 100 * (datos['-DM14'] / datos['TR14'])
datos['DX'] = 100 * abs(datos['+DI'] - datos['-DI']) / (datos['+DI'] + datos['-DI'])
datos['ADX'] = datos['DX'].ewm(alpha=1/window, adjust=False).mean()
datos.drop(columns=['TR14', '+DM14', '-DM14', 'DX'], inplace=True)
datos['Vol_SMA_short'] = datos['Volume'].rolling(window=vol_short).mean()
datos['Vol_SMA_long'] = datos['Volume'].rolling(window=vol_long).mean()
datos['Vol_Osc'] = ((datos['Vol_SMA_short'] - datos['Vol_SMA_long']) / datos['Vol_SMA_long']) * 100
datos.drop(columns=['Vol_SMA_short', 'Vol_SMA_long'], inplace=True)
datos['CRSI_Price'] = calcular_rsi(datos['Close'], 3)
streak_data = [0] * len(datos)
close_prices = datos['Close'].values
for i in range(1, len(datos)):
    if close_prices[i] > close_prices[i-1]:
        streak_data[i] = streak_data[i-1] + 1 if streak_data[i-1] >= 0 else 1
    elif close_prices[i] < close_prices[i-1]:
        streak_data[i] = streak_data[i-1] -1 if streak_data[i-1] <= 0 else -1
    else:
        streak_data[i] = 0
datos['Streak'] = streak_data
datos['CRSI_Streak'] = calcular_rsi(datos['Streak'], 3)
datos['Return'] = datos['Close'].pct_change()
datos['PercentRank'] = datos['Return'].rolling(window=100).rank(pct=True) * 100
datos['CRSI'] = (datos['CRSI_Price'] + datos['CRSI_Streak'] + datos['PercentRank']) / 3
columnas_borrar = ['CRSI_Price', 'CRSI_Streak', 'PercentRank', 'Return', 'Streak']
datos.drop(columns=columnas_borrar, inplace=True, errors='ignore')
ultimo_dia = datos.iloc[-1]
print(f"\n--- INFORME DE ANALISIS PARA {ticker}) ---")
print(f"Precio Cierre: {ultimo_dia['Close']:.2f}")
print(f"Fuerza Tendencia (ADX): {ultimo_dia['ADX']:.2f}")
print(f"Nivel de Sobrecompra (CRSI): {ultimo_dia['CRSI']:.2f}")
print(f"Salud del Volumnen (Vol_Osc): {ultimo_dia['Vol_Osc']:.2f}")
print("-" * 40)
decision = "NEUTRO / ESPERAR"
if ultimo_dia['ADX'] > 25:
    if ultimo_dia['CRSI'] < 20:
        decision = "COMPRAR"
        if ultimo_dia['Vol_Osc'] > 0:
            decision += " +  CONFIRMACION DE VOLUMEN"
    elif ultimo_dia['CRSI'] > 80:
        decision = "VENDER / TOMAR BENEFICIOS"
else: 
    decision = "MERCADO LATERAL (ADX  DÉBIL) - MEJOR NO OPERAR"
print(f"CONCLUSIÓN: {decision}")
print("\n" + "="*70)
print(f"--- PREDICCIÓN DE VALOR PARA LOS PRÓXIMOS 7 DÍAS ({ticker}) ---")
print("="*70)

# 1. Configuración
dias_a_proyectar = 7
num_simulaciones = 1000

# 2. Matemáticas de la Acción
# Usamos los últimos 50 días para que la "memoria" sea reciente
log_returns = np.log(datos['Close'] / datos['Close'].shift(1)).tail(50).dropna()

u = log_returns.mean()
var = log_returns.var()
stdev = log_returns.std()
drift = u - (0.5 * var) # Dirección promedio

# 3. Motor de Simulación
Z = np.random.normal(0, 1, (dias_a_proyectar, num_simulaciones))
daily_returns = np.exp(drift + stdev * Z)

# 4. Construcción de Caminos
price_paths = np.zeros_like(daily_returns)
precio_ultimo = datos['Close'].iloc[-1]
price_paths[0] = precio_ultimo * daily_returns[0]

for t in range(1, dias_a_proyectar):
    price_paths[t] = price_paths[t-1] * daily_returns[t]

# --- RESULTADOS CLAROS ---

print(f"Precio HOY: ${precio_ultimo:.2f}")
print("-" * 70)
print(f"{'FECHA':<12} | {'ESTIMACIÓN VALOR':<18} | {'CAMBIO ($)':<12} | {'TENDENCIA'}")
print("-" * 70)

fecha_hoy = datetime.date.today()
precio_ayer = precio_ultimo

for t in range(dias_a_proyectar):
    # Calculamos la MEDIA de los 1000 escenarios para este día
    valor_estimado = np.mean(price_paths[t])
    
    # Fecha
    fecha_futura = fecha_hoy + datetime.timedelta(days=t+1)
    
    # Diferencia con el día anterior (para ver si sube o baja)
    diferencia = valor_estimado - precio_ayer
    
    # Icono visual
    if diferencia > 0:
        icono = "🟢 SUBE"
        signo = "+"
    else:
        icono = "🔴 BAJA"
        signo = "" # El negativo ya sale solo
        
    print(f"{fecha_futura.strftime('%d-%m-%Y'):<12} | ${valor_estimado:<17.2f} | {signo}{diferencia:<11.2f} | {icono}")
    
    # Actualizamos el precio "ayer" para la siguiente vuelta del bucle
    precio_ayer = valor_estimado

print("-" * 70)
print("NOTA: Este valor es el 'Promedio Matemático' de 1.000 simulaciones.")
#print(datos.head())
#print(datos[['High', 'Low', 'Close', 'TR']].tail())
#print(datos[['Close', '+DI', '-DI', 'ADX']].tail(10))
#print(datos[['Volume', 'Vol_Osc']].tail())
#print(datos[['Close', 'Streak', 'CRSI_Streak']].tail())
#print("--- Dato con todos los indicadores ---")
#print(datos[['Close', 'ADX', 'Vol_Osc', 'CRSI']].tail(10))