import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sqlite3
import bcrypt
import threading
import matplotlib.pyplot as plt
import os
import hashlib
import joblib
import time
# --- LIBRERIAS IA AVANZADA ---
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

# ==========================================
# 0. CONFIGURACIÓN E IDIOMAS
# ==========================================
ETORO_FEE_PER_ORDER = 1.0  
ETORO_ROUND_TRIP = 2.0     
# Configuración de apariencia
ctk.set_appearance_mode("dark")  # Modos: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue") # Temas: "blue" (standard), "green", "dark-blue"

# Paleta de colores mejorada
C_BG_MODERN = "#1a1a1a"
C_PANEL_MODERN = "#2b2b2b"
C_ACCENT_MODERN = "#1f6aa5"

LANG = {
    "ES": {
        "app_title": "Gestor Pro v26.0 (Quant Architect)",
        "port_title": "📂 MI CARTERA & VIGILANCIA",
        "opp_title": "💎 OPORTUNIDADES & OBJETIVOS",
        "scan_own": "⚡ ACTUALIZAR",
        "save": "💾", "sell": "💰 VENDER", "del_btn": "🗑", "viz_btn": "📊 VIZ", "stats_btn": "📈 STATS", "risk_btn": "🔥 RIESGO", "snap_btn": "📷 SNAPSHOT", "hist": "📜 HIST", "exp": "📄 EXP", "scan_mkt": "🔍 ESCANEAR", "analyze": "▶ ANALIZAR", "reset_zoom": "RESET", "buy_price": "Compra:", "qty": "Cant:",
        "col_ticker": "Ticker", "col_entry": "Entrada", "col_state": "Estado", "col_score": "Pts", "col_diag": "Diagnóstico / Objetivo",
        "vigil": "👁 VIGILANDO", "msg_wait": "⏳...", "msg_scan": "⏳ ANALIZANDO...", "msg_exp_ok": "✅ Guardado.", "msg_snap_ok": "✅ Reporte guardado.", "msg_sell_title": "Cerrar Posición", "msg_sell_ask": "Precio Venta ($):", "msg_del_confirm": "¿Borrar de la lista?",
        "hist_title": "Historial (Neto)", "hist_tot": "P/L Neto:", "viz_title": "Distribución", "stats_title": "Auditoría", "risk_title": "Matriz Riesgo", "conf_title": "Config", "conf_lang": "Idioma:", "conf_logout": "🔒 SALIR", "conf_del": "⚠️ BORRAR", "conf_del_confirm": "¿Seguro?", "refresh_all": "🔄 TODO",
        "fund_title": "📊 FUNDAMENTALES:", "fund_pe": "PER:", "fund_cap": "Mkt Cap:", "fund_div": "Div:", "ws_rating": "Analistas:", "ws_target": "Objetivo Wall St:",
        "graham_title": "💎 VALOR GRAHAM:", "bench_title": "🆚 MERCADO:", "bench_beta": "Beta:", "bench_rel": "Relativo:",
        "ai_title": "🤖 IA (Gradient Boosting):", "ai_prob": "Prob. Éxito:", "ai_acc": "Precisión Test:", "ai_factors": "Factores Clave:",
        "tech_title": "📐 TECH & STOP LOSS:", "tech_sup": "Soporte:", "tech_res": "Resistencia:", "tech_sl": "Stop Loss:", "trend_wk": "Tendencia Semanal:",
        "target_title": "🎯 OBJETIVO (Take Profit):", "news_title": "📰 NOTICIAS:", "calc_title": "Calc Riesgo", "calc_cap": "Capital:", "calc_risk": "Riesgo %:", "calc_stop": "Stop:", "calc_btn": "CALCULAR", "calc_res": "Comprar:", "calc_apply": "APLICAR",
        "dash_inv": "Invertido:", "dash_val": "Valor:", "dash_pl": "Neto P/L:", "macro_fear": "😨 MIEDO", "macro_greed": "🤑 CODICIA", "macro_neutral": "😐 NEUTRO",
        "login_title": "ACCESO", "user": "Usuario:", "pass": "Clave:", "btn_enter": "ENTRAR", "btn_reg": "REGISTRO", "err_login": "Error", "ok_reg": "OK", "err_reg": "Existe"
    },
    "EN": { "app_title": "Pro Manager v26.0", "port_title": "📂 PORTFOLIO", "opp_title": "💎 OPPORTUNITIES", "scan_own": "⚡ REFRESH", "save": "💾", "sell": "💰 SELL", "del_btn": "🗑", "viz_btn": "📊 VIZ", "stats_btn": "📈 STATS", "risk_btn": "🔥 RISK", "snap_btn": "📷 SNAPSHOT", "hist": "📜 HIST", "exp": "📄 EXP", "scan_mkt": "🔍 SCAN", "analyze": "▶ ANALYZE", "reset_zoom": "RESET", "buy_price": "Price:", "qty": "Qty:", "col_ticker": "Ticker", "col_entry": "Entry", "col_state": "Status", "col_score": "Pts", "col_diag": "Diagnosis / Target", "vigil": "👁 WATCH", "msg_wait": "⏳...", "msg_scan": "⏳...", "msg_exp_ok": "✅ Saved.", "msg_snap_ok": "✅ Report saved.", "msg_sell_title": "Close Position", "msg_sell_ask": "Sell Price ($):", "msg_del_confirm": "Delete?", "hist_title": "Trade History", "hist_tot": "Total Net P/L:", "viz_title": "Portfolio Allocation", "stats_title": "Performance Audit", "risk_title": "Correlation Matrix", "conf_title": "Settings", "conf_lang": "Language:", "conf_logout": "🔒 LOGOUT", "conf_del": "⚠️ DELETE", "conf_del_confirm": "Sure?", "refresh_all": "🔄 ALL", "fund_title": "📊 FUNDAMENTALS:", "fund_pe": "P/E:", "fund_cap": "Cap:", "fund_div": "Div:", "ws_rating": "Analysts:", "ws_target": "WS Target:", "graham_title": "💎 GRAHAM VALUE:", "bench_title": "🆚 MARKET:", "bench_beta": "Beta:", "bench_rel": "Rel. Perf:", "ai_title": "🤖 AI PREDICTION:", "ai_prob": "Win Prob:", "ai_acc": "Test Accuracy:", "ai_factors": "Key Factors:", "tech_title": "📐 TECH & STOP LOSS:", "tech_sup": "Support:", "tech_res": "Resistance:", "tech_sl": "Suggested Stop:", "trend_wk": "Weekly Trend:", "target_title": "🎯 TARGET PRICE:", "news_title": "📰 NEWS:", "calc_title": "Risk Calc", "calc_cap": "Capital:", "calc_risk": "Risk %:", "calc_stop": "Stop Loss:", "calc_btn": "CALCULATE", "calc_res": "Buy:", "calc_apply": "APPLY", "dash_inv": "Invested:", "dash_val": "Value:", "dash_pl": "Net P/L:", "macro_fear": "😨 FEAR", "macro_greed": "🤑 GREED", "macro_neutral": "😐 NEUTRAL", "login_title": "LOGIN", "user": "User:", "pass": "Pass:", "btn_enter": "GO", "btn_reg": "REG", "err_login": "Error", "ok_reg": "OK", "err_reg": "Exists" },
}
if "FR" not in LANG: LANG["FR"] = LANG["EN"]
if "PT" not in LANG: LANG["PT"] = LANG["EN"]

CANDIDATOS_VIP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", 
    "AMD", "INTC", "PYPL", "KO", "PEP", "DIS", "BA", "CSCO", "WMT", "JPM",
    "UBER", "ABNB", "PLTR", "SHOP", "SBUX", "NKE", "MCD", "V", "MA"
]

C_BG = "#1e1e1e"; C_FG = "#ffffff"; C_ACCENT = "#007acc"
C_PANEL = "#252526"; C_GREEN = "#4ec9b0"; C_RED = "#f44747"; C_GOLD = "#ffd700"; C_PURPLE = "#8a2be2"; C_ORANGE = "#e67e22"

# ==========================================
# 1. BASE DE DATOS (APUNTANDO A v24)
# ==========================================
class DatabaseManager:
    def __init__(self, db_name="bolsa_datos_v24.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.crear_tablas()

    def crear_tablas(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            username TEXT UNIQUE NOT NULL, 
            password TEXT NOT NULL)''')
        # ... (resto de tablas se mantienen igual)
        self.conn.commit()

    def registrar_usuario(self, u, p):
        # Generar hash con salting automático
        salt = bcrypt.gensalt()
        hashed_pw = bcrypt.hashpw(p.encode('utf-8'), salt)
        
        try:
            # Guardamos el hash como string para la DB
            self.conn.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", 
                              (u, hashed_pw.decode('utf-8')))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: 
            return False # Usuario ya existe

    def verificar_usuario(self, u, p):
        res = self.conn.execute("SELECT id, password FROM usuarios WHERE username=?", (u,)).fetchone()
        if res:
            uid, hashed_db = res
            # 1. Intentar verificar con BCrypt
            try:
                if bcrypt.checkpw(p.encode('utf-8'), hashed_db.encode('utf-8')):
                    return uid
            except ValueError:
                # 2. Si falla por "Invalid Salt", es que es un hash viejo (SHA-256)
                # Vamos a verificarlo a la antigua para dejarte entrar
                old_hash = hashlib.sha256(p.encode()).hexdigest()
                if old_hash == hashed_db:
                    # Opcional: Podrías actualizar el hash aquí a Bcrypt
                    return uid
        return None

    def guardar_posicion(self, uid, t, p, q):
        f = datetime.datetime.now().strftime("%Y-%m-%d")
        self.conn.execute("INSERT INTO cartera (user_id, ticker, precio_compra, cantidad, fecha_guardado) VALUES (?,?,?,?,?)", (uid, t, p, q, f))
        self.conn.commit()

    def obtener_cartera(self, uid):
        return self.conn.execute("SELECT ticker, precio_compra, cantidad, fecha_guardado, id FROM cartera WHERE user_id=?", (uid,)).fetchall()

    def borrar_posicion(self, pid):
        self.conn.execute("DELETE FROM cartera WHERE id=?", (pid,)); self.conn.commit()

    def cerrar_posicion(self, uid, pid, sell_price):
        data = self.conn.execute("SELECT ticker, precio_compra, cantidad FROM cartera WHERE id=?", (pid,)).fetchone()
        if not data: return
        tkr, buy_p, qty = data
        gross_profit = (sell_price - buy_p) * qty
        net_profit = gross_profit - ETORO_ROUND_TRIP 
        date_out = datetime.datetime.now().strftime("%Y-%m-%d")
        self.conn.execute("INSERT INTO historial (user_id, ticker, buy_price, sell_price, qty, profit, date_out) VALUES (?,?,?,?,?,?,?)", 
                          (uid, tkr, buy_p, sell_price, qty, net_profit, date_out))
        self.conn.execute("DELETE FROM cartera WHERE id=?", (pid,)); self.conn.commit()

    def obtener_historial_completo(self, uid):
        return self.conn.execute("SELECT ticker, buy_price, sell_price, profit, date_out FROM historial WHERE user_id=?", (uid,)).fetchall()

    def borrar_usuario_completo(self, uid):
        self.conn.execute("DELETE FROM cartera WHERE user_id=?", (uid,))
        self.conn.execute("DELETE FROM historial WHERE user_id=?", (uid,))
        self.conn.execute("DELETE FROM usuarios WHERE id=?", (uid,))
        self.conn.commit()

# ==========================================
# 2. MOTOR ANALÍTICO (GRADIENT BOOSTING)
# ==========================================
class AnalistaBolsa:
    def __init__(self):
        self.data = None; self.ticker = ""; self.spy_data = None; self.data_weekly = None
        self.model = None; self.scaler = None; self.features_cols = []

    def descargar_datos(self, ticker):
        self.ticker = ticker.upper()
        try:
            # CAMBIO: period="max" descarga TODO el historial disponible
            d = yf.download(self.ticker, period="max", progress=False)
            
            if d.empty: raise ValueError("Datos vacíos")
            
            # Limpieza de MultiIndex (necesario para versiones nuevas de yfinance)
            if isinstance(d.columns, pd.MultiIndex): 
                d.columns = d.columns.droplevel(1)
            
            self.data = d.astype(float)
            
            # Datos semanales (también ponemos max para ver tendencias de largo plazo)
            w = yf.download(self.ticker, period="max", interval="1wk", progress=False)
            if isinstance(w.columns, pd.MultiIndex): 
                w.columns = w.columns.droplevel(1)
            self.data_weekly = w.astype(float)
            
            return d
        except Exception as e: 
            print(f"Error descarga: {e}")
            raise ValueError("Error descarga")

    # --- DATOS EXTERNOS ---

    def calcular_indicadores(self):
        if self.data is None or self.data.empty: return self.data
        
        # Trabajamos sobre una copia para asegurar consistencia
        df = self.data.copy()
        
        # 1. Medias Móviles (SMA)
        df['SMA_50'] = df.ta.sma(length=50)
        df['SMA_200'] = df.ta.sma(length=200)
        
        # 2. RSI (Relative Strength Index)
        # Usamos length=3 para mantener tu lógica de "CRSI" (Connors RSI simplificado)
        df['CRSI'] = df.ta.rsi(length=3)
        
        # 3. MACD (Moving Average Convergence Divergence)
        # pandas_ta devuelve un DataFrame con 3 columnas (MACD, Histograma, Señal)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        
        # 4. Bandas de Bollinger (BBANDS)
        bbands = df.ta.bbands(length=20, std=2)
        df['UpperBB'] = bbands['BBU_20_2.0']
        df['LowerBB'] = bbands['BBL_20_2.0']
        # Posición relativa en bandas (%B)
        df['BB_Pct'] = (df['Close'] - df['LowerBB']) / (df['UpperBB'] - df['LowerBB'])
        
        # 5. Volatilidad y Tendencia (ATR y ADX)
        df['ATR'] = df.ta.atr(length=14)
        adx = df.ta.adx(length=14)
        df['ADX'] = adx['ADX_14']
        
        # 6. Oscilador de Volumen (PVO)
        # Sustituimos tu Vol_Osc manual por el Percentage Volume Oscillator
        pvo = df.ta.pvo(fast=12, slow=26, signal=9)
        df['Vol_Osc'] = pvo['PVOh_12_26_9'] # Usamos el histograma como indicador de momentum
        
        # 7. VWAP (Volume Weighted Average Price)
        # Importante: pandas_ta calcula el VWAP acumulado diario automáticamente
        # $$VWAP = \frac{\sum (Precio \times Volumen)}{\sum Volumen}$$
        df['VWAP'] = df.ta.vwap()
        
        # Limpieza final: Rellenar NaNs iniciales con 0
        self.data = df.fillna(0)
        return self.data
    
    def calcular_fibonacci(self):
        try:
            df = self.data.tail(126)
            max_p = df['High'].max(); min_p = df['Low'].min(); diff = max_p - min_p
            return {"0": min_p, "1": max_p, "0.382": max_p - 0.382*diff, "0.5": max_p - 0.5*diff, "0.618": max_p - 0.618*diff}
        except: return None

    def obtener_consenso_analistas(self, ticker):
        try:
            t = yf.Ticker(ticker); i = t.info
            rec = i.get('recommendationKey', 'none').upper().replace("_", " ")
            target = i.get('targetMeanPrice', 0)
            return rec, target
        except: return "N/A", 0

    def obtener_sentimiento_mercado(self):
        try:
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            spy = yf.Ticker("SPY").history(period="6mo")['Close']
            sma50 = spy.rolling(50).mean().iloc[-1]; price_spy = spy.iloc[-1]
            if vix > 25: return "FEAR", vix
            elif vix < 15 and price_spy > sma50: return "GREED", vix
            else: return "NEUTRAL", vix
        except: return "NEUTRAL", 0

    def calcular_valor_graham(self, ticker):
        try:
            t = yf.Ticker(ticker); i = t.info
            eps = i.get('trailingEps'); bvps = i.get('bookValue')
            if eps and bvps and eps > 0 and bvps > 0: return np.sqrt(22.5 * eps * bvps)
            return 0
        except: return 0

    def obtener_fundamentales(self, ticker):
        try:
            t = yf.Ticker(ticker); i = t.info
            per = i.get('trailingPE', i.get('forwardPE', 0))
            cap = i.get('marketCap', 0)
            div = i.get('dividendYield'); 
            if div is None: div = i.get('trailingAnnualDividendYield', 0)
            sec = i.get('sector', 'N/A'); ind = i.get('industry', 'N/A')
            if cap > 1e12: s_cap = f"{cap/1e12:.2f}T"
            elif cap > 1e9: s_cap = f"{cap/1e9:.2f}B"
            else: s_cap = str(cap)
            g_val = self.calcular_valor_graham(ticker)
            return {"per": f"{per:.2f}" if per else "N/A", "cap": s_cap, "div": f"{div*100:.2f}%", "sec": sec, "ind": ind, "graham": g_val, "valid": True}
        except: return {"per": "-", "cap": "-", "div": "0%", "sec": "-", "ind": "-", "graham": 0, "valid": False}

    def calcular_indicadores(self):
        if self.data is None or self.data.empty: return self.data
        df = self.data.copy()

        # 1. SMAs (Medias Móviles)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # 2. RSI (Standard 14 periods - Wilder's Smoothing)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['CRSI'] = 100 - (100 / (1 + rs))

        # 3. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # 4. Bandas de Bollinger
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['UpperBB'] = sma20 + (std20 * 2)
        df['LowerBB'] = sma20 - (std20 * 2)
        df['BB_Pct'] = (df['Close'] - df['LowerBB']) / (df['UpperBB'] - df['LowerBB'])

        # 5. ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()

        # 6. ADX (Average Directional Index)
        up_move = df['High'].diff()
        down_move = df['Low'].diff().multiply(-1)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_smooth = true_range.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr_smooth)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # 7. Vol_Osc (Percentage Volume Oscillator - Histograma)
        vol_ema12 = df['Volume'].ewm(span=12, adjust=False).mean()
        vol_ema26 = df['Volume'].ewm(span=26, adjust=False).mean()
        df['Vol_Osc'] = ((vol_ema12 - vol_ema26) / vol_ema26) * 100

        # 8. VWAP (Volume Weighted Average Price)
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()

        self.data = df.fillna(0)
        return self.data

    def analizar_tendencia_semanal(self):
        try:
            if self.data_weekly is None or len(self.data_weekly) < 30: return "Neutral"
            last = self.data_weekly.iloc[-1]
            ma30 = self.data_weekly['Close'].rolling(30).mean().iloc[-1]
            if last['Close'] > ma30: return "Alcista"
            else: return "Bajista"
        except: return "Neutral"

    # --- IA GRADIENT BOOSTING (MEJORADA) ---
    def calcular_probabilidad_ia(self):
        try:
            # Directorio para guardar modelos
            if not os.path.exists("models"): os.makedirs("models")
            model_path = f"models/{self.ticker}_model.pkl"
            
            df = self.data.copy()
            if len(df) < 150: return 50.0, 0.0, []

            # --- 1. INGENIERÍA DE FEATURES (SIEMPRE SE EJECUTA) ---
            # Movemos esto AL PRINCIPIO para que el df siempre tenga estas columnas
            df['Retorno'] = df['Close'].pct_change()
            df['Lag_1'] = df['Retorno'].shift(1)
            df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
            df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
            
            # Definimos las columnas que usa el modelo
            self.features_cols = ['CRSI', 'MACD', 'BB_Pct', 'ADX', 'Dist_SMA50', 'Dist_VWAP', 'Retorno', 'Lag_1', 'Vol_Osc']
            
            # Limpieza básica para evitar errores de predicción
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # --- 2. INTENTAR CARGAR MODELO EXISTENTE ---
            if os.path.exists(model_path):
                try:
                    # Comprobamos si el archivo es reciente (< 24h)
                    file_age = time.time() - os.path.getmtime(model_path)
                    if file_age < 86400:
                        saved_data = joblib.load(model_path)
                        
                        # Verificación de seguridad: ¿El modelo guardado usa las mismas columnas?
                        if saved_data.get('features') == self.features_cols:
                            self.model = saved_data['model']
                            self.scaler = saved_data['scaler']
                            
                            # Predicción directa
                            last_day = df[self.features_cols].tail(1)
                            last_day_scaled = self.scaler.transform(last_day)
                            prob = self.model.predict_proba(last_day_scaled)[0][1] * 100
                            
                            importances = self.model.feature_importances_
                            top_factors = sorted(zip(self.features_cols, importances), key=lambda x: x[1], reverse=True)[:2]
                            
                            return prob, 0.0, top_factors # 0.0 Acc porque no re-evaluamos
                except Exception:
                    pass # Si falla la carga, simplemente re-entrenamos abajo

            # --- 3. SI NO EXISTE O ES VIEJO: ENTRENAR DESDE CERO ---
            # Preparar Target
            future_close = df['Close'].shift(-3)
            df['Target'] = (future_close > df['Close'] * 1.015).astype(int)
            
            # Limpieza estricta para entrenamiento
            df_train = df.dropna(subset=self.features_cols + ['Target'])
            
            if len(df_train) < 50: return 50.0, 0.0, []

            data_model = df_train.iloc[:-3].copy()
            X = data_model[self.features_cols]
            y = data_model['Target']
            
            if len(y.unique()) < 2: return 50.0, 0.0, []

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            split = int(len(X) * 0.85)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(X_train, y_train)
            
            acc = precision_score(y_test, self.model.predict(X_test), zero_division=0) * 100
            
            # Guardar modelo, scaler y la lista de columnas usada
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features_cols
            }, model_path)
            
            # Predicción hoy
            last_day = df[self.features_cols].iloc[[-1]]
            last_day_scaled = self.scaler.transform(last_day)
            prob = self.model.predict_proba(last_day_scaled)[0][1] * 100
            
            importances = self.model.feature_importances_
            top_factors = sorted(zip(self.features_cols, importances), key=lambda x: x[1], reverse=True)[:2]
            
            return prob, acc, top_factors

        except Exception as e:
            print(f"Error detallado en IA: {e}")
            return 50.0, 0.0, []

    def detectar_niveles(self):
        try:
            df = self.data.tail(60) 
            minimo = df['Low'].min()
            maximo = df['High'].max()
            return minimo, maximo
        except: return 0, 0

    def obtener_benchmark(self):
        if self.spy_data is not None: return self.spy_data
        try:
            d = yf.download("SPY", period="2y", progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.droplevel(1)
            self.spy_data = d.astype(float)
            return self.spy_data
        except: return None

    def calcular_beta_relativa(self, stock_df, spy_df):
        try:
            df = pd.DataFrame({'STOCK': stock_df['Close'], 'SPY': spy_df['Close']}).dropna()
            if len(df) < 50: return {"beta": 0, "rel_perf": 0}
            rets = df.pct_change().dropna()
            cov = rets['STOCK'].cov(rets['SPY']); var = rets['SPY'].var()
            beta = cov / var if var != 0 else 1.0
            lookback = min(126, len(df))
            stock_ret = (df['STOCK'].iloc[-1] / df['STOCK'].iloc[-lookback]) - 1
            spy_ret = (df['SPY'].iloc[-1] / df['SPY'].iloc[-lookback]) - 1
            rel_perf = (stock_ret - spy_ret) * 100
            return {"beta": beta, "rel_perf": rel_perf}
        except: return {"beta": 0, "rel_perf": 0}

    def obtener_noticias_analizadas(self, ticker):
        try:
            t = yf.Ticker(ticker); news = t.news; analisis_noticias = []
            bull = ['surge', 'jump', 'rise', 'gain', 'profit', 'beat', 'growth', 'record', 'buy', 'bull', 'upgrade', 'high', 'positive', 'soar', 'rally']
            bear = ['drop', 'fall', 'plunge', 'loss', 'miss', 'cut', 'bear', 'downgrade', 'low', 'negative', 'crash', 'risk', 'slump']
            for n in news[:4]:
                tit = n.get('title'); 
                if not tit and 'content' in n: tit = n['content'].get('title')
                if not tit: continue
                src = n.get('publisher', 'Yahoo')
                if isinstance(src, dict): src = src.get('title', 'Yahoo')
                elif isinstance(src, str): src = src
                score = 0; tit_l = tit.lower()
                for w in bull: 
                    if w in tit_l: score += 1
                for w in bear: 
                    if w in tit_l: score -= 1
                sent = "neutral"
                if score>0: sent="bull"
                elif score<0: sent="bear"
                analisis_noticias.append({"title": tit, "source": src, "sentiment": sent})
            return analisis_noticias
        except: return []

    def simular(self, dias=7):
        ret = np.log(self.data['Close']/self.data['Close'].shift(1)).tail(60).dropna()
        if len(ret) < 2: return np.zeros((dias, 1000))
        u, v = ret.mean(), ret.var(); dr = u-(0.5*v); st = ret.std()
        Z = np.random.normal(0,1,(dias,1000)); drt = np.exp(dr+st*Z)
        p = np.zeros_like(drt); p[0] = self.data['Close'].iloc[-1]*drt[0]
        for t in range(1,dias): p[t] = p[t-1]*drt[t]
        return p

    def generar_diagnostico_interno(self, p_compra=0):
        try:
            last = self.data.iloc[-1]
            adx = last.get('ADX', 0); crsi = last.get('CRSI', 50); curr = last['Close']; atr = last.get('ATR', 0)
            trend_weekly = self.analizar_tendencia_semanal()
            raw_score = 100 - crsi
            if adx > 25: raw_score += 10
            if trend_weekly == "Alcista": raw_score += 10 
            if trend_weekly == "Bajista": raw_score -= 20 
            score = max(0, min(100, int(raw_score)))
            min_p = self.data['Low'].tail(60).min()
            max_p = self.data['High'].tail(60).max()
            target = max_p
            if curr >= max_p * 0.98: 
                target = last['UpperBB'] 
                if target < curr: target = curr * 1.05 
            profit_pot = ((target - curr) / curr) * 100
            stop_loss = curr - (atr * 2) 
            msg = "Neutro"; tag = ""
            if p_compra > 0: 
                 price = last['Close']
                 if crsi > 80: msg = "⚠️ VENDER"; tag = "sell"
                 elif price < p_compra*0.9: msg = "🛑 STOP LOSS"; tag = "sell"
                 elif crsi < 20 and adx > 25 and trend_weekly == "Alcista": msg = "💎 ACUMULAR"; tag = "buy"
                 else: msg = "✋ MANTENER"; tag = "hold"
                 return {"valid": True, "ticker": self.ticker, "score": score, "msg": msg, "tag": tag, "crsi": crsi, "adx": adx, "price": curr, "target": target, "profit": profit_pot, "stop_loss": stop_loss, "weekly": trend_weekly}
            if score >= 90: msg = "🚀 COMPRA"; tag = "buy"
            elif score >= 75: msg = "👀 BARATO"; tag = "near"
            elif score >= 60: msg = "📈 TENDENCIA"; tag = "trend"
            elif score <= 20: msg = "⚠️ CARO"; tag = "sell"
            elif score <= 40: msg = "💤 DÉBIL"; tag = "weak"
            return {"valid": True, "ticker": self.ticker, "score": score, "msg": msg, "tag": tag, "crsi": crsi, "adx": adx, "price": curr, "target": target, "profit": profit_pot, "stop_loss": stop_loss, "weekly": trend_weekly}
        except: return {"valid": False, "msg": "Error", "score": 0, "tag": "", "target": 0, "profit": 0, "stop_loss": 0, "weekly": "N/A"}

# ==========================================
# 3. GUI THEME MANAGER
# ==========================================
def apply_dark_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure(".", background=C_BG, foreground=C_FG, font=("Segoe UI", 9))
    style.configure("TLabel", background=C_BG, foreground=C_FG)
    style.configure("TLabelFrame", background=C_BG, foreground=C_ACCENT, borderwidth=1, relief="solid")
    style.configure("TLabelFrame.Label", background=C_BG, foreground=C_ACCENT, font=("Segoe UI", 10, "bold"))
    style.configure("TButton", background=C_PANEL, foreground=C_FG, borderwidth=0, padding=5)
    style.map("TButton", background=[('active', C_ACCENT), ('pressed', C_ACCENT)], foreground=[('active', 'white')])
    style.configure("TEntry", fieldbackground=C_PANEL, foreground=C_FG, borderwidth=0)
    style.configure("Treeview", background=C_PANEL, foreground=C_FG, fieldbackground=C_PANEL, borderwidth=0)
    style.configure("Treeview.Heading", background="#333", foreground="white", relief="flat")
    style.map("Treeview", background=[('selected', C_ACCENT)], foreground=[('selected', 'white')])
    style.configure("TPanedwindow", background=C_BG)
    root.configure(bg=C_BG)

# ==========================================
# 4. APP PRINCIPAL
# ==========================================
class LoginWindow:
    def __init__(self, root, db, on_success):
        self.root = root
        self.db = db
        self.on_success = on_success
        
        # CAMBIO: Usamos un Frame que cubre toda la ventana principal
        self.frame = ctk.CTkFrame(root, fg_color="#1a1a1a")
        self.frame.pack(fill="both", expand=True)
        
        # Decoración
        ctk.CTkLabel(self.frame, text="QUANT ARCHITECT", font=("Segoe UI", 24, "bold"), text_color="#1f6aa5").pack(pady=(60, 5))
        ctk.CTkLabel(self.frame, text="Professional Trading Terminal", font=("Segoe UI", 12), text_color="gray").pack(pady=(0, 40))

        # Inputs
        input_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=100) # Más margen lateral para centrar

        self.e_u = ctk.CTkEntry(input_frame, placeholder_text="Usuario", height=45, corner_radius=10)
        self.e_u.pack(fill="x", pady=10)

        self.e_p = ctk.CTkEntry(input_frame, placeholder_text="Contraseña", show="*", height=45, corner_radius=10)
        self.e_p.pack(fill="x", pady=10)

        # Botones
        self.btn_log = ctk.CTkButton(self.frame, text="ENTRAR", command=self.log, font=("Segoe UI", 14, "bold"), height=45, corner_radius=10)
        self.btn_log.pack(pady=(30, 10), padx=100, fill="x")

        self.btn_reg = ctk.CTkButton(self.frame, text="CREAR CUENTA", command=self.reg, fg_color="transparent", border_width=2, height=45, corner_radius=10)
        self.btn_reg.pack(pady=5, padx=100, fill="x")

    def log(self):
        user_val = self.e_u.get()
        pass_val = self.e_p.get()
        uid = self.db.verificar_usuario(user_val, pass_val)
        if uid:
            # Truco para evitar el error en Python 3.14:
            # Ocultamos el frame primero, y luego lo destruimos
            self.frame.pack_forget()
            self.frame.destroy() 
            self.on_success(uid, user_val)
        else:
            messagebox.showerror("Error", "Credenciales incorrectas")

    def reg(self):
        if self.db.registrar_usuario(self.e_u.get(), self.e_p.get()):
            messagebox.showinfo("OK", "Usuario registrado. Puedes entrar.")
        else:
            messagebox.showerror("Error", "El usuario ya existe")

class AppBolsa:
    def __init__(self, root, uid, uname, db):
        self.root = root
        self.uid = uid
        self.db = db
        self.eng = AnalistaBolsa()
        self.current_lang = "ES"
        self.texts = LANG[self.current_lang]
        
        # Configuración Ventana
        self.root.title(f"{self.texts['app_title']} - {uname}")
        self.root.geometry("1600x950")
        self.root.configure(fg_color=C_BG_MODERN)

        # Contenedor Principal
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Panel Izquierdo (Ancho fijo para herramientas)
        self.side_panel = ctk.CTkFrame(self.main_container, width=380, corner_radius=15, fg_color=C_PANEL_MODERN)
        self.side_panel.pack(side="left", fill="both", padx=(0, 10))
        self.side_panel.pack_propagate(False)

        # Panel Derecho (Gráficos y Controles)
        self.content_panel = ctk.CTkFrame(self.main_container, corner_radius=15, fg_color=C_PANEL_MODERN)
        self.content_panel.pack(side="right", fill="both", expand=True)

        # Montar todas las piezas
        self.crear_widgets_laterales()
        self.crear_widgets_principales()
        
        # Carga Inicial
        self.update_ui_language()
        self.load_init()
        threading.Thread(target=self.actualizar_mood, daemon=True).start()

    def setup_side_panel(self):
        # Título de Cartera
        self.lbl_port_title = ctk.CTkLabel(self.side_panel, text=self.texts["port_title"], 
                                          font=("Segoe UI", 16, "bold"), text_color=C_ACCENT_MODERN)
        self.lbl_port_title.pack(pady=15)

        # Dashboard de P&L (Una "tarjeta" dentro del panel)
        self.dash_frame = ctk.CTkFrame(self.side_panel, fg_color=C_PANEL_MODERN, corner_radius=10)
        self.dash_frame.pack(fill="x", padx=15, pady=5)
        
        self.lbl_invested = ctk.CTkLabel(self.dash_frame, text="Invertido: ---", font=("Segoe UI", 12))
        self.lbl_invested.pack(anchor="w", padx=10, pady=2)
        
        self.lbl_pl = ctk.CTkLabel(self.dash_frame, text="Neto P/L: ---", font=("Segoe UI", 13, "bold"))
        self.lbl_pl.pack(anchor="w", padx=10, pady=(0, 5))

        # El Treeview (Mantenemos ttk porque CTK no tiene, pero le damos estilo)
        style = ttk.Style()
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        
        self.tr1 = ttk.Treeview(self.side_panel, columns=("tk", "pr", "sg"), show="headings")
        self.tr1.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Botones de Acción
        self.btn_act = ctk.CTkButton(self.side_panel, text=self.texts["scan_own"], 
                                     command=self.scan_own, font=("Segoe UI", 12, "bold"))
        self.btn_act.pack(fill="x", padx=15, pady=5)

    def crear_widgets_laterales(self):
        """Panel Izquierdo: Mood, Dashboard, Cartera, Herramientas y Oportunidades"""
        
        # --- 0. SENTIMIENTO DE MERCADO (¡AQUÍ ESTÁBA EL ERROR!) ---
        # Esta es la etiqueta que faltaba
        self.lbl_mood = ctk.CTkLabel(self.side_panel, text="MERCADO: ⏳ Cargando...", 
                                     font=("Segoe UI", 12, "bold"), text_color="gray")
        self.lbl_mood.pack(pady=(15, 5))

        # --- 1. DASHBOARD SUPERIOR ---
        self.lbl_port_title = ctk.CTkLabel(self.side_panel, text="📂 MI CARTERA", 
                                          font=("Segoe UI", 16, "bold"), text_color="#1f6aa5")
        self.lbl_port_title.pack(pady=(5, 5))

        # Tarjeta de P/L
        self.dash_frame = ctk.CTkFrame(self.side_panel, fg_color="#1e1e1e", corner_radius=8)
        self.dash_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_invested = ctk.CTkLabel(self.dash_frame, text="Inv: ---", font=("Segoe UI", 11))
        self.lbl_invested.pack(anchor="w", padx=10, pady=2)
        self.lbl_current = ctk.CTkLabel(self.dash_frame, text="Val: ---", font=("Segoe UI", 11))
        self.lbl_current.pack(anchor="w", padx=10, pady=2)
        self.lbl_pl = ctk.CTkLabel(self.dash_frame, text="P/L: ---", font=("Segoe UI", 12, "bold"))
        self.lbl_pl.pack(anchor="w", padx=10, pady=(0, 5))

        # Botón Actualizar Cartera (Grande)
        self.btn_act = ctk.CTkButton(self.side_panel, text="⚡ ACTUALIZAR", command=self.scan_own, height=30)
        self.btn_act.pack(fill="x", padx=10, pady=5)

        # --- 2. TABLA CARTERA ---
        self.tr1 = ttk.Treeview(self.side_panel, columns=("tk", "pr", "sg"), show="headings", height=8)
        self.tr1.pack(fill="both", expand=True, padx=10, pady=5)
        self.tr1.bind("<Double-1>", lambda e: self.sel_load(self.tr1, True))

        # --- 3. BOTONERA DE ACCIONES (Grid 2x4) ---
        tools_frame = ctk.CTkFrame(self.side_panel, fg_color="transparent")
        tools_frame.pack(fill="x", padx=10, pady=5)

        # Fila 1: Operaciones Básicas
        self.btn_save = ctk.CTkButton(tools_frame, text="💾", width=40, command=self.save, fg_color="#333", hover_color="#444")
        self.btn_save.grid(row=0, column=0, padx=2, pady=2)
        
        self.btn_sell = ctk.CTkButton(tools_frame, text="💰", width=40, command=self.vender_posicion, fg_color="#800000", hover_color="#a00000")
        self.btn_sell.grid(row=0, column=1, padx=2, pady=2)
        
        self.btn_del = ctk.CTkButton(tools_frame, text="🗑", width=40, command=self.dele, fg_color="#333", hover_color="#555")
        self.btn_del.grid(row=0, column=2, padx=2, pady=2)
        
        self.btn_hist = ctk.CTkButton(tools_frame, text="📜", width=40, command=self.ver_historial, fg_color="#333", hover_color="#444")
        self.btn_hist.grid(row=0, column=3, padx=2, pady=2)

        # Fila 2: Análisis Avanzado
        self.btn_stats = ctk.CTkButton(tools_frame, text="📈", width=40, command=self.ver_estadisticas, fg_color="#2e8b57", hover_color="#1e5a38")
        self.btn_stats.grid(row=1, column=0, padx=2, pady=2)

        self.btn_risk = ctk.CTkButton(tools_frame, text="🔥", width=40, command=self.ver_correlaciones, fg_color="#e67e22", hover_color="#d35400")
        self.btn_risk.grid(row=1, column=1, padx=2, pady=2)

        self.btn_viz = ctk.CTkButton(tools_frame, text="📊", width=40, command=self.ver_distribucion, fg_color="#8a2be2", hover_color="#5e17eb")
        self.btn_viz.grid(row=1, column=2, padx=2, pady=2)

        self.btn_exp = ctk.CTkButton(tools_frame, text="📄", width=40, command=self.exportar_cartera, fg_color="#333", hover_color="#444")
        self.btn_exp.grid(row=1, column=3, padx=2, pady=2)
        
        # Centrar columnas del grid
        for i in range(4): tools_frame.columnconfigure(i, weight=1)

        # --- 4. SECCIÓN OPORTUNIDADES ---
        ctk.CTkLabel(self.side_panel, text="💎 OPORTUNIDADES", 
                    font=("Segoe UI", 14, "bold"), text_color="#ffd700").pack(pady=(15, 5))

        self.tr2 = ttk.Treeview(self.side_panel, columns=("tk", "sc", "ms"), show="headings", height=8)
        self.tr2.pack(fill="both", expand=True, padx=10, pady=5)
        self.tr2.bind("<Double-1>", lambda e: self.sel_load(self.tr2, False))

        self.btn_gem = ctk.CTkButton(self.side_panel, text="🔍 ESCANEAR MERCADO", command=self.scan_mkt, fg_color="#2e8b57")
        self.btn_gem.pack(fill="x", padx=10, pady=(5, 15))

    def crear_widgets_principales(self):
        """Panel Derecho: Barra de Control Superior y Zona de Trabajo"""
        
        # --- 1. BARRA DE CONTROL SUPERIOR ---
        self.ctrl_bar = ctk.CTkFrame(self.content_panel, height=50, fg_color="#2b2b2b", corner_radius=10)
        self.ctrl_bar.pack(fill="x", padx=15, pady=10)

        # Grupo Izquierda: Input Ticker y Analizar
        self.e_tk = ctk.CTkEntry(self.ctrl_bar, placeholder_text="TICKER", width=100)
        self.e_tk.pack(side="left", padx=(10, 5), pady=8)
        self.e_tk.bind('<Return>', lambda e: self.run())

        self.btn_run = ctk.CTkButton(self.ctrl_bar, text="▶ ANALIZAR", width=90, command=self.run)
        self.btn_run.pack(side="left", padx=5)

        # Grupo Centro: Precio, Cantidad y Herramientas
        ctk.CTkLabel(self.ctrl_bar, text="Precio:").pack(side="left", padx=(15, 2))
        self.e_pr = ctk.CTkEntry(self.ctrl_bar, width=70)
        self.e_pr.pack(side="left", padx=2)

        ctk.CTkLabel(self.ctrl_bar, text="Cant:").pack(side="left", padx=(10, 2))
        self.e_qt = ctk.CTkEntry(self.ctrl_bar, width=50)
        self.e_qt.pack(side="left", padx=2)

        # Botones pequeños de utilidad (Limpiar y Calculadora)
        ctk.CTkButton(self.ctrl_bar, text="🗑", width=30, command=self.limpiar_campos, fg_color="#444").pack(side="left", padx=(10, 2))
        ctk.CTkButton(self.ctrl_bar, text="🧮", width=30, command=self.abrir_calculadora, fg_color="#444").pack(side="left", padx=2)

        # Grupo Derecha: SOLO Refresh (Se eliminó Configuración)
        self.btn_refresh = ctk.CTkButton(self.ctrl_bar, text="🔄 TODO", width=60, command=self.refresh_all, fg_color="#8a2be2")
        self.btn_refresh.pack(side="right", padx=10)

        # --- 2. ZONA DE TRABAJO (TEXTO + GRÁFICO) ---
        self.work_area = ctk.CTkFrame(self.content_panel, fg_color="transparent")
        self.work_area.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Panel de Texto (Izquierda)
        text_frame = ctk.CTkFrame(self.work_area, fg_color="transparent")
        text_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # Toolbar del texto (Reset Zoom y Snapshot)
        tools_text = ctk.CTkFrame(text_frame, fg_color="transparent", height=30)
        tools_text.pack(fill="x", pady=(0, 5))
        
        self.b_rst = ctk.CTkButton(tools_text, text="RESET ZOOM", width=80, state="disabled", command=self.zoom_rst, fg_color="#444")
        self.b_rst.pack(side="left")
        
        self.btn_snap = ctk.CTkButton(tools_text, text="📷 SNAP", width=60, command=self.generar_reporte, fg_color="teal")
        self.btn_snap.pack(side="right")

        # El Widget de Texto
        self.txt = tk.Text(text_frame, width=45, bg="#1e1e1e", fg="white", 
                          font=("Consolas", 10), borderwidth=0, padx=10, pady=10)
        self.txt.pack(fill="both", expand=True)
        # Tags de colores
        self.txt.tag_config("t", foreground="#1f6aa5", font=("Consolas", 11, "bold"))
        self.txt.tag_config("p", foreground="#4ec9b0")
        self.txt.tag_config("n", foreground="#f44747")
        self.txt.tag_config("gold", foreground="#ffd700")
        self.txt.tag_config("w", foreground="white")
        self.txt.tag_config("news_bull", foreground="#4ec9b0")
        self.txt.tag_config("news_bear", foreground="#f44747")
        self.txt.tag_config("ai_good", foreground="#00ff00", font=("Consolas", 11, "bold"))
        self.txt.tag_config("ai_bad", foreground="#ff4444", font=("Consolas", 11, "bold"))

        # Panel Gráfico (Derecha)
        self.graph_frame = ctk.CTkFrame(self.work_area, fg_color="#1e1e1e", corner_radius=10)
        self.graph_frame.pack(side="right", fill="both", expand=True)

        plt.style.use('dark_background')
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="#1e1e1e")
        self.cv = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.cv.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)
        
        self.tb = NavigationToolbar2Tk(self.cv, self.graph_frame)
        self.tb.configure(background="#1e1e1e")
        self.tb._message_label.config(background="#1e1e1e", foreground="white")
        self.tb.update()
        
    # ==========================================
    # FUNCIONES DE GESTIÓN DE DATOS Y UI
    # (Pega esto dentro de la clase AppBolsa)
    # ==========================================

    def load_init(self):
        """Carga la cartera desde la base de datos al Treeview"""
        for i in self.tr1.get_children(): self.tr1.delete(i)
        for d in self.db.obtener_cartera(self.uid):
            # d = (ticker, precio_compra, cantidad, fecha, id)
            pr = f"${d[1]}" if d[1] > 0 else "VIGILANDO"
            # Insertamos usando el ID de la base de datos como iid del treeview
            self.tr1.insert("", "end", iid=d[4], values=(d[0], pr, "..."))

    def sel_load(self, tree, is_own):
        """Carga los datos al hacer doble clic en una tabla"""
        s = tree.selection()
        if not s: return
        vals = tree.item(s[0])['values']
        tkr = vals[0]
        pc = 0; qt = 0
        
        if is_own:
            # Si es mi cartera, busco el precio y cantidad real
            tid = int(s[0])
            for d in self.db.obtener_cartera(self.uid):
                if d[4] == tid: 
                    pc = d[1]; qt = d[2]; break
        
        self.e_tk.delete(0, tk.END); self.e_tk.insert(0, tkr)
        self.e_pr.delete(0, tk.END); self.e_pr.insert(0, str(pc))
        self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qt))
        self.run()

    def save(self):
        """Guarda o actualiza una posición"""
        t = self.e_tk.get()
        p = self.e_pr.get()
        q = self.e_qt.get()
        if not t: return
        if not p: p = "0"
        if not q: q = "0"
        
        # Guardamos en DB
        self.db.guardar_posicion(self.uid, t, float(p), float(q))
        # Recargamos la tabla
        self.load_init()
        # Limpiamos campos
        self.limpiar_campos()

    def dele(self):
        """Borra la posición seleccionada"""
        s = self.tr1.selection()
        if s: 
            if messagebox.askyesno("Confirmar", "¿Borrar de la lista?"):
                self.db.borrar_posicion(s[0])
                self.load_init()

    def vender_posicion(self):
        """Gestiona la venta y pase al historial"""
        s = self.tr1.selection()
        if not s: return
        iid = int(s[0]); item = None
        for d in self.db.obtener_cartera(self.uid):
            if d[4] == iid: item = d; break
        if not item: return
        
        precio_actual = 0
        try:
            # Intentamos adivinar el precio actual
            data = yf.Ticker(item[0]).history(period='1d')
            if not data.empty: precio_actual = data['Close'].iloc[-1]
        except: pass
        
        ask_price = simpledialog.askfloat("Cerrar Posición", f"Precio Venta ($) para {item[0]}:", initialvalue=precio_actual)
        if ask_price is not None:
            self.db.cerrar_posicion(self.uid, iid, ask_price)
            self.load_init()
            self.scan_own() # Actualizamos dashboard

    def update_row_ui(self, iid, tkr, pr_text, msg, tag):
        """Actualiza una fila específica tras el escaneo (hilo seguro)"""
        if self.tr1.exists(iid): 
            self.tr1.item(iid, values=(tkr, pr_text, msg), tags=(tag,))

    def limpiar_campos(self):
        self.e_pr.delete(0, tk.END)
        self.e_qt.delete(0, tk.END)
        self.e_tk.delete(0, tk.END)

    def zoom_rst(self):
        self.tb.home()

    def refresh_all(self): 
        self.run()
        self.scan_own()
        self.scan_mkt()

    def generar_reporte(self):
        tkr = self.e_tk.get()
        if not tkr: return
        if not os.path.exists("Reports"): os.makedirs("Reports")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"Reports/{tkr}_{timestamp}"
        try: 
            self.fig.savefig(f"{fname}.png", facecolor='#1e1e1e') # Fondo oscuro
            content = self.txt.get("1.0", tk.END)
            with open(f"{fname}.txt", "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Snapshot", "✅ Reporte guardado en carpeta 'Reports'")
        except Exception as e: 
            messagebox.showerror("Error", str(e))

    # ==========================================
    # VENTANAS SECUNDARIAS (MODERNIZADAS A CTK)
    # ==========================================

    def abrir_calculadora(self):
        cw = ctk.CTkToplevel(self.root)
        cw.title("Calculadora de Riesgo")
        cw.geometry("300x400")
        cw.attributes("-topmost", True) # Siempre encima
        
        ctk.CTkLabel(cw, text="Capital ($):").pack(pady=2)
        e_cap = ctk.CTkEntry(cw); e_cap.pack(); e_cap.insert(0, "10000")
        
        ctk.CTkLabel(cw, text="Riesgo (%):").pack(pady=2)
        e_risk = ctk.CTkEntry(cw); e_risk.pack(); e_risk.insert(0, "1")
        
        ctk.CTkLabel(cw, text="Entrada ($):").pack(pady=2)
        e_ent = ctk.CTkEntry(cw); e_ent.pack()
        if self.e_pr.get(): e_ent.insert(0, self.e_pr.get())
        
        ctk.CTkLabel(cw, text="Stop Loss ($):").pack(pady=2)
        e_stop = ctk.CTkEntry(cw); e_stop.pack()
        
        lbl_res = ctk.CTkLabel(cw, text="---", font=("bold", 14), text_color="#1f6aa5")
        lbl_res.pack(pady=15)
        
        def calcular():
            try:
                c = float(e_cap.get()); r = float(e_risk.get()); en = float(e_ent.get()); st = float(e_stop.get())
                if en <= st: lbl_res.configure(text="¡Stop >= Entrada!", text_color="red"); return
                risk_amt = c * (r/100)
                diff = en - st
                qty = int(risk_amt / diff)
                lbl_res.configure(text=f"{qty} Acciones", text_color="#4ec9b0")
                return qty
            except: lbl_res.configure(text="Error", text_color="red")

        def aplicar():
            qty = calcular()
            if qty and qty > 0:
                self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qty))
                self.e_pr.delete(0, tk.END); self.e_pr.insert(0, e_ent.get())
                cw.destroy()

        ctk.CTkButton(cw, text="CALCULAR", command=calcular).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(cw, text="APLICAR A APP", command=aplicar, fg_color="#2e8b57").pack(fill="x", padx=20, pady=5)

    def ver_historial(self):
        hw = ctk.CTkToplevel(self.root)
        hw.title("Historial de Operaciones")
        hw.geometry("600x400")
        
        cols = ("Ticker", "Buy", "Sell", "P/L", "Date")
        trh = ttk.Treeview(hw, columns=cols, show="headings")
        for c in cols: trh.heading(c, text=c); trh.column(c, width=80, anchor="center")
        trh.pack(fill="both", expand=True, padx=10, pady=10)
        
        total_pl = 0.0
        data = self.db.obtener_historial_completo(self.uid)
        for d in data:
            pl = d[3]; total_pl += pl
            tag = "win" if pl >= 0 else "loss"
            trh.insert("", "end", values=(d[0], f"{d[1]:.2f}", f"{d[2]:.2f}", f"{pl:+.2f}", d[4]), tags=(tag,))
        
        trh.tag_configure("win", foreground="#4ec9b0")
        trh.tag_configure("loss", foreground="#f44747")
        
        col_lbl = "#4ec9b0" if total_pl >= 0 else "#f44747"
        ctk.CTkLabel(hw, text=f"Total P/L Realizado: ${total_pl:+.2f}", 
                     font=("Segoe UI", 16, "bold"), text_color=col_lbl).pack(pady=10)

    def abrir_config(self):
        cw = ctk.CTkToplevel(self.root)
        cw.title("Configuración")
        cw.geometry("300x250")
        
        ctk.CTkLabel(cw, text="Idioma del Sistema:", font=("bold", 12)).pack(pady=10)
        cb = ctk.CTkComboBox(cw, values=["ES", "EN", "FR", "PT"], state="readonly")
        cb.set(self.current_lang)
        cb.pack(pady=5)
        
        def save_conf():
            self.current_lang = cb.get()
            self.texts = LANG[self.current_lang]
            self.update_ui_language()
            cw.destroy()
            
        def realizar_logout():
            cw.destroy()
            self.logout()
            
        ctk.CTkButton(cw, text="GUARDAR", command=save_conf).pack(pady=10)
        ctk.CTkButton(cw, text="CERRAR SESIÓN", command=realizar_logout, fg_color="#333").pack(pady=5)
        ctk.CTkButton(cw, text="BORRAR CUENTA", command=self.borrar_cuenta, fg_color="#800000").pack(pady=5)

    def logout(self):
        self.root.withdraw()
        for widget in self.root.winfo_children(): widget.destroy()
        LoginWindow(self.root, self.db, lambda u, n: (self.root.deiconify(), AppBolsa(self.root, u, n, self.db)))

    def borrar_cuenta(self):
        if messagebox.askyesno("PELIGRO", "¿Estás seguro de borrar tu cuenta y todos los datos?"):
            self.db.borrar_usuario_completo(self.uid)
            self.root.destroy()

    # --- Funciones Visuales Menores (Estadísticas, Distribución, Correlaciones) ---
    # Para no alargar demasiado, si necesitas ver_estadisticas, ver_distribucion, etc.
    # dímelo y te paso sus versiones modernizadas. Por ahora, pon esto para que no falle:
    
    def ver_estadisticas(self):
        data = self.db.obtener_historial_completo(self.uid)
        if not data: 
            messagebox.showinfo("Vacío", "No hay operaciones cerradas en el historial.")
            return

        # Cálculos Financieros
        profits = [d[3] for d in data]
        wins = [p for p in profits if p >= 0]
        losses = [p for p in profits if p < 0]
        
        total_ops = len(profits)
        win_rate = (len(wins) / total_ops) * 100 if total_ops > 0 else 0
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 999.0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        # Ventana Modal
        sw = ctk.CTkToplevel(self.root)
        sw.title("Auditoría de Rendimiento")
        sw.geometry("700x550")
        sw.attributes("-topmost", True)
        
        # Tarjetas de Métricas
        f_metrics = ctk.CTkFrame(sw, fg_color="transparent")
        f_metrics.pack(fill="x", padx=10, pady=10)

        def create_card(parent, title, value, color):
            card = ctk.CTkFrame(parent, fg_color="#2b2b2b", corner_radius=10)
            card.pack(side="left", expand=True, fill="both", padx=5)
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 11, "bold"), text_color="gray").pack(pady=(10, 0))
            ctk.CTkLabel(card, text=value, font=("Segoe UI", 18, "bold"), text_color=color).pack(pady=(0, 10))

        create_card(f_metrics, "WIN RATE", f"{win_rate:.1f}%", "#4ec9b0" if win_rate > 50 else "#f44747")
        create_card(f_metrics, "PROFIT FACTOR", f"{profit_factor:.2f}", "#4ec9b0" if profit_factor > 1.5 else "#ffd700")
        create_card(f_metrics, "MAX DRAWDOWN", f"-${max_dd:.2f}", "#f44747")
        create_card(f_metrics, "OPERACIONES", f"{total_ops}", "white")

        # Gráfico de Equity Curve
        graph_frame = ctk.CTkFrame(sw, fg_color="#1e1e1e", corner_radius=10)
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#1e1e1e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        
        # Línea Verde Neón y Relleno
        ax.plot(cumulative, color='#00ff00', linewidth=2, label="Capital Acumulado")
        ax.fill_between(range(len(cumulative)), cumulative, color='#00ff00', alpha=0.1)
        
        ax.set_title("Equity Curve (Curva de Capital)", color="white", fontsize=10)
        ax.grid(True, alpha=0.1, linestyle='--')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('#1e1e1e')
        ax.spines['left'].set_color('white'); ax.spines['right'].set_color('#1e1e1e')

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def ver_correlaciones(self):
        db_data = self.db.obtener_cartera(self.uid)
        if not db_data or len(db_data) < 2: 
            messagebox.showinfo("Requisito", "Necesitas al menos 2 activos en cartera para ver correlaciones.")
            return

        tickers = [row[0] for row in db_data]
        
        # Ventana de Carga
        loading = ctk.CTkToplevel(self.root)
        loading.geometry("200x100")
        ctk.CTkLabel(loading, text="Calculando Matrix...").pack(pady=30)
        self.root.update()

        try:
            # Descargamos solo los cierres ajustados de los últimos 6 meses
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            corr_matrix = data.corr()
            loading.destroy()

            cw = ctk.CTkToplevel(self.root)
            cw.title("Matriz de Riesgo y Correlación")
            cw.geometry("700x700")
            cw.attributes("-topmost", True)

            fig = Figure(figsize=(6, 6), dpi=100, facecolor="#1e1e1e")
            ax = fig.add_subplot(111)
            
            # Mapa de calor (Rojo = Alta correlación/Riesgo, Verde = Baja/Diversificación)
            cax = ax.imshow(corr_matrix, cmap='RdYlGn_r', vmin=-1, vmax=1)
            
            # Etiquetas
            ax.set_xticks(range(len(tickers)))
            ax.set_yticks(range(len(tickers)))
            ax.set_xticklabels(tickers, rotation=90, color="white")
            ax.set_yticklabels(tickers, color="white")
            
            # Valores dentro de las celdas
            for i in range(len(tickers)):
                for j in range(len(tickers)):
                    val = corr_matrix.iloc[i, j]
                    text_color = "black" if abs(val) < 0.5 else "white"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

            ax.set_title("Matriz de Correlación (1.0 = Mismo Riesgo)", color="white", pad=20)
            fig.colorbar(cax, fraction=0.046, pad=0.04)
            
            canvas = FigureCanvasTkAgg(fig, master=cw)
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            loading.destroy()
            messagebox.showerror("Error", f"No se pudo calcular correlación.\n{e}")

    def ver_distribucion(self):
        db_data = self.db.obtener_cartera(self.uid)
        if not db_data: return

        tickers = []
        values = []
        
        # Calcular valor actual aproximado (Precio compra * Cantidad)
        # Nota: Idealmente usarías precio actual, pero usamos precio compra para rapidez
        for row in db_data:
            val = row[1] * row[2]
            if val > 0:
                tickers.append(row[0])
                values.append(val)

        if sum(values) == 0: 
            messagebox.showinfo("Info", "No tienes capital invertido.")
            return

        vw = ctk.CTkToplevel(self.root)
        vw.title("Distribución de Activos")
        vw.geometry("600x500")
        vw.attributes("-topmost", True)

        fig = Figure(figsize=(5, 4), dpi=100, facecolor="#1e1e1e")
        ax = fig.add_subplot(111)
        
        # Gráfico de Donut Moderno
        wedges, texts, autotexts = ax.pie(values, labels=tickers, autopct='%1.1f%%', 
                                          startangle=90, pctdistance=0.85,
                                          textprops={'color':"white"})
        
        # Círculo central para hacer el "Donut"
        centre_circle = plt.Circle((0,0),0.70,fc='#1e1e1e')
        ax.add_artist(centre_circle)
        
        ax.set_title("Asignación de Capital", color="white", fontsize=12, fontweight="bold")
        
        # Colorear los textos de porcentaje
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

        canvas = FigureCanvasTkAgg(fig, master=vw)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def exportar_cartera(self):
        datos = self.db.obtener_cartera(self.uid)
        if not datos: 
            messagebox.showinfo("Info", "Cartera vacía.")
            return
            
        # Convertimos a DataFrame para facilitar la exportación
        df_export = pd.DataFrame(datos, columns=['Ticker', 'Precio Compra', 'Cantidad', 'Fecha', 'ID_DB'])
        df_export = df_export.drop(columns=['ID_DB'])
        
        # Calcular valor total por posición
        df_export['Total Invertido'] = df_export['Precio Compra'] * df_export['Cantidad']
        
        f = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV (Excel)", "*.csv"), ("Texto", "*.txt")], 
            title="Exportar Cartera"
        )
        
        if f:
            try: 
                df_export.to_csv(f, index=False, encoding='utf-8-sig', sep=';') # Punto y coma para Excel europeo
                messagebox.showinfo("Éxito", "✅ Cartera exportada correctamente.")
            except Exception as e: 
                messagebox.showerror("Error", str(e))

    def actualizar_mood(self):
        mood, vix = self.eng.obtener_sentimiento_mercado()
        txt = ""
        col = "gray"
        
        if mood == "FEAR": 
            txt = f"{self.texts['macro_fear']} (VIX {vix:.2f})"
            col = "#f44747" # Rojo (C_RED)
        elif mood == "GREED": 
            txt = f"{self.texts['macro_greed']} (VIX {vix:.2f})"
            col = "#4ec9b0" # Verde (C_GREEN)
        else: 
            txt = f"{self.texts['macro_neutral']} (VIX {vix:.2f})"
            col = "#ffd700" # Oro (C_GOLD)
            
        # CAMBIO CLAVE: .configure() y text_color
        self.lbl_mood.configure(text=f"MERCADO: {txt}", text_color=col)

    def generar_reporte(self):
        tkr = self.e_tk.get()
        if not tkr: return
        if not os.path.exists("Reports"): os.makedirs("Reports")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"Reports/{tkr}_{timestamp}"
        try: self.fig.savefig(f"{fname}.png")
        except: pass
        try:
            content = self.txt.get("1.0", tk.END)
            with open(f"{fname}.txt", "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Snapshot", self.texts["msg_snap_ok"])
        except Exception as e: messagebox.showerror("Error", str(e))

    def vender_posicion(self):
        s = self.tr1.selection()
        if not s: return
        iid = int(s[0]); item = None
        for d in self.db.obtener_cartera(self.uid):
            if d[4] == iid: item = d; break
        if not item: return
        precio_actual = 0
        try:
            data = yf.Ticker(item[0]).history(period='1d')
            if not data.empty: precio_actual = data['Close'].iloc[-1]
        except: pass
        ask_price = simpledialog.askfloat(self.texts["msg_sell_title"], f"{self.texts['msg_sell_ask']} {item[0]}", initialvalue=precio_actual)
        if ask_price is not None:
            self.db.cerrar_posicion(self.uid, iid, ask_price)
            self.load_init(); self.scan_own()

    def dele(self):
        s = self.tr1.selection()
        if s: 
            if messagebox.askyesno("!!!", self.texts["msg_del_confirm"]):
                self.db.borrar_posicion(s[0])
                self.load_init()

    def ver_historial(self):
        hw = tk.Toplevel(self.root); hw.title(self.texts["hist_title"]); hw.geometry("600x400")
        apply_dark_theme(hw)
        cols = ("Ticker", "Buy", "Sell", "P/L", "Date")
        trh = ttk.Treeview(hw, columns=cols, show="headings")
        for c in cols: trh.heading(c, text=c); trh.column(c, width=80, anchor="center")
        trh.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        total_pl = 0.0
        data = self.db.obtener_historial_completo(self.uid)
        for d in data:
            pl = d[3]; total_pl += pl
            tag = "win" if pl >= 0 else "loss"
            trh.insert("", "end", values=(d[0], f"{d[1]:.2f}", f"{d[2]:.2f}", f"{pl:+.2f}", d[4]), tags=(tag,))
        trh.tag_configure("win", foreground=C_GREEN); trh.tag_configure("loss", foreground=C_RED)
        lbl_tot = ttk.Label(hw, text=f"{self.texts['hist_tot']} ${total_pl:+.2f}", font=("Segoe UI", 12, "bold"))
        if total_pl >= 0: lbl_tot.configure(foreground=C_GREEN)
        else: lbl_tot.configure(foreground=C_RED)
        lbl_tot.pack(pady=10)

    def ver_estadisticas(self):
        data = self.db.obtener_historial_completo(self.uid)
        if not data: return
        profits = [d[3] for d in data]
        wins = [p for p in profits if p >= 0]; losses = [p for p in profits if p < 0]
        total_ops = len(profits)
        win_rate = (len(wins) / total_ops) * 100 if total_ops > 0 else 0
        total_profit = sum(wins); total_loss = abs(sum(losses))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 999.0
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = drawdown.max() if len(drawdown) > 0 else 0
        sw = tk.Toplevel(self.root); sw.title(self.texts["stats_title"]); sw.geometry("700x500")
        apply_dark_theme(sw)
        f_metrics = ttk.Frame(sw); f_metrics.pack(fill=tk.X, padx=10, pady=10)
        def make_metric(p, title, val, color):
            f = ttk.Frame(p, borderwidth=1, relief="solid"); f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            ttk.Label(f, text=title, font=("Segoe UI", 9)).pack(pady=2)
            ttk.Label(f, text=val, font=("Segoe UI", 14, "bold"), foreground=color).pack(pady=5)
        make_metric(f_metrics, "WIN RATE", f"{win_rate:.1f}%", C_GREEN if win_rate > 50 else C_RED)
        make_metric(f_metrics, "PROFIT FACTOR", f"{profit_factor:.2f}", C_GREEN if profit_factor > 1.5 else C_ACCENT)
        make_metric(f_metrics, "MAX DRAWDOWN", f"-${max_dd:.2f}", C_RED)
        make_metric(f_metrics, "TOTAL OPS", f"{total_ops}", C_FG)
        fig = Figure(figsize=(6,4), dpi=100, facecolor=C_BG)
        ax = fig.add_subplot(111)
        ax.plot(cumulative, color=C_GREEN, linewidth=2, label="Equity")
        ax.fill_between(range(len(cumulative)), cumulative, color=C_GREEN, alpha=0.1)
        ax.set_title("Curva de Capital Realizada (Equity Curve)", color="white")
        ax.grid(True, alpha=0.1); ax.tick_params(colors='white')
        cv = FigureCanvasTkAgg(fig, master=sw)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def ver_correlaciones(self):
        db_data = self.db.obtener_cartera(self.uid)
        if not db_data or len(db_data) < 2: 
            messagebox.showinfo("Info", "Necesitas al menos 2 acciones en cartera.")
            return
        tickers = [row[0] for row in db_data]
        try:
            data = yf.download(tickers, period="6mo")['Close']
            corr_matrix = data.corr()
            cw = tk.Toplevel(self.root); cw.title(self.texts["risk_title"]); cw.geometry("600x600")
            apply_dark_theme(cw)
            fig = Figure(figsize=(6,6), dpi=100, facecolor=C_BG)
            ax = fig.add_subplot(111)
            cax = ax.matshow(corr_matrix, cmap='RdYlGn_r') 
            fig.colorbar(cax)
            ax.set_xticks(range(len(tickers))); ax.set_yticks(range(len(tickers)))
            ax.set_xticklabels(tickers, rotation=90, color="white"); ax.set_yticklabels(tickers, color="white")
            ax.set_title(self.texts["risk_title"], color="white", pad=20)
            cv = FigureCanvasTkAgg(fig, master=cw)
            cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e: messagebox.showerror("Error", str(e))

    def ver_distribucion(self):
        db_data = self.db.obtener_cartera(self.uid)
        if not db_data: return
        tickers = [row[0] for row in db_data]
        values = [row[1] * row[2] for row in db_data] 
        if sum(values) == 0: return
        vw = tk.Toplevel(self.root); vw.title(self.texts["viz_title"]); vw.geometry("600x500")
        apply_dark_theme(vw)
        fig = Figure(figsize=(5,4), dpi=100, facecolor=C_BG)
        ax = fig.add_subplot(111)
        ax.pie(values, labels=tickers, autopct='%1.1f%%', startangle=90, textprops={'color':"w"})
        ax.set_title(self.texts["viz_title"], color="white")
        cv = FigureCanvasTkAgg(fig, master=vw)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def abrir_calculadora(self):
        cw = tk.Toplevel(self.root); cw.title(self.texts["calc_title"]); cw.geometry("250x350")
        apply_dark_theme(cw)
        ttk.Label(cw, text=self.texts["calc_cap"]).pack(pady=2); e_cap = ttk.Entry(cw); e_cap.pack(); e_cap.insert(0, "10000") 
        ttk.Label(cw, text=self.texts["calc_risk"]).pack(pady=2); e_risk = ttk.Entry(cw); e_risk.pack(); e_risk.insert(0, "1") 
        ttk.Label(cw, text="Entry ($):").pack(pady=2); e_ent = ttk.Entry(cw); e_ent.pack()
        if self.e_pr.get(): e_ent.insert(0, self.e_pr.get())
        ttk.Label(cw, text=self.texts["calc_stop"]).pack(pady=2); e_stop = ttk.Entry(cw); e_stop.pack()
        lbl_res = ttk.Label(cw, text="---", font=("bold", 12), foreground=C_ACCENT); lbl_res.pack(pady=10)
        def calcular():
            try:
                cap = float(e_cap.get()); r_pct = float(e_risk.get()); ent = float(e_ent.get()); stop = float(e_stop.get())
                if ent <= stop: lbl_res.configure(text="Entry <= Stop!", foreground=C_RED); return
                risk_amt = cap * (r_pct/100); diff = ent - stop; qty = int(risk_amt / diff)
                lbl_res.configure(text=f"{qty} Shares", foreground=C_GREEN); return qty
            except: lbl_res.configure(text="Error", foreground=C_RED)
        def aplicar():
            qty = calcular()
            if qty and qty > 0:
                self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qty)); self.e_pr.delete(0, tk.END); self.e_pr.insert(0, e_ent.get()); cw.destroy()
        ctk.CTkButton(cw, text=self.texts["calc_btn"], command=calcular, bg="#333", fg="white", relief="flat", corner_radius=8).pack(fill=tk.X, padx=20, pady=2)
        ctk.CTkButton(cw, text=self.texts["calc_apply"], command=aplicar, bg=C_GREEN, fg="black", relief="flat", corner_radius=8).pack(fill=tk.X, padx=20, pady=5)

    def exportar_cartera(self):
        datos = self.db.obtener_cartera(self.uid)
        if not datos: return
        df_export = pd.DataFrame(datos, columns=['Ticker', 'Precio Compra', 'Cantidad', 'Fecha', 'ID_DB']).drop(columns=['ID_DB'])
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Exportar")
        if f:
            try: df_export.to_csv(f, index=False, encoding='utf-8-sig'); messagebox.showinfo("OK", self.texts["msg_exp_ok"])
            except Exception as e: messagebox.showerror("Error", str(e))

    def refresh_all(self): self.run(); self.scan_own(); self.scan_mkt()

    def abrir_config(self):
        cw = tk.Toplevel(self.root); cw.title(self.texts["conf_title"]); cw.geometry("300x300")
        apply_dark_theme(cw)
        ttk.Label(cw, text=self.texts["conf_lang"], font=("bold", 10)).pack(pady=10)
        cb = ttk.Combobox(cw, values=["ES", "EN", "FR", "PT"], state="readonly")
        cb.set(self.current_lang); cb.pack(pady=5)
        def change_lang(e): self.current_lang = cb.get(); self.texts = LANG[self.current_lang]; self.update_ui_language(); cw.title(self.texts["conf_title"])
        cb.bind("<<ComboboxSelected>>", change_lang)
        ttk.Separator(cw, orient='horizontal').pack(fill='x', pady=20)
        def realizar_logout(): cw.destroy(); self.logout()
        ctk.CTkButton(cw, text=self.texts["conf_logout"], bg="#333", fg="white", font=("bold", 10), relief="flat", command=realizar_logout, corner_radius=8).pack(pady=5, fill=tk.X, padx=20)
        ctk.CTkButton(cw, text=self.texts["conf_del"], bg="#800000", fg="white", font=("bold", 10), relief="flat", command=self.borrar_cuenta, corner_radius=8).pack(pady=5, fill=tk.X, padx=20)

    def update_ui_language(self):
        t = self.texts
        self.root.title(f"{t['app_title']} - {self.uid}")
        
        # Títulos
        if hasattr(self, 'lbl_port_title'): self.lbl_port_title.configure(text=t["port_title"])
        
        # Botones Principales
        self.btn_act.configure(text=t["scan_own"])
        self.btn_gem.configure(text=t["scan_mkt"])
        self.btn_run.configure(text=t["analyze"])
        self.btn_refresh.configure(text=t["refresh_all"])
        
        # Botones Herramientas (Solo si quieres que cambien de texto, aunque usan iconos mayormente)
        # self.btn_stats.configure(text="📈") # Los iconos no necesitan traducción
        # self.btn_viz.configure(text="📊")
        
        # Botones con Texto
        self.btn_snap.configure(text=t["snap_btn"])
        self.b_rst.configure(text=t["reset_zoom"])
        
        # Dashboard
        self.lbl_invested.configure(text=f"{t['dash_inv']} ---")
        self.lbl_current.configure(text=f"{t['dash_val']} ---")
        self.lbl_pl.configure(text=f"{t['dash_pl']} ---")
        
        # Tablas
        self.tr1.heading("tk", text=t["col_ticker"])
        self.tr1.heading("pr", text=t["col_entry"])
        self.tr1.heading("sg", text=t["col_state"])
        self.tr2.heading("tk", text=t["col_ticker"])
        self.tr2.heading("sc", text=t["col_score"])
        self.tr2.heading("ms", text=t["col_diag"])
        
        self.load_init()

    def logout(self):
        # Destruimos todos los hijos de la raíz (limpiamos la pantalla)
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Volvemos a cargar el Login en la misma raíz
        LoginWindow(self.root, self.db, lambda u, n: AppBolsa(self.root, u, n, self.db))

    def borrar_cuenta(self):
        if messagebox.askyesno("!!!", self.texts["conf_del_confirm"]): self.db.borrar_usuario_completo(self.uid); self.root.destroy()

    def load_init(self):
        for i in self.tr1.get_children(): self.tr1.delete(i)
        for d in self.db.obtener_cartera(self.uid):
            pr = f"${d[1]}" if d[1]>0 else self.texts["vigil"]
            self.tr1.insert("", "end", iid=d[4], values=(d[0], pr, "..."))

    def limpiar_campos(self): self.e_pr.delete(0, tk.END); self.e_qt.delete(0, tk.END)

    def scan_own(self):
        # CAMBIO: .configure en lugar de .config
        self.btn_act.configure(state="disabled", text=self.texts["msg_wait"])
        threading.Thread(target=self._th_own, daemon=True).start()

    def _th_own(self):
        total_inv = 0.0; total_val = 0.0
        db_data = self.db.obtener_cartera(self.uid) 
        for row in db_data:
            tkr = row[0]; buy_p = row[1]; qty = row[2]; iid = row[4]
            total_inv += (buy_p * qty)
            try:
                self.eng.descargar_datos(tkr); self.eng.calcular_indicadores()
                res = self.eng.generar_diagnostico_interno(buy_p)
                self.root.after(0, self.update_row_ui, iid, tkr, f"${buy_p}", res["msg"], res["tag"])
                curr_p = res.get("price", buy_p); total_val += (curr_p * qty)
            except: total_val += (buy_p * qty)
        
        self.root.after(0, self.update_dashboard_ui, total_inv, total_val)
        # CAMBIO: .configure para volver al estado normal
        self.root.after(0, lambda: self.btn_act.configure(state="normal", text=self.texts["scan_own"]))

    def update_dashboard_ui(self, inv, val):
        pl = val - inv - (ETORO_ROUND_TRIP * len(self.db.obtener_cartera(self.uid)))
        pl_pct = (pl / inv * 100) if inv > 0 else 0.0
        
        # Elegir color
        color = "#4ec9b0" if pl >= 0 else "#f44747"
        
        # Actualizar etiquetas con .configure y text_color
        self.lbl_invested.configure(text=f"{self.texts['dash_inv']} ${inv:,.2f}")
        self.lbl_current.configure(text=f"{self.texts['dash_val']} ${val:,.2f}")
        self.lbl_pl.configure(
            text=f"{self.texts['dash_pl']} ${pl:,.2f} ({pl_pct:+.2f}%)", 
            text_color=color
        )

    def update_row_ui(self, iid, tkr, pr_text, msg, tag):
        if self.tr1.exists(iid): self.tr1.item(iid, values=(tkr, pr_text, msg), tags=(tag,))

    def scan_mkt(self):
        # CAMBIO: .configure
        self.btn_gem.configure(state="disabled", text=self.texts["msg_scan"])
        for i in self.tr2.get_children(): self.tr2.delete(i)
        threading.Thread(target=self._th_mkt, daemon=True).start()

    def _th_mkt(self):
        cands = []
        for tkr in CANDIDATOS_VIP:
            try:
                self.eng.descargar_datos(tkr); self.eng.calcular_indicadores()
                res = self.eng.generar_diagnostico_interno(0)
                if res["valid"] and res["score"] > 30: cands.append(res)
            except: pass
        cands.sort(key=lambda x: x["score"], reverse=True)
        for c in cands:
            final_msg = c["msg"]
            if c["score"] > 60:
                final_msg += f" | 🎯 ${c['target']:.2f} (+{c['profit']:.1f}%)"
            self.root.after(0, self.tr2.insert, "", "end", values=(c["ticker"], c["score"], final_msg), tags=(c["tag"],))
        
        # CAMBIO: .configure
        self.root.after(0, lambda: self.btn_gem.configure(state="normal", text=self.texts["scan_mkt"]))

    def sel_load(self, tree, is_own):
        s = tree.selection(); 
        if not s: return
        vals = tree.item(s[0])['values']; tkr = vals[0]; pc = 0; qt = 0
        if is_own:
            tid = int(s[0])
            for d in self.db.obtener_cartera(self.uid):
                if d[4] == tid: pc=d[1]; qt=d[2]; break
        self.e_tk.delete(0, tk.END); self.e_tk.insert(0, tkr)
        self.e_pr.delete(0, tk.END); self.e_pr.insert(0, str(pc))
        self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qt))
        self.run()

    def run(self):
        tkr = self.e_tk.get().strip().upper()
        if not tkr: return
        
        # Obtener parámetros de entrada (Precio y Cantidad)
        pp = 0.0; qq = 0.0; pos = False
        try:
            if self.e_pr.get(): pp = float(self.e_pr.get())
            if self.e_qt.get(): qq = float(self.e_qt.get())
            pos = (pp > 0)
        except: pass
        
        # Cambiar cursor a "reloj" mientras procesa
        self.root.configure(cursor="watch")
        self.root.update()

        try:
            # 1. ANÁLISIS DE DATOS
            self.eng.descargar_datos(tkr)
            df = self.eng.calcular_indicadores()
            
            # IA y Métricas
            prob_ai, acc, factors = self.eng.calcular_probabilidad_ia()
            spy = self.eng.obtener_benchmark()
            fund = self.eng.obtener_fundamentales(tkr)
            ws_rec, ws_target = self.eng.obtener_consenso_analistas(tkr)
            fibo = self.eng.calcular_fibonacci()
            noticias = self.eng.obtener_noticias_analizadas(tkr)
            sop, resi = self.eng.detectar_niveles()
            sim = self.eng.simular()
            ev = self.eng.generar_diagnostico_interno(pp)
            bench_stats = self.eng.calcular_beta_relativa(df, spy) if spy is not None else {"beta": 0, "rel_perf": 0}
            
            # 2. GRAFICADO (MODO OSCURO PRO)
            self.fig.clear()
            # Fondo exacto de la UI
            self.fig.patch.set_facecolor('#1e1e1e')
            
            gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
            ax1 = self.fig.add_subplot(gs[0])
            ax2 = self.fig.add_subplot(gs[1], sharex=ax1)
            ax3 = self.fig.add_subplot(gs[2], sharex=ax1)
            
            self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, hspace=0.15)
            
            # Estilos comunes para todos los ejes
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white', which='both')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('#1e1e1e') 
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('#1e1e1e')
                ax.grid(True, alpha=0.1, color='white', linestyle='--')

            d = df

            # --- PANEL 1: PRECIO ---
            ax1.plot(d.index, d['Close'], color='white', linewidth=1.5, label='Precio')
            ax1.plot(d.index, d['SMA_50'], color='#00ffff', linestyle='--', linewidth=1, label='SMA 50 (Cyan)') # Cyan
            ax1.plot(d.index, d['SMA_200'], color='#ff00ff', linewidth=1.5, label='SMA 200 (Magenta)') # Magenta
            ax1.fill_between(d.index, d['UpperBB'], d['LowerBB'], color='#1f6aa5', alpha=0.15, label='Bandas')
            
            if fibo:
                ax1.axhline(fibo['0.382'], color='#ffd700', linestyle=':', alpha=0.5)
                ax1.axhline(fibo['0.618'], color='#ffd700', linestyle=':', alpha=0.5)

            if sop > 0: ax1.axhline(sop, color='#4ec9b0', linestyle='--', linewidth=0.8, alpha=0.8) # Soporte Verde
            if resi > 0: ax1.axhline(resi, color='#f44747', linestyle='--', linewidth=0.8, alpha=0.8) # Resistencia Roja
            
            if pos: ax1.axhline(pp, color='#e67e22', linewidth=2, label='TU ENTRADA')

            ax1.set_title(f"{tkr} - {fund['sec']}", fontsize=12, color="white", fontweight="bold")
            legend = ax1.legend(fontsize=8, facecolor='#2b2b2b', edgecolor="white", labelcolor="white", loc='upper left')
            legend.get_frame().set_alpha(0.8)

            # --- PANEL 2: RSI + ADX ---
            ax2.plot(d.index, d['CRSI'], color='#1f6aa5', linewidth=1.5, label="RSI")
            ax2.plot(d.index, d['ADX'], color='gray', linewidth=0.8, alpha=0.7, label="ADX")
            ax2.axhline(80, color='#f44747', linestyle=':', alpha=0.5)
            ax2.axhline(20, color='#4ec9b0', linestyle=':', alpha=0.5)
            ax2.set_ylabel("RSI / ADX", fontsize=8, color="white")
            ax2.set_ylim(-5, 105)

            # --- PANEL 3: VOLUMEN OSC ---
            # Colores condicionales (Verde si sube volumen, Rojo si baja)
            colors_vol = np.where(d['Vol_Osc'] > 0, '#4ec9b0', '#f44747')
            ax3.bar(d.index, d['Vol_Osc'], color=colors_vol, width=1.0, alpha=0.8)
            ax3.axhline(0, color='white', linewidth=0.5)
            ax3.set_ylabel("Vol%", fontsize=8, color="white")

            # Limpiar ejes X duplicados
            ax1.tick_params(axis='x', labelbottom=False)
            ax2.tick_params(axis='x', labelbottom=False)
            ax3.tick_params(axis='x', rotation=45)

            self.cv.draw()
            self.b_rst.configure(state="normal") # Reactivar botón reset

            # 3. TEXTO DE DIAGNÓSTICO (ESTILIZADO)
            self.txt.delete(1.0, tk.END)
            self.txt.insert(tk.END, f"📊 ANÁLISIS: {tkr}\n", "t")
            self.txt.insert(tk.END, f"Precio: ${d['Close'].iloc[-1]:.2f} | ", "w")
            
            # P/L de la posición
            if pos:
                gross_pl = (d['Close'].iloc[-1] * qq) - (pp * qq)
                net_pl = gross_pl - ETORO_ROUND_TRIP
                pc = (net_pl / (pp * qq)) * 100
                tag = "p" if net_pl >= 0 else "n"
                self.txt.insert(tk.END, f"P&L: {net_pl:+.2f} ({pc:+.2f}%)\n", tag)
            else:
                self.txt.insert(tk.END, "\n")

            self.txt.insert(tk.END, f"\n🎯 OBJETIVO & DIAGNÓSTICO:\n", "gold")
            self.txt.insert(tk.END, f"• {ev['msg']} (Score: {ev['score']}/100)\n", "t")
            self.txt.insert(tk.END, f"• Target: ${ev['target']:.2f} (+{ev['profit']:.1f}%)\n", "p")
            self.txt.insert(tk.END, f"• Stop Loss Sugerido: ${ev['stop_loss']:.2f}\n", "n")

            self.txt.insert(tk.END, f"\n🤖 PREDICCIÓN IA (Gradient Boosting):\n", "w")
            ai_tag = "ai_good" if prob_ai > 55 else "ai_bad" if prob_ai < 45 else "w"
            self.txt.insert(tk.END, f"• Probabilidad Subida (3d): {prob_ai:.1f}%\n", ai_tag)
            
            if factors:
                self.txt.insert(tk.END, "• Factores Clave: ", "w")
                for f, imp in factors:
                    self.txt.insert(tk.END, f"{f} ", "gold")
                self.txt.insert(tk.END, "\n")

            self.txt.insert(tk.END, f"\n🏢 FUNDAMENTALES:\n", "gold")
            self.txt.insert(tk.END, f"• Wall St: {ws_rec} (Obj: ${ws_target:.2f})\n", "w")
            self.txt.insert(tk.END, f"• Valor Graham: ${fund['graham']:.2f}\n", "p" if fund['graham'] > ev['price'] else "n")
            self.txt.insert(tk.END, f"• Beta: {bench_stats['beta']:.2f} (vs SPY)\n", "w")

            self.txt.insert(tk.END, f"\n📰 NOTICIAS RECIENTES:\n", "w")
            if noticias:
                for n in noticias:
                    tag = "news_bull" if n['sentiment'] == 'bull' else "news_bear" if n['sentiment'] == 'bear' else "w"
                    prefix = "🟢" if n['sentiment'] == 'bull' else "🔴" if n['sentiment'] == 'bear' else "⚪"
                    self.txt.insert(tk.END, f"{prefix} {n['title'][:50]}...\n", tag)
            else:
                self.txt.insert(tk.END, "(Sin noticias relevantes)\n", "w")

            # Forecast simple
            self.txt.insert(tk.END, "\n🔮 SIMULACIÓN 7 DÍAS:\n", "t")
            hoy = datetime.date.today()
            prev = d['Close'].iloc[-1]
            for i in range(5):
                v = np.mean(sim[i])
                c = "p" if v > prev else "n"
                self.txt.insert(tk.END, f"{(hoy + datetime.timedelta(days=i+1)).strftime('%d/%m')}: ${v:.2f}\n", c)
                prev = v

        except Exception as e:
            messagebox.showerror("Error de Análisis", f"No se pudo analizar {tkr}.\nDetalle: {str(e)}")
        
        finally:
            self.root.configure(cursor="") # Restaurar cursor

if __name__ == "__main__":
    # Configuración global de CustomTkinter
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    db = DatabaseManager()
    
    # Creamos la ventana PRINCIPAL una sola vez
    root = ctk.CTk()
    root.title("Quant Architect v26")
    root.geometry("1600x950")
    
    # Lanzamos el Login dentro de la ventana principal
    # Nota: Ya no usamos root.withdraw() ni root.deiconify()
    LoginWindow(root, db, lambda u, n: AppBolsa(root, u, n, db))
    
    root.mainloop()