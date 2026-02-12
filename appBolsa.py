import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sqlite3
import hashlib
import threading
import matplotlib.pyplot as plt
import os

# --- LIBRERIAS IA ---
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score
except ImportError:
    pass # Se maneja si no esta instalado

# ==========================================
# 0. CONFIGURACIÓN E IDIOMAS
# ==========================================
ETORO_FEE_PER_ORDER = 1.0  
ETORO_ROUND_TRIP = 2.0     

LANG = {
    "ES": {
        "app_title": "Gestor Pro v47.0 (Color Update)",
        "port_title": "📂 MI CARTERA",
        "opp_title": "💎 OPORTUNIDADES",
        "scan_own": "⚡ ACTUALIZAR CARTERA",
        "save": "💾 GUARDAR", "sell": "💰 VENDER", "del_btn": "🗑 BORRAR", 
        "viz_btn": "📊 VIZ", "stats_btn": "📈 STATS", "risk_btn": "🔥 RIESGO", "snap_btn": "📷 FOTO",
        "col_ticker": "Ticker", "col_entry": "Entrada", "col_state": "Estado", "col_score": "Score", "col_diag": "Diagnóstico / Objetivo",
        "vigil": "👁 VIGILANDO", "msg_wait": "⏳...", "msg_scan": "⏳ ESCANEANDO...", 
        "msg_load_err": "⚠️ Error DB: ", "msg_db_ok": "✅ DB Conectada", 
        "msg_exp_ok": "✅ Exportado.", "msg_snap_ok": "✅ Foto guardada.",
        "msg_del_confirm": "¿Borrar posición?", "msg_sell_title": "Vender", "msg_sell_ask": "Precio de venta:",
        "fund_title": "📊 FUNDAMENTALES & MACRO:", 
        "graham_title": "💎 VALORACIÓN (Confluencia):", 
        "ai_title": "🤖 PREDICCIÓN IA:", "ai_prob": "Prob. Éxito:", "ai_acc": "Precisión Test:", "ai_factors": "Factores:",
        "tech_title": "📐 TECH & STOP LOSS:", 
        "target_title": "🎯 OBJETIVO:", 
        "calc_title": "Gestión Monetaria (Kelly)", 
        "calc_cap": "Capital Total ($):",
        "calc_risk": "Riesgo Máx (%):",
        "calc_stop": "Stop Loss ($):",
        "calc_btn": "CALCULAR LOTE", "calc_apply": "APLICAR A CONTROL",
        "dash_inv": "Invertido:", "dash_val": "Valor:", "dash_pl": "Neto P/L:",
        "macro_fear": "😨 MIEDO", "macro_greed": "🤑 CODICIA", "macro_neutral": "😐 NEUTRO",
        "login_title": "ACCESO", "user": "Usuario:", "pass": "Clave:", 
        "btn_enter": "ENTRAR", "btn_reg": "REGISTRO", 
        "err_login": "Error de usuario/clave", "ok_reg": "Usuario registrado", "err_reg": "Usuario ya existe",
        "db_select_title": "Base de Datos", "db_select_msg": "Selecciona tu archivo .db",
        "hist_title": "Historial Operaciones", "hist_tot": "Total P/L:", 
        "stats_title": "Estadísticas", "risk_title": "Matriz Correlación",
        "viz_title": "Distribución Cartera", 
        "scan_mkt": "🔍 ESCANEAR MERCADO", "analyze": "▶ ANALIZAR", "reset_zoom": "RESET",
        "buy_price": "Compra:", "qty": "Cant:", "refresh_all": "🔄 TODO",
        "hist": "📜 HISTORIAL", "exp": "📄 EXPORTAR",
        "ws_rating": "Analistas:", "ws_target": "Obj. Analistas:",
        "tech_sup": "Soporte:", "tech_res": "Resistencia:", "tech_sl": "Stop Loss Sugerido:",
        "bench_title": "COMPARATIVA S&P500:", "bench_beta": "Beta:", "bench_rel": "Rend. Relativo:",
        "news_title": "NOTICIAS & SENTIMIENTO:", "trend_wk": "Tendencia Semanal:"
    },
    "EN": { 
        "app_title": "Pro Manager v47.0", 
        "port_title": "📂 PORTFOLIO", "opp_title": "💎 OPPORTUNITIES", "scan_own": "⚡ REFRESH", "save": "💾 SAVE", "sell": "💰 SELL", "del_btn": "🗑 DEL", "viz_btn": "📊 VIZ", "stats_btn": "📈 STATS", "risk_btn": "🔥 RISK", "snap_btn": "📷 SNAP", 
        "col_ticker": "Ticker", "col_entry": "Entry", "col_state": "Status", "col_score": "Score", "col_diag": "Diagnosis", 
        "vigil": "👁 WATCH", "msg_wait": "⏳...", "msg_scan": "⏳ SCANNING...", "msg_load_err": "⚠️ DB Error: ", "msg_db_ok": "✅ DB Connected", 
        "msg_exp_ok": "✅ Exported.", "msg_snap_ok": "✅ Snap saved.",
        "msg_del_confirm": "Delete pos?", "msg_sell_title": "Sell", "msg_sell_ask": "Sell Price:",
        "fund_title": "📊 FUNDAMENTALS:", "graham_title": "💎 VALUATION:", "ai_title": "🤖 AI PREDICTION:", "ai_prob": "Win Prob:", "ai_acc": "Accuracy:", "ai_factors": "Factors:",
        "tech_title": "📐 TECH:", "target_title": "🎯 TARGET:", 
        "calc_title": "Kelly Money Manager", "calc_cap": "Total Capital:", "calc_kelly": "Kelly Suggestion:", "calc_risk_reward": "Risk/Reward Ratio:", "calc_btn": "CALCULATE", "calc_apply": "APPLY", 
        "login_title": "LOGIN", "user": "User:", "pass": "Pass:", "btn_enter": "GO", "btn_reg": "REG", "err_login": "Error", "ok_reg": "OK", "err_reg": "Exists", "macro_fear": "FEAR", "macro_greed": "GREED", "macro_neutral": "NEUTRAL",
        "dash_inv": "Invested:", "dash_val": "Value:", "dash_pl": "Net P/L:", "db_select_title": "Database", "db_select_msg": "Select .db file",
        "hist_title": "Trade History", "hist_tot": "Total P/L:", "stats_title": "Stats", "risk_title": "Correlation", "viz_title": "Allocation", 
        "scan_mkt": "🔍 SCAN MARKET", "analyze": "▶ ANALYZE", "reset_zoom": "RESET", "buy_price": "Price:", "qty": "Qty:", "refresh_all": "🔄 ALL",
        "hist": "📜 HISTORY", "exp": "📄 EXPORT",
        "ws_rating": "Analysts:", "ws_target": "Target:", "tech_sup": "Support:", "tech_res": "Resistance:", "tech_sl": "Stop Loss:",
        "bench_title": "VS S&P500:", "bench_beta": "Beta:", "bench_rel": "Rel. Perf:", "news_title": "NEWS:", "trend_wk": "Weekly Trend:"
    }
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
# 1. BASE DE DATOS
# ==========================================
class DatabaseManager:
    def __init__(self, db_name="bolsa_datos_v24.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.crear_tablas()

    def crear_tablas(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS cartera (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, ticker TEXT NOT NULL, 
                precio_compra REAL, cantidad REAL, fecha_guardado TEXT, FOREIGN KEY(user_id) REFERENCES usuarios(id))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, ticker TEXT, 
                buy_price REAL, sell_price REAL, qty REAL, profit REAL, date_out TEXT)''')
            self.conn.commit()
        except Exception as e: print(f"DB Error: {e}")

    def registrar_usuario(self, u, p):
        ph = hashlib.sha256(p.encode()).hexdigest()
        try:
            self.conn.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", (u, ph))
            self.conn.commit(); return True
        except: return False

    def verificar_usuario(self, u, p):
        ph = hashlib.sha256(p.encode()).hexdigest()
        res = self.conn.execute("SELECT id FROM usuarios WHERE username=? AND password=?", (u, ph)).fetchone()
        return res[0] if res else None

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
# 2. MOTOR ANALÍTICO
# ==========================================
class AnalistaBolsa:
    def __init__(self):
        self.data = None; self.ticker = ""; self.data_weekly = None
        self.spy_data = None 
        self.macro_data = {"vix": 0, "tnx": 0, "trend": "NEUTRAL", "tnx_chg": 0} 

    def descargar_datos(self, ticker):
        self.ticker = ticker.upper()
        try:
            d = yf.download(self.ticker, period="2y", progress=False)
            if d.empty: raise ValueError
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.droplevel(1)
            self.data = d.astype(float)
            try:
                w = yf.download(self.ticker, period="2y", interval="1wk", progress=False)
                if isinstance(w.columns, pd.MultiIndex): w.columns = w.columns.droplevel(1)
                self.data_weekly = w.astype(float)
            except: self.data_weekly = None
            return d
        except: raise ValueError("Error descarga")

    def obtener_sentimiento_mercado(self):
        try:
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            spy = yf.Ticker("SPY").history(period="6mo")['Close']
            sma50 = spy.rolling(50).mean().iloc[-1]; price_spy = spy.iloc[-1]
            if vix > 25: return "FEAR", vix
            elif vix < 15 and price_spy > sma50: return "GREED", vix
            else: return "NEUTRAL", vix
        except: return "NEUTRAL", 0

    def obtener_consenso_analistas(self, ticker):
        try:
            t = yf.Ticker(ticker); i = t.info
            rec = i.get('recommendationKey', 'none').upper().replace("_", " ")
            target = i.get('targetMeanPrice', 0)
            return rec, target
        except: return "N/A", 0

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
            per = i.get('trailingPE', 0); cap = i.get('marketCap', 0); div = i.get('dividendYield', 0)
            sec = i.get('sector', 'N/A'); ind = i.get('industry', 'N/A')
            
            if cap > 1e12: s_cap = f"{cap/1e12:.2f}T"
            elif cap > 1e9: s_cap = f"{cap/1e9:.2f}B"
            else: s_cap = str(cap)
            g_val = self.calcular_valor_graham(ticker)
            return {
                "per": f"{per:.2f}" if per else "-", 
                "cap": s_cap, 
                "div": f"{div*100:.2f}%" if div else "0%", 
                "graham": g_val, 
                "sec": sec, 
                "ind": ind, 
                "valid": True
            }
        except: return {"per": "-", "cap": "-", "div": "0%", "graham": 0, "sec": "-", "ind": "-", "valid": False}

    def calcular_indicadores(self):
        if self.data is None or self.data.empty: return self.data
        for col in ['ADX', 'Vol_Osc', 'CRSI', 'SMA_50', 'SMA_200', 'MACD', 'UpperBB', 'LowerBB', 'ATR', 'VWAP']: self.data[col] = 0.0
        df = self.data.copy()
        df['SMA_50'] = df['Close'].rolling(50).mean(); df['SMA_200'] = df['Close'].rolling(200).mean()
        v = df['Volume'].values; tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
        df['SMA_20'] = df['Close'].rolling(20).mean(); df['StdDev'] = df['Close'].rolling(20).std()
        df['UpperBB'] = df['SMA_20'] + (df['StdDev'] * 2); df['LowerBB'] = df['SMA_20'] - (df['StdDev'] * 2)
        df['BB_Pct'] = (df['Close'] - df['LowerBB']) / (df['UpperBB'] - df['LowerBB'])
        exp12 = df['Close'].ewm(span=12, adjust=False).mean(); exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Prev'] = df['Close'].shift(1)
        df['TR'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Prev']), abs(df['Low']-df['Prev'])))
        df['ATR'] = df['TR'].rolling(14).mean(); tr14 = df['TR'].ewm(alpha=1/14).mean()
        up = df['High'] - df['High'].shift(1); down = df['Low'].shift(1) - df['Low']
        pdm = np.where((up>down)&(up>0), up, 0.0); mdm = np.where((down>up)&(down>0), down, 0.0)
        with np.errstate(all='ignore'):
            pdi = 100*(pd.Series(pdm, index=df.index).ewm(alpha=1/14).mean()/tr14)
            mdi = 100*(pd.Series(mdm, index=df.index).ewm(alpha=1/14).mean()/tr14)
            df['ADX'] = (100*abs(pdi-mdi)/(pdi+mdi)).ewm(alpha=1/14).mean()
        vol_ma = df['Volume'].rolling(10).mean().replace(0, np.nan)
        df['Vol_Osc'] = ((df['Volume'].rolling(5).mean()-vol_ma)/vol_ma)*100
        delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss; df['CRSI'] = 100 - (100 / (1 + rs))
        df = df.fillna(0); self.data = df
        return self.data

    def analizar_tendencia_semanal(self):
        try:
            if self.data_weekly is None or len(self.data_weekly) < 30: return "Neutral"
            last = self.data_weekly.iloc[-1]
            ma30 = self.data_weekly['Close'].rolling(30).mean().iloc[-1]
            if last['Close'] > ma30: return "Alcista"
            else: return "Bajista"
        except: return "Neutral"

    def calcular_probabilidad_ia(self):
        try:
            df = self.data.copy()
            if len(df) < 100: return 50.0, 0.0, []
            df['Retorno'] = df['Close'].pct_change(); df['Lag_1'] = df['Retorno'].shift(1)
            df = df.dropna()
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            features = ['CRSI', 'MACD', 'Retorno', 'Vol_Osc', 'Lag_1']
            X = df[features].iloc[:-1]; y = df['Target'].iloc[:-1]
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_scaled, y)
            last_day = df[features].iloc[[-1]]; last_day_scaled = scaler.transform(last_day)
            prob = model.predict_proba(last_day_scaled)[0][1] * 100
            importances = model.feature_importances_
            top_factors = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:2]
            return prob, 0.0, top_factors
        except: return 50.0, 0.0, []

    def generar_diagnostico_interno(self, p_compra=0, ia_prob=50, ws_rating="HOLD", graham_val=0):
        try:
            last = self.data.iloc[-1]
            adx = last.get('ADX', 0)
            crsi = last.get('CRSI', 50)
            curr = last['Close']
            atr = last.get('ATR', 0)
            
            # Cálculo de Score
            score = 50
            if crsi < 30: score += 15
            elif crsi > 70: score -= 15
            if ia_prob > 60: score += 15
            elif ia_prob < 40: score -= 10
            if graham_val > 0 and graham_val > curr: score += 10
            final_score = max(0, min(100, int(score)))
            
            # Cálculo de Objetivos
            max_p = self.data['High'].tail(60).max()
            target = max_p if curr < max_p * 0.98 else curr * 1.05
            
            # --- ARREGLO AQUÍ: Cálculo de profit potencial ---
            profit_pot = ((target - curr) / curr) * 100
            # ------------------------------------------------
            
            stop_loss = curr - (atr * 2) 
            
            msg = "Neutro"; tag = ""
            if p_compra > 0: 
                 if crsi > 80: msg = "⚠️ VENDER"; tag = "sell"
                 elif curr < p_compra*0.9: msg = "🛑 STOP"; tag = "sell"
                 elif final_score > 70: msg = "💎 ACUMULAR"; tag = "buy"
                 else: msg = "✋ MANTENER"; tag = "hold"
            else:
                 if final_score >= 80: msg = "🚀 FUERTE"; tag = "buy"
                 elif final_score >= 60: msg = "✅ COMPRA"; tag = "near"
                 elif final_score <= 30: msg = "⚠️ EVITAR"; tag = "sell"

            # Devolvemos el diccionario incluyendo la clave 'profit'
            return {
                "valid": True, 
                "ticker": self.ticker, 
                "score": final_score, 
                "msg": msg, 
                "tag": tag, 
                "price": curr, 
                "stop_loss": stop_loss, 
                "target": target, 
                "profit": profit_pot, # Clave añadida para evitar el error
                "ia_prob": ia_prob,
                "weekly": self.analizar_tendencia_semanal(),
                "adx": adx,
                "crsi": crsi
            }
        except: 
            return {"valid": False, "msg": "Error", "score": 0, "tag": "", "price": 0, "stop_loss": 0, "target": 0, "profit": 0, "ia_prob": 50}

    def simular(self, dias=7):
        try:
            ret = np.log(self.data['Close']/self.data['Close'].shift(1)).tail(60).dropna()
            if len(ret) < 2: return np.zeros((dias, 10))
            u = ret.mean(); v = ret.var(); drift = u - (0.5 * v); stdev = ret.std()
            daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (dias, 10)))
            price_paths = np.zeros_like(daily_returns)
            price_paths[0] = self.data['Close'].iloc[-1] * daily_returns[0]
            for t in range(1, dias): price_paths[t] = price_paths[t-1] * daily_returns[t]
            return price_paths
        except: return np.zeros((dias, 10))

    def detectar_niveles(self):
        try: df = self.data.tail(60); return df['Low'].min(), df['High'].max()
        except: return 0, 0
    
    def calcular_fibonacci(self):
        try:
            df = self.data.tail(126); max_p = df['High'].max(); min_p = df['Low'].min(); diff = max_p - min_p
            return {"0": min_p, "1": max_p, "0.382": max_p - 0.382*diff, "0.5": max_p - 0.5*diff, "0.618": max_p - 0.618*diff}
        except: return None
        
    def obtener_benchmark(self):
        if self.spy_data is not None: return self.spy_data
        try: d = yf.download("SPY", period="2y", progress=False); self.spy_data = d.astype(float); return self.spy_data
        except: return None
        
    def calcular_beta_relativa(self, stock_df, spy_df):
        try:
            df = pd.DataFrame({'STOCK': stock_df['Close'], 'SPY': spy_df['Close']}).dropna()
            if len(df) < 50: return {"beta": 0, "rel_perf": 0}
            rets = df.pct_change().dropna(); cov = rets['STOCK'].cov(rets['SPY']); var = rets['SPY'].var()
            beta = cov / var if var != 0 else 1.0; lookback = min(126, len(df))
            stock_ret = (df['STOCK'].iloc[-1] / df['STOCK'].iloc[-lookback]) - 1
            spy_ret = (df['SPY'].iloc[-1] / df['SPY'].iloc[-lookback]) - 1
            return {"beta": beta, "rel_perf": (stock_ret - spy_ret) * 100}
        except: return {"beta": 0, "rel_perf": 0}
        
    def obtener_noticias_analizadas(self, ticker): return []

# ==========================================
# 3. GUI THEME MANAGER
# ==========================================
def apply_dark_theme(root):
    style = ttk.Style(root); style.theme_use('clam')
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
        self.root = root; self.db = db; self.on_success = on_success
        self.win = tk.Toplevel(root); self.win.title("Acceso v47.0"); self.win.geometry("350x300")
        apply_dark_theme(self.win)
        self.texts = LANG["ES"]
        ttk.Label(self.win, text=self.texts["login_title"], font=("Segoe UI", 16, "bold"), foreground=C_ACCENT).pack(pady=30)
        f = ttk.Frame(self.win); f.pack(pady=10)
        ttk.Label(f, text=self.texts["user"]).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.e_u = ttk.Entry(f); self.e_u.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(f, text=self.texts["pass"]).grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.e_p = ttk.Entry(f, show="*"); self.e_p.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.win, text=self.texts["btn_enter"], command=self.log, bg=C_ACCENT, fg="white", font=("Segoe UI", 10, "bold"), relief="flat").pack(pady=15, fill=tk.X, padx=40)
        tk.Button(self.win, text=self.texts["btn_reg"], command=self.reg, bg="#333", fg="white", relief="flat").pack(pady=5, fill=tk.X, padx=40)
        self.win.protocol("WM_DELETE_WINDOW", root.destroy)

    def log(self):
        user_val = self.e_u.get(); pass_val = self.e_p.get()
        uid = self.db.verificar_usuario(user_val, pass_val)
        if uid: self.win.destroy(); self.on_success(uid, user_val)
        else: messagebox.showerror("Error", self.texts["err_login"])

    def reg(self):
        if self.db.registrar_usuario(self.e_u.get(), self.e_p.get()): messagebox.showinfo("OK", self.texts["ok_reg"])
        else: messagebox.showerror("Error", self.texts["err_reg"])

class AppBolsa:
    def __init__(self, root, uid, uname, db):
        self.root = root; self.uid = uid; self.db = db; self.eng = AnalistaBolsa()
        self.current_lang = "ES"; self.texts = LANG[self.current_lang]
        self.current_analysis = None 
        
        self.root.geometry("1600x950")
        apply_dark_theme(root)
        
        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL); main.pack(fill=tk.BOTH, expand=True)
        self.side = ttk.Frame(main, width=500, relief=tk.FLAT); main.add(self.side, weight=1)
        cont = ttk.Frame(main); main.add(cont, weight=4)

        header_frame = ttk.Frame(self.side); header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.lbl_mood = ttk.Label(header_frame, text="MERCADO: ⏳ ...", font=("Segoe UI", 10, "bold"), foreground="gray")
        self.lbl_mood.pack(anchor="center", pady=2)
        
        dash_frame = ttk.Frame(self.side); dash_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)
        self.lbl_invested = ttk.Label(dash_frame, text="---", font=("Segoe UI", 9)); self.lbl_invested.pack(anchor="w")
        self.lbl_current = ttk.Label(dash_frame, text="---", font=("Segoe UI", 9)); self.lbl_current.pack(anchor="w")
        self.lbl_pl = ttk.Label(dash_frame, text="---", font=("Segoe UI", 9)); self.lbl_pl.pack(anchor="w")

        # 1. LISTA CARTERA (PRIMERO, ARRIBA)
        f_port = ttk.LabelFrame(self.side, text=self.texts["port_title"])
        f_port.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        cols = ("tk", "pr", "sg")
        self.tr1 = ttk.Treeview(f_port, columns=cols, show="headings", height=8)
        for c in cols: self.tr1.column(c, anchor="center", stretch=True)
        self.tr1.heading("tk", text=self.texts["col_ticker"]); self.tr1.heading("pr", text=self.texts["col_entry"]); self.tr1.heading("sg", text=self.texts["col_score"])
        self.tr1.pack(fill=tk.BOTH, expand=True)
        self.tr1.bind("<Double-1>", lambda e: self.sel_load(self.tr1, True))

        # 2. BOTONERA (EN MEDIO)
        f_controls = ttk.LabelFrame(self.side, text="Acciones", padding=5)
        f_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.btn_act = tk.Button(f_controls, text=self.texts["scan_own"], bg=C_ACCENT, fg="white", relief="flat", command=self.scan_own)
        self.btn_act.pack(fill=tk.X, pady=1)
        
        fr2 = ttk.Frame(f_controls); fr2.pack(fill=tk.X, pady=1)
        tk.Button(fr2, text=self.texts["save"], bg="#333", fg="white", command=self.save).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(fr2, text=self.texts["sell"], bg="#800000", fg="white", command=self.vender_posicion).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(fr2, text=self.texts["del_btn"], bg="#333", fg="#999", command=self.dele).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        fr3 = ttk.Frame(f_controls); fr3.pack(fill=tk.X, pady=1)
        tk.Button(fr3, text=self.texts["hist"], bg="#333", fg="white", command=self.ver_historial).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(fr3, text=self.texts["stats_btn"], bg="#2e8b57", fg="white", command=self.ver_estadisticas).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(fr3, text=self.texts["risk_btn"], bg=C_ORANGE, fg="white", command=self.ver_correlaciones).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        fr4 = ttk.Frame(f_controls); fr4.pack(fill=tk.X, pady=1)
        tk.Button(fr4, text=self.texts["viz_btn"], bg=C_PURPLE, fg="white", command=self.ver_distribucion).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(fr4, text=self.texts["exp"], bg="#333", fg="white", command=self.exportar_cartera).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_var = tk.StringVar(); self.status_var.set("Listo.")
        ttk.Label(f_controls, textvariable=self.status_var, font=("Segoe UI", 8), foreground="gray").pack(side=tk.BOTTOM, fill=tk.X)

        # 3. LISTA OPORTUNIDADES (ABAJO)
        f_opp = ttk.LabelFrame(self.side, text=self.texts["opp_title"])
        f_opp.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.btn_gem = tk.Button(f_opp, text=self.texts["scan_mkt"], bg=C_ACCENT, fg="white", font=("Segoe UI", 9, "bold"), relief="flat", command=self.scan_mkt)
        self.btn_gem.pack(side=tk.TOP, fill=tk.X)
        
        cols2 = ("tk", "sc", "ms")
        self.tr2 = ttk.Treeview(f_opp, columns=cols2, show="headings")
        for c in cols2: self.tr2.column(c, anchor="center", stretch=True)
        self.tr2.heading("tk", text=self.texts["col_ticker"]); self.tr2.heading("sc", text=self.texts["col_score"]); self.tr2.heading("ms", text=self.texts["col_diag"])
        self.tr2.column("ms", width=200)
        self.tr2.pack(fill=tk.BOTH, expand=True)
        self.tr2.bind("<Double-1>", lambda e: self.sel_load(self.tr2, False))

        # === PANEL DERECHO ===
        ctrl = ttk.LabelFrame(cont, text=" Control ", padding=5); ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="TICKER:").pack(side=tk.LEFT)
        self.e_tk = ttk.Entry(ctrl, width=10); self.e_tk.pack(side=tk.LEFT, padx=5)
        self.e_tk.bind('<Return>', lambda e: self.run())
        self.btn_run = tk.Button(ctrl, text=self.texts["analyze"], command=self.run, bg="#eee", fg="black", relief="flat"); self.btn_run.pack(side=tk.LEFT)
        self.b_rst = tk.Button(ctrl, text=self.texts["reset_zoom"], command=self.zoom_rst, state="disabled", bg="#333", fg="white", relief="flat"); self.b_rst.pack(side=tk.LEFT, padx=5)
        self.lbl_buy = ttk.Label(ctrl, text=self.texts["buy_price"]); self.lbl_buy.pack(side=tk.LEFT)
        self.e_pr = ttk.Entry(ctrl, width=8); self.e_pr.pack(side=tk.LEFT)
        self.lbl_qty = ttk.Label(ctrl, text=self.texts["qty"]); self.lbl_qty.pack(side=tk.LEFT)
        self.e_qt = ttk.Entry(ctrl, width=8); self.e_qt.pack(side=tk.LEFT)
        tk.Button(ctrl, text="🗑", command=self.limpiar_campos, bg="#333", fg="white", relief="flat").pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="🧮", command=self.abrir_calculadora, bg="#333", fg="white", relief="flat").pack(side=tk.LEFT, padx=2)
        self.btn_snap = tk.Button(ctrl, text=self.texts["snap_btn"], bg="teal", fg="white", relief="flat", command=self.generar_reporte); self.btn_snap.pack(side=tk.LEFT, padx=5)
        self.btn_conf = tk.Button(ctrl, text="⚙️", bg="#333", fg="white", relief="flat", command=self.abrir_config); self.btn_conf.pack(side=tk.RIGHT, padx=5)
        self.btn_refresh = tk.Button(ctrl, text=self.texts["refresh_all"], bg="#8a2be2", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", command=self.refresh_all); self.btn_refresh.pack(side=tk.RIGHT, padx=5)

        pan = ttk.PanedWindow(cont, orient=tk.HORIZONTAL); pan.pack(fill=tk.BOTH, expand=True, padx=10)
        self.txt = tk.Text(pan, width=40, bg=C_PANEL, fg=C_FG, font=("Consolas", 10), borderwidth=0); pan.add(self.txt, weight=1)
        self.txt.tag_config("t", foreground=C_ACCENT, font=("Consolas", 11, "bold"))
        self.txt.tag_config("p", foreground=C_GREEN); self.txt.tag_config("n", foreground=C_RED)
        self.txt.tag_config("gold", foreground=C_GOLD); self.txt.tag_config("w", foreground="white")
        self.txt.tag_config("news_bull", foreground=C_GREEN); self.txt.tag_config("news_bear", foreground=C_RED)
        self.txt.tag_config("ai_good", foreground="#00ff00", font=("bold",12)); self.txt.tag_config("ai_bad", foreground="#ff4444", font=("bold",12))
        
        frg = ttk.Frame(pan)
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(5,5), dpi=100, facecolor=C_BG)
        self.cv = FigureCanvasTkAgg(self.fig, master=frg)
        self.tb = NavigationToolbar2Tk(self.cv, frg); self.tb.update()
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        pan.add(frg, weight=3)
        
        self.load_init()
        threading.Thread(target=self.actualizar_mood, daemon=True).start()

    # --- FUNCIONES DENTRO DE LA CLASE ---

    def generar_reporte(self):
        tkr = self.e_tk.get().strip()
        if not tkr: 
            messagebox.showwarning("Atención", "Introduce un ticker primero")
            return
        if not os.path.exists("Reports"): 
            os.makedirs("Reports")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"Reports/{tkr}_{timestamp}"
        
        try: 
            self.fig.savefig(f"{fname}.png")
        except: 
            pass
            
        try:
            content = self.txt.get("1.0", tk.END)
            with open(f"{fname}.txt", "w", encoding="utf-8") as f: 
                f.write(content)
            messagebox.showinfo("Snapshot", self.texts["msg_snap_ok"])
        except Exception as e: 
            messagebox.showerror("Error", str(e))

    def load_init(self):
        # CONFIGURACION DE COLORES (VERDE/ROJO)
        self.tr1.tag_configure('win', background='#163832', foreground='white')
        self.tr1.tag_configure('loss', background='#381616', foreground='white')
        self.tr1.tag_configure('neutral', background=C_PANEL, foreground='white')

        self.status_var.set(self.texts["msg_db_ok"])
        for i in self.tr1.get_children(): self.tr1.delete(i)
        try:
            for d in self.db.obtener_cartera(self.uid):
                pr = f"${d[1]}" if d[1]>0 else self.texts["vigil"]
                self.tr1.insert("", "end", iid=d[4], values=(d[0], pr, "..."), tags=('neutral',))
        except Exception as e:
            self.status_var.set(f"{self.texts['msg_load_err']} {e}")

    def actualizar_mood(self):
        mood, vix = self.eng.obtener_sentimiento_mercado()
        txt = ""; col = "gray"
        if mood == "FEAR": txt = f"{self.texts['macro_fear']} (VIX {vix:.2f})"; col = C_RED
        elif mood == "GREED": txt = f"{self.texts['macro_greed']} (VIX {vix:.2f})"; col = C_GREEN
        else: txt = f"{self.texts['macro_neutral']} (VIX {vix:.2f})"; col = C_GOLD
        self.lbl_mood.config(text=f"MERCADO: {txt}", foreground=col)

    def scan_own(self):
        self.btn_act.config(state="disabled")
        self.status_var.set(self.texts["msg_wait"])
        threading.Thread(target=self._th_own, daemon=True).start()

    def _th_own(self):
        total_inv = 0.0; total_val = 0.0
        db_data = self.db.obtener_cartera(self.uid)
        for row in db_data:
            tkr = row[0]; buy_p = row[1]; qty = row[2]; iid = row[4]
            total_inv += (buy_p * qty)
            try:
                self.root.after(0, lambda t=tkr: self.status_var.set(f"Analizando {t}..."))
                self.eng.descargar_datos(tkr); self.eng.calcular_indicadores()
                prob_ai, _, _ = self.eng.calcular_probabilidad_ia()
                res = self.eng.generar_diagnostico_interno(buy_p, prob_ai, "N/A", 0)
                
                curr_p = res.get("price", buy_p) if buy_p > 0 else 0
                
                # COLOR LOGIC
                row_tag = 'neutral'
                if buy_p > 0 and curr_p > 0:
                    if curr_p >= buy_p: row_tag = 'win'
                    else: row_tag = 'loss'

                self.root.after(0, self.update_row_ui, iid, tkr, f"${buy_p}", res["score"], res["msg"], row_tag)
                if buy_p > 0: total_val += (curr_p * qty)
            except: 
                if buy_p > 0: total_val += (buy_p * qty)
        self.root.after(0, self.update_dashboard_ui, total_inv, total_val)
        self.root.after(0, lambda: [self.btn_act.config(state="normal"), self.status_var.set("Listo.")])

    def scan_mkt(self):
        self.btn_gem.config(state="disabled")
        self.status_var.set(self.texts["msg_scan"])
        for i in self.tr2.get_children(): self.tr2.delete(i)
        threading.Thread(target=self._th_mkt, daemon=True).start()

    def _th_mkt(self):
        cands = []
        for tkr in CANDIDATOS_VIP:
            try:
                self.root.after(0, lambda t=tkr: self.status_var.set(f"Escaneando {t}..."))
                self.eng.descargar_datos(tkr); self.eng.calcular_indicadores()
                prob_ai, _, _ = self.eng.calcular_probabilidad_ia()
                res = self.eng.generar_diagnostico_interno(0, prob_ai, "N/A", 0)
                if res["valid"] and res["score"] > 40: cands.append(res)
            except: pass
        cands.sort(key=lambda x: x["score"], reverse=True)
        for c in cands:
            final_msg = c["msg"]
            if c["score"] > 60: final_msg += " | 🎯 Check full analysis"
            self.root.after(0, self.tr2.insert, "", "end", values=(c["ticker"], c["score"], final_msg), tags=(c["tag"],))
        self.root.after(0, lambda: [self.btn_gem.config(state="normal"), self.status_var.set("Listo.")])

    def run(self):
        tkr = self.e_tk.get().strip(); 
        if not tkr: return
        self.status_var.set(f"Analizando a fondo {tkr}...")
        self.root.update()
        
        pp=0; qq=0; pos=False
        if self.e_pr.get(): 
            try: pp=float(self.e_pr.get()); qq=float(self.e_qt.get()); pos=(pp>0)
            except: pass

        try:
            self.eng.descargar_datos(tkr); df = self.eng.calcular_indicadores()
            prob_ai, acc, factors = self.eng.calcular_probabilidad_ia()
            spy = self.eng.obtener_benchmark() 
            fund = self.eng.obtener_fundamentales(tkr)
            ws_rec, ws_target = self.eng.obtener_consenso_analistas(tkr) 
            fibo = self.eng.calcular_fibonacci() 
            noticias = self.eng.obtener_noticias_analizadas(tkr)
            sop, resi = self.eng.detectar_niveles()
            
            sim = self.eng.simular()

            self.current_analysis = self.eng.generar_diagnostico_interno(pp, prob_ai, ws_rec, fund['graham'])
            ev = self.current_analysis # Alias
            
            bench_stats = self.eng.calcular_beta_relativa(df, spy) if spy is not None else {"beta": 0, "rel_perf": 0}
            
            self.fig.clear()
            gs = self.fig.add_gridspec(3, 1, height_ratios=[3,1,1])
            ax1=self.fig.add_subplot(gs[0]); ax2=self.fig.add_subplot(gs[1], sharex=ax1); ax3=self.fig.add_subplot(gs[2], sharex=ax1)
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.15)
            
            d = df.tail(150)
            ax1.plot(d.index, d['Close'], color='white', linewidth=1.2, label='Precio') 
            ax1.plot(d.index, d['SMA_50'], color='orange', linestyle='--', linewidth=1, label='SMA 50')
            ax1.plot(d.index, d['SMA_200'], color='cyan', linewidth=1.5, label='SMA 200')
            ax1.fill_between(d.index, d['UpperBB'], d['LowerBB'], color='gray', alpha=0.15, label='BB')
            
            if fibo:
                ax1.axhline(fibo['0.382'], color=C_GOLD, linestyle=':', alpha=0.6, label="Fib 0.382")
                ax1.axhline(fibo['0.5'], color=C_GOLD, linestyle='--', alpha=0.6, label="Fib 0.5")
                ax1.axhline(fibo['0.618'], color=C_GOLD, linestyle=':', alpha=0.6, label="Fib 0.618")

            if sop > 0: ax1.axhline(sop, color=C_GREEN, linestyle='--', linewidth=0.8, alpha=0.7)
            if resi > 0: ax1.axhline(resi, color=C_RED, linestyle='--', linewidth=0.8, alpha=0.7)
            if ev['stop_loss'] > 0: ax1.axhline(ev['stop_loss'], color='red', linestyle=':', linewidth=1.5, label='Stop Loss')

            if spy is not None:
                d_spy = spy.tail(150)
                ax1b = ax1.twinx() 
                ax1b.plot(d_spy.index, d_spy['Close'], color='gray', alpha=0.3, linewidth=3, label='SPY (Ref)')
                ax1b.set_yticks([]) 
            if pos: ax1.axhline(pp, color=C_ACCENT, linewidth=1.5, label='Entry')
            ax1.set_title(f"{tkr} (D) - {fund['sec']} / {fund['ind']}", fontsize=10, color="white"); 
            ax1.legend(fontsize=8, facecolor=C_BG, edgecolor="white", labelcolor="white")
            ax1.grid(True, alpha=0.1, color="white")
            ax1.tick_params(colors='white')
            
            ax2.plot(d.index, d['CRSI'], color=C_ACCENT, linewidth=1.2)
            ax2.axhline(80, color=C_RED, linestyle=':', alpha=0.5); ax2.axhline(20, color=C_GREEN, linestyle=':', alpha=0.5)
            ax2.set_ylabel("CRSI", fontsize=8, color="white"); ax2.set_ylim(-5, 105); ax2.grid(True, alpha=0.1, color="white")
            ax2.tick_params(colors='white')
            cl = np.where(d['Vol_Osc']>0, C_GREEN, C_RED)
            ax3.bar(d.index, d['Vol_Osc'], color=cl, width=0.8, alpha=0.7)
            ax3.axhline(0, color='white', linewidth=0.5); ax3.set_ylabel("Vol%", fontsize=8, color="white"); ax3.grid(True, alpha=0.1, color="white")
            ax1.tick_params(axis='x', labelbottom=False); ax2.tick_params(axis='x', labelbottom=False)
            ax3.tick_params(colors='white')
            self.cv.draw(); self.b_rst.config(state="normal")
            
            self.txt.delete(1.0, tk.END); self.txt.insert(tk.END, f"{tkr} - ${d['Close'].iloc[-1]:.2f}\n", "t")
            if pos:
                gross_pl = (d['Close'].iloc[-1]*qq) - (pp*qq)
                net_pl = gross_pl - ETORO_ROUND_TRIP
                pc = (net_pl/(pp*qq))*100
                tag = "p" if net_pl>=0 else "n"
                self.txt.insert(tk.END, f"P&L: {net_pl:+.2f} ({pc:+.2f}%)\n", tag)
            
            self.txt.insert(tk.END, f"\n{self.texts['target_title']}\n", "gold")
            self.txt.insert(tk.END, f"${ev['target']:.2f} (Potential: {ev['profit']:.1f}%)\n", "p")

            self.txt.insert(tk.END, f"\n{self.texts['ai_title']}\n", "w")
            ai_tag = "ai_good" if prob_ai > 55 else "ai_bad" if prob_ai < 45 else "w"
            self.txt.insert(tk.END, f"{self.texts['ai_prob']} {prob_ai:.1f}%\n", ai_tag)
            acc_tag = "p" if acc > 60 else "n"
            self.txt.insert(tk.END, f"{self.texts['ai_acc']} {acc:.1f}%\n", acc_tag)
            if factors:
                self.txt.insert(tk.END, f"{self.texts['ai_factors']} ", "w")
                for f, imp in factors:
                    self.txt.insert(tk.END, f"{f}({imp*100:.0f}%) ", "gold")
                self.txt.insert(tk.END, "\n")
            
            self.txt.insert(tk.END, f"\n{self.texts['fund_title']}\n", "gold")
            ws_col = "p" if "BUY" in ws_rec or "STRONG" in ws_rec else "n" if "SELL" in ws_rec else "w"
            self.txt.insert(tk.END, f"{self.texts['ws_rating']} {ws_rec}\n", ws_col)
            if ws_target > 0: self.txt.insert(tk.END, f"{self.texts['ws_target']} ${ws_target:.2f}\n", "w")
            
            self.txt.insert(tk.END, f"PER: {fund['per']} | Div: {fund['div']}\n")
            g_tag = "p" if fund['graham'] > ev['price'] else "n"
            self.txt.insert(tk.END, f"Intrinsic (Graham): ${fund['graham']:.2f}\n", g_tag)

            self.txt.insert(tk.END, f"\n{self.texts['tech_title']}\n", "gold")
            self.txt.insert(tk.END, f"{self.texts['tech_sup']} ${sop:.2f}\n", "p")
            self.txt.insert(tk.END, f"{self.texts['tech_res']} ${resi:.2f}\n", "n")
            self.txt.insert(tk.END, f"{self.texts['tech_sl']} ${ev['stop_loss']:.2f}\n", "n")
            
            w_col = "p" if ev['weekly'] == "Alcista" else "n" if ev['weekly'] == "Bajista" else "w"
            self.txt.insert(tk.END, f"{self.texts['trend_wk']} {ev['weekly']}\n", w_col)

            self.txt.insert(tk.END, f"\n{self.texts['bench_title']}\n", "gold")
            b_col = "news_bull" if bench_stats['rel_perf'] > 0 else "news_bear"
            self.txt.insert(tk.END, f"{self.texts['bench_beta']} {bench_stats['beta']:.2f} | {self.texts['bench_rel']} ", "w")
            self.txt.insert(tk.END, f"{bench_stats['rel_perf']:.2f}%\n", b_col)
            
            self.txt.insert(tk.END, f"\nADX: {ev.get('adx',0):.1f} | CRSI: {ev.get('crsi',0):.1f}\n")
            sc_col = "ai_good" if ev['score'] > 70 else "ai_bad" if ev['score'] < 30 else "w"
            self.txt.insert(tk.END, f"SNIPER SCORE: {ev['score']}/100 ({ev['msg']})\n", sc_col)
            
            self.txt.insert(tk.END, f"\n{self.texts['news_title']}\n", "w")
            if noticias:
                for n in noticias:
                    tag = "w"; prefix = "⚪"
                    if n['sentiment'] == 'bull': tag = "news_bull"; prefix = "🟢"
                    elif n['sentiment'] == 'bear': tag = "news_bear"; prefix = "🔴"
                    self.txt.insert(tk.END, f"{prefix} {n['title']} ({n['source']})\n", tag)
            else: self.txt.insert(tk.END, "(Sin noticias)\n")
            self.txt.insert(tk.END, "\nFORECAST 7D:\n", "t")
            hoy = datetime.date.today(); prev=d['Close'].iloc[-1]
            for i in range(7):
                v = np.mean(sim[i]); c="p" if v>prev else "n"
                self.txt.insert(tk.END, f"{(hoy+datetime.timedelta(days=i+1)).strftime('%d-%m')}: ${v:.2f}\n", c); prev=v
        except Exception as e:
            messagebox.showerror("Err", str(e))
            self.status_var.set("Error en análisis.")

    def save(self):
        t=self.e_tk.get(); p=self.e_pr.get(); q=self.e_qt.get()
        if not p: p="0"; 
        if not q: q="0"
        if t: self.db.guardar_posicion(self.uid, t, float(p), float(q)); self.load_init()

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
        if total_pl >= 0: lbl_tot.config(foreground=C_GREEN)
        else: lbl_tot.config(foreground=C_RED)
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
        ax.set_title("Equity Curve", color="white")
        ax.grid(True, alpha=0.1); ax.tick_params(colors='white')
        cv = FigureCanvasTkAgg(fig, master=sw)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def ver_correlaciones(self):
        db_data = self.db.obtener_cartera(self.uid)
        if not db_data or len(db_data) < 2: 
            messagebox.showinfo("Info", "Necesitas al menos 2 acciones."); return
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
        cv = FigureCanvasTkAgg(fig, master=vw)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def abrir_calculadora(self):
        cw = tk.Toplevel(self.root); cw.title(self.texts["calc_title"]); cw.geometry("300x400")
        apply_dark_theme(cw)
        ttk.Label(cw, text=self.texts["calc_cap"]).pack(pady=2); e_cap = ttk.Entry(cw); e_cap.pack(); e_cap.insert(0, "10000") 
        ttk.Label(cw, text=self.texts["calc_risk"]).pack(pady=2); e_risk = ttk.Entry(cw); e_risk.pack(); e_risk.insert(0, "1") 
        ttk.Label(cw, text="Entry ($):").pack(pady=2); e_ent = ttk.Entry(cw); e_ent.pack()
        if self.e_pr.get(): e_ent.insert(0, self.e_pr.get())
        ttk.Label(cw, text=self.texts["calc_stop"]).pack(pady=2); e_stop = ttk.Entry(cw); e_stop.pack()
        
        ai_win_prob = 0.5
        if self.current_analysis:
            if self.current_analysis.get('price'): e_ent.delete(0,tk.END); e_ent.insert(0, str(round(self.current_analysis['price'], 2)))
            if self.current_analysis.get('stop_loss'): e_stop.delete(0,tk.END); e_stop.insert(0, str(round(self.current_analysis['stop_loss'], 2)))
            ai_win_prob = self.current_analysis.get('ia_prob', 50) / 100.0

        lbl_res = ttk.Label(cw, text="---", font=("bold", 12), foreground=C_ACCENT); lbl_res.pack(pady=10)
        lbl_kelly = ttk.Label(cw, text="", font=("Segoe UI", 9), foreground="gray"); lbl_kelly.pack(pady=2)

        def calcular():
            try:
                cap = float(e_cap.get()); r_pct = float(e_risk.get()); ent = float(e_ent.get()); stop = float(e_stop.get())
                if ent <= stop: lbl_res.config(text="Entry <= Stop!", foreground=C_RED); return
                risk_amt = cap * (r_pct/100)
                diff = ent - stop
                qty_base = int(risk_amt / diff)
                target = self.current_analysis.get('target', ent*1.05) if self.current_analysis else ent*1.05
                reward = target - ent
                risk = ent - stop
                R = reward / risk if risk > 0 else 1.0
                W = ai_win_prob
                kelly_pct = W - ((1-W)/R)
                kelly_safe_pct = kelly_pct / 4
                if kelly_safe_pct > 0:
                    kelly_cash = cap * kelly_safe_pct
                    kelly_qty = int(kelly_cash / ent)
                    msg_k = f"Kelly (1/4): {kelly_qty} shares (${kelly_cash:.0f})"
                    col_k = C_GREEN
                else:
                    msg_k = "Kelly says: NO TRADE (Negative Edge)"
                    col_k = C_RED
                    kelly_qty = 0
                lbl_res.config(text=f"Risk Basis: {qty_base} Shares", foreground=C_FG)
                lbl_kelly.config(text=msg_k, foreground=col_k)
                return kelly_qty if kelly_safe_pct > 0 else qty_base
            except: lbl_res.config(text="Error", foreground=C_RED); return 0

        def aplicar():
            qty = calcular()
            if qty > 0:
                self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qty))
                self.e_pr.delete(0, tk.END); self.e_pr.insert(0, e_ent.get())
                cw.destroy()

        tk.Button(cw, text=self.texts["calc_btn"], command=calcular, bg="#333", fg="white", relief="flat").pack(fill=tk.X, padx=20, pady=2)
        tk.Button(cw, text=self.texts["calc_apply"], command=aplicar, bg=C_GREEN, fg="black", relief="flat").pack(fill=tk.X, padx=20, pady=5)

    def exportar_cartera(self):
        datos = self.db.obtener_cartera(self.uid)
        if not datos: return
        df_export = pd.DataFrame(datos, columns=['Ticker', 'Precio', 'Cant', 'Fecha', 'ID']).drop(columns=['ID'])
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if f: df_export.to_csv(f, index=False); messagebox.showinfo("OK", self.texts["msg_exp_ok"])

    def abrir_config(self):
        messagebox.showinfo("Config", "Idioma fijo: ES (Por estabilidad)")

    def limpiar_campos(self): self.e_pr.delete(0, tk.END); self.e_qt.delete(0, tk.END)
    def zoom_rst(self): self.tb.home()
    def refresh_all(self): self.run(); self.scan_own(); self.scan_mkt()

    def update_dashboard_ui(self, inv, val):
        pl = val - inv - (ETORO_ROUND_TRIP * len(self.db.obtener_cartera(self.uid)))
        pl_pct = (pl / inv * 100) if inv > 0 else 0.0
        color = C_GREEN if pl >= 0 else C_RED
        self.lbl_invested.config(text=f"{self.texts['dash_inv']} ${inv:,.2f}")
        self.lbl_current.config(text=f"{self.texts['dash_val']} ${val:,.2f}")
        self.lbl_pl.config(text=f"{self.texts['dash_pl']} ${pl:,.2f} ({pl_pct:+.2f}%)", foreground=color)

    def update_row_ui(self, iid, tkr, pr_text, score, msg, tag_color):
        if self.tr1.exists(iid): 
            self.tr1.item(iid, values=(tkr, pr_text, score, msg), tags=(tag_color,))

    def sel_load(self, tree, is_own):
        s = tree.selection()
        if not s: return
        vals = tree.item(s[0])['values']; tkr = vals[0]
        self.e_tk.delete(0, tk.END); self.e_tk.insert(0, tkr)
        if is_own:
            tid = int(s[0])
            for d in self.db.obtener_cartera(self.uid):
                if d[4] == tid: 
                    self.e_pr.delete(0, tk.END); self.e_pr.insert(0, str(d[1]))
                    self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(d[2]))
        self.run()

    def update_ui_language(self):
        t = self.texts; self.root.title(f"{t['app_title']} - {self.uid}")
        self.btn_act.config(text=t["scan_own"])
        self.btn_gem.config(text=t["scan_mkt"]); self.btn_run.config(text=t["analyze"]); self.b_rst.config(text=t["reset_zoom"])
        self.lbl_buy.config(text=t["buy_price"]); self.lbl_qty.config(text=t["qty"]); self.btn_refresh.config(text=t["refresh_all"])
        self.lbl_invested.config(text=f"{t['dash_inv']} ---"); self.lbl_current.config(text=f"{t['dash_val']} ---"); self.lbl_pl.config(text=f"{t['dash_pl']} ---")
        self.btn_snap.config(text=t["snap_btn"])
        self.load_init()

    def logout(self):
        self.root.withdraw(); 
        for widget in self.root.winfo_children(): widget.destroy()
        LoginWindow(self.root, self.db, lambda u, n: (self.root.deiconify(), AppBolsa(self.root, u, n, self.db)))

    def borrar_cuenta(self):
        if messagebox.askyesno("!!!", self.texts["conf_del_confirm"]): self.db.borrar_usuario_completo(self.uid); self.root.destroy()

if __name__ == "__main__":
    db = DatabaseManager()
    root = tk.Tk(); root.withdraw()
    LoginWindow(root, db, lambda u, n: (root.deiconify(), AppBolsa(root, u, n, db)))
    root.mainloop()