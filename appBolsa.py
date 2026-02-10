import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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

# ==========================================
# 0. CONFIGURACIÓN E IDIOMAS
# ==========================================
LANG = {
    "ES": {
        "app_title": "Gestor Pro v13.0 (Benchmark)",
        "port_title": "📂 MI CARTERA",
        "opp_title": "💎 OPORTUNIDADES",
        "scan_own": "⚡ ACTUALIZAR",
        "save": "Guardar",
        "del": "Borrar",
        "exp": "📄 EXPORTAR",
        "scan_mkt": "🔍 ESCANEAR",
        "analyze": "▶ ANALIZAR",
        "reset_zoom": "RESET ZOOM",
        "buy_price": "Compra:",
        "qty": "Cant:",
        "col_ticker": "Ticker",
        "col_entry": "Entrada",
        "col_state": "Estado",
        "col_score": "Pts",
        "col_diag": "Diagnóstico",
        "vigil": "👁 VIGILANDO",
        "msg_wait": "⏳...",
        "msg_scan": "⏳ ANALIZANDO...",
        "msg_exp_ok": "✅ Guardado.",
        "conf_title": "Configuración",
        "conf_lang": "Idioma / Language:",
        "conf_logout": "🔒 SALIR",
        "conf_del": "⚠️ BORRAR CUENTA",
        "conf_del_confirm": "¿Seguro? Se borrarán tus datos.",
        "refresh_all": "🔄 TODO",
        "fund_title": "📊 FUNDAMENTALES:",
        "fund_pe": "PER:",
        "fund_cap": "Mkt Cap:",
        "fund_div": "Div Yield:",
        "bench_title": "🆚 MERCADO (vs SPY):",
        "bench_beta": "Beta:",
        "bench_rel": "Rendimiento Relativo:",
        "news_title": "📰 NOTICIAS & SENTIMIENTO:",
        "calc_title": "Calculadora Riesgo",
        "calc_cap": "Capital Total ($):",
        "calc_risk": "Riesgo Máx (%):",
        "calc_stop": "Stop Loss ($):",
        "calc_btn": "CALCULAR",
        "calc_res": "Acciones a Comprar:",
        "calc_apply": "APLICAR",
        "dash_inv": "Invertido:",
        "dash_val": "Valor Actual:",
        "dash_pl": "Total P/L:",
        "login_title": "ACCESO", "user": "Usuario:", "pass": "Clave:", "btn_enter": "ENTRAR", "btn_reg": "REGISTRO", "err_login": "Error", "ok_reg": "OK", "err_reg": "Existe"
    },
    "EN": { "app_title": "Pro Manager v13.0", "port_title": "📂 PORTFOLIO", "opp_title": "💎 OPPORTUNITIES", "scan_own": "⚡ REFRESH", "save": "Save", "del": "Delete", "exp": "📄 EXPORT", "scan_mkt": "🔍 SCAN", "analyze": "▶ ANALYZE", "reset_zoom": "RESET", "buy_price": "Price:", "qty": "Qty:", "col_ticker": "Ticker", "col_entry": "Entry", "col_state": "Status", "col_score": "Pts", "col_diag": "Diagnosis", "vigil": "👁 WATCH", "msg_wait": "⏳...", "msg_scan": "⏳...", "msg_exp_ok": "✅ Saved.", "conf_title": "Settings", "conf_lang": "Language:", "conf_logout": "🔒 LOGOUT", "conf_del": "⚠️ DELETE", "conf_del_confirm": "Sure?", "refresh_all": "🔄 ALL", "fund_title": "📊 FUNDAMENTALS:", "fund_pe": "P/E:", "fund_cap": "Cap:", "fund_div": "Div:", "bench_title": "🆚 MARKET (vs SPY):", "bench_beta": "Beta:", "bench_rel": "Rel. Perf:", "news_title": "📰 NEWS & SENTIMENT:", "calc_title": "Risk Calculator", "calc_cap": "Total Capital ($):", "calc_risk": "Max Risk (%):", "calc_stop": "Stop Loss ($):", "calc_btn": "CALCULATE", "calc_res": "Shares to Buy:", "calc_apply": "APPLY", "dash_inv": "Invested:", "dash_val": "Cur. Value:", "dash_pl": "Total P/L:", "login_title": "LOGIN", "user": "User:", "pass": "Pass:", "btn_enter": "GO", "btn_reg": "REG", "err_login": "Invalid", "ok_reg": "OK", "err_reg": "Exists" },
    "FR": { "app_title": "Gestion Pro v13.0", "port_title": "📂 PORTEFEUILLE", "opp_title": "💎 OPPORTUNITÉS", "scan_own": "⚡ ACTUALISER", "save": "Sauver", "del": "Effacer", "exp": "📄 EXPORTER", "scan_mkt": "🔍 SCANNER", "analyze": "▶ ANALYSER", "reset_zoom": "ZOOM", "buy_price": "Prix:", "qty": "Qté:", "col_ticker": "Ticker", "col_entry": "Entrée", "col_state": "État", "col_score": "Pts", "col_diag": "Diagnostic", "vigil": "👁 VOIR", "msg_wait": "⏳...", "msg_scan": "⏳...", "msg_exp_ok": "✅ Sauvé.", "conf_title": "Paramètres", "conf_lang": "Langue:", "conf_logout": "🔒 SORTIR", "conf_del": "⚠️ SUPPRIMER", "conf_del_confirm": "Sûr?", "refresh_all": "🔄 TOUT", "fund_title": "📊 FONDAMENTAUX:", "fund_pe": "PER:", "fund_cap": "Cap:", "fund_div": "Div:", "bench_title": "🆚 MARCHÉ (vs SPY):", "bench_beta": "Beta:", "bench_rel": "Perf. Rel:", "news_title": "📰 NOUVELLES:", "calc_title": "Calculateur Risque", "calc_cap": "Capital Total:", "calc_risk": "Risque (%):", "calc_stop": "Stop Loss:", "calc_btn": "CALCULER", "calc_res": "Acheter:", "calc_apply": "APPLIQUER", "dash_inv": "Investi:", "dash_val": "Val. Act:", "dash_pl": "Total P/L:", "login_title": "LOGIN", "user": "User:", "pass": "Pass:", "btn_enter": "ENTRER", "btn_reg": "CREER", "err_login": "Erreur", "ok_reg": "OK", "err_reg": "Existe" },
    "PT": { "app_title": "Gestor Pro v13.0", "port_title": "📂 CARTEIRA", "opp_title": "💎 OPORTUNIDADES", "scan_own": "⚡ ATUALIZAR", "save": "Salvar", "del": "Apagar", "exp": "📄 EXPORTAR", "scan_mkt": "🔍 BUSCAR", "analyze": "▶ ANALISAR", "reset_zoom": "ZOOM", "buy_price": "Preço:", "qty": "Qtd:", "col_ticker": "Ticker", "col_entry": "Entrada", "col_state": "Estado", "col_score": "Pts", "col_diag": "Diagnóstico", "vigil": "👁 VIGIAR", "msg_wait": "⏳...", "msg_scan": "⏳...", "msg_exp_ok": "✅ Salvo.", "conf_title": "Config", "conf_lang": "Idioma:", "conf_logout": "🔒 SAIR", "conf_del": "⚠️ APAGAR", "conf_del_confirm": "Certeza?", "refresh_all": "🔄 TUDO", "fund_title": "📊 FUNDAMENTAIS:", "fund_pe": "P/L:", "fund_cap": "Val. Merc:", "fund_div": "Div:", "bench_title": "🆚 MERCADO (vs SPY):", "bench_beta": "Beta:", "bench_rel": "Perf. Rel:", "news_title": "📰 NOTÍCIAS:", "calc_title": "Calculadora Risco", "calc_cap": "Capital Total:", "calc_risk": "Risco (%):", "calc_stop": "Stop Loss:", "calc_btn": "CALCULAR", "calc_res": "Comprar:", "calc_apply": "APLICAR", "dash_inv": "Investido:", "dash_val": "Val. Atual:", "dash_pl": "Total P/L:", "login_title": "LOGIN", "user": "User:", "pass": "Senha:", "btn_enter": "ENTRAR", "btn_reg": "CRIAR", "err_login": "Erro", "ok_reg": "OK", "err_reg": "Existe" }
}

CANDIDATOS_VIP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", 
    "AMD", "INTC", "PYPL", "KO", "PEP", "DIS", "BA", "CSCO", "WMT", "JPM",
    "UBER", "ABNB", "PLTR", "SHOP", "SBUX", "NKE", "MCD", "V", "MA"
]

# ==========================================
# 1. BASE DE DATOS
# ==========================================
class DatabaseManager:
    def __init__(self, db_name="bolsa_datos_v13.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.crear_tablas()

    def crear_tablas(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cartera (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, ticker TEXT NOT NULL, 
            precio_compra REAL, cantidad REAL, fecha_guardado TEXT, FOREIGN KEY(user_id) REFERENCES usuarios(id))''')
        self.conn.commit()

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
        self.conn.execute("DELETE FROM cartera WHERE id=?", (pid,))
        self.conn.commit()

    def borrar_usuario_completo(self, uid):
        self.conn.execute("DELETE FROM cartera WHERE user_id=?", (uid,))
        self.conn.execute("DELETE FROM usuarios WHERE id=?", (uid,))
        self.conn.commit()

# ==========================================
# 2. MOTOR ANALÍTICO
# ==========================================
class AnalistaBolsa:
    def __init__(self):
        self.data = None; self.ticker = ""
        self.spy_data = None # Cache para SPY

    def descargar_datos(self, ticker):
        self.ticker = ticker.upper()
        try:
            d = yf.download(self.ticker, period="2y", progress=False)
            if d.empty: raise ValueError
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.droplevel(1)
            self.data = d.astype(float)
            return d
        except: raise ValueError("Error descarga")

    # --- NUEVO: BENCHMARK (SPY) ---
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
            # Alinear fechas (Inner Join)
            df = pd.DataFrame({'STOCK': stock_df['Close'], 'SPY': spy_df['Close']}).dropna()
            if len(df) < 50: return {"beta": 0, "rel_perf": 0}
            
            # Rentabilidad Diaria
            rets = df.pct_change().dropna()
            
            # Beta: Covarianza / Varianza del Mercado
            cov = rets['STOCK'].cov(rets['SPY'])
            var = rets['SPY'].var()
            beta = cov / var if var != 0 else 1.0
            
            # Rendimiento Relativo (Ultimos 6 meses aprox 126 dias)
            lookback = min(126, len(df))
            stock_ret = (df['STOCK'].iloc[-1] / df['STOCK'].iloc[-lookback]) - 1
            spy_ret = (df['SPY'].iloc[-1] / df['SPY'].iloc[-lookback]) - 1
            rel_perf = (stock_ret - spy_ret) * 100 # Diferencia en %
            
            return {"beta": beta, "rel_perf": rel_perf}
        except: return {"beta": 0, "rel_perf": 0}

    def obtener_fundamentales(self, ticker):
        try:
            t = yf.Ticker(ticker); i = t.info
            per = i.get('trailingPE', i.get('forwardPE', 0))
            cap = i.get('marketCap', 0)
            div = i.get('dividendYield', 0)
            sec = i.get('sector', 'N/A')
            ind = i.get('industry', 'N/A')
            
            if cap > 1e12: s_cap = f"{cap/1e12:.2f}T"
            elif cap > 1e9: s_cap = f"{cap/1e9:.2f}B"
            elif cap > 1e6: s_cap = f"{cap/1e6:.2f}M"
            else: s_cap = str(cap)
            
            return {"per": f"{per:.2f}" if per else "N/A", "cap": s_cap, "div": f"{div*100:.2f}%" if div else "0%", "sec": sec, "ind": ind, "valid": True}
        except: return {"per": "-", "cap": "-", "div": "-", "sec": "-", "ind": "-", "valid": False}

    def obtener_noticias_analizadas(self, ticker):
        try:
            t = yf.Ticker(ticker); news = t.news; analisis_noticias = []
            bull = ['surge', 'jump', 'rise', 'gain', 'profit', 'beat', 'growth', 'record', 'buy', 'bull', 'upgrade', 'high', 'positive']
            bear = ['drop', 'fall', 'plunge', 'loss', 'miss', 'cut', 'bear', 'downgrade', 'low', 'negative', 'crash', 'risk']
            for n in news[:4]:
                tit = n.get('title'); 
                if not tit and 'content' in n: tit = n['content'].get('title')
                if not tit: continue
                src = n.get('publisher', 'Yahoo')
                if isinstance(src, dict): src = src.get('title', 'Yahoo')
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

    def calcular_indicadores(self):
        if self.data is None or self.data.empty: return self.data
        for col in ['ADX', 'Vol_Osc', 'CRSI', 'SMA_50', 'SMA_200']: self.data[col] = 0.0
        if len(self.data) < 50: return self.data

        df = self.data.copy(); w = 14
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        df['Prev'] = df['Close'].shift(1)
        df['TR'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Prev']), abs(df['Low']-df['Prev'])))
        tr14 = df['TR'].ewm(alpha=1/w).mean()
        up = df['High'] - df['High'].shift(1); down = df['Low'].shift(1) - df['Low']
        pdm = np.where((up>down)&(up>0), up, 0.0); mdm = np.where((down>up)&(down>0), down, 0.0)
        with np.errstate(all='ignore'):
            pdi = 100*(pd.Series(pdm, index=df.index).ewm(alpha=1/w).mean()/tr14)
            mdi = 100*(pd.Series(mdm, index=df.index).ewm(alpha=1/w).mean()/tr14)
            df['ADX'] = (100*abs(pdi-mdi)/(pdi+mdi)).ewm(alpha=1/w).mean()
        
        v_s, v_l = 5, 10
        vol_ma = df['Volume'].rolling(v_l).mean().replace(0, np.nan)
        df['Vol_Osc'] = ((df['Volume'].rolling(v_s).mean()-vol_ma)/vol_ma)*100
        
        df['RSI_P'] = self._rsi(df['Close'], 3)
        chg = df['Close'].diff().values; st = np.zeros(len(df))
        for i in range(1, len(chg)):
            if chg[i]>0: st[i] = st[i-1]+1 if st[i-1]>=0 else 1
            elif chg[i]<0: st[i] = st[i-1]-1 if st[i-1]<=0 else -1
        df['St_RSI'] = self._rsi(pd.Series(st, index=df.index), 2)
        df['PRank'] = df['Close'].pct_change().rolling(100).rank(pct=True)*100
        df['PRank'] = df['PRank'].fillna(50) 
        df['CRSI'] = (df['RSI_P'] + df['St_RSI'] + df['PRank'])/3
        
        df['ADX'] = df['ADX'].fillna(20); df['CRSI'] = df['CRSI'].fillna(50); df['Vol_Osc'] = df['Vol_Osc'].fillna(0)
        self.data = df
        return self.data

    def _rsi(self, s, p):
        d = s.diff(); g = d.where(d>0,0).ewm(alpha=1/p).mean(); l = -d.where(d<0,0).ewm(alpha=1/p).mean()
        return 100-(100/(1+(g/l)))

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
            adx = last.get('ADX', 0); crsi = last.get('CRSI', 50); curr = last['Close']
            score = 0; msg = "Neutro"; tag = ""
            if p_compra > 0: 
                 price = last['Close']
                 if crsi > 80: msg = "⚠️ VENDER"; tag = "sell"
                 elif price < p_compra*0.9: msg = "🛑 STOP LOSS"; tag = "sell"
                 elif crsi < 20 and adx > 25: msg = "💎 ACUMULAR"; tag = "buy"
                 else: msg = "✋ MANTENER"; tag = "hold"
                 return {"valid": True, "ticker": self.ticker, "score": 0, "msg": msg, "tag": tag, "crsi": crsi, "adx": adx, "price": curr}

            if crsi < 20 and adx > 25: score = 100; msg = "🚀 COMPRA"; tag = "buy"
            elif crsi < 15: score = 90; msg = "🚀 REBOTE"; tag = "buy"
            elif crsi < 30 and adx > 20: score = 80; msg = "👀 PREPARAR"; tag = "near"
            elif crsi < 30: score = 75; msg = "👀 BARATO"; tag = "near"
            elif adx > 30 and crsi < 60: score = 60; msg = "📈 TENDENCIA"; tag = "trend"
            elif crsi < 40: score = 40; msg = "💤 BAJANDO"; tag = "weak"
            return {"valid": True, "ticker": self.ticker, "score": score, "msg": msg, "tag": tag, "crsi": crsi, "adx": adx, "price": curr}
        except: return {"valid": False, "msg": "Error"}

    def evaluar_candidato_externo(self, ticker):
        try:
            self.descargar_datos(ticker); self.calcular_indicadores()
            return self.generar_diagnostico_interno(0)
        except: return {"valid": False}

# ==========================================
# 3. LOGIN GUI
# ==========================================
class LoginWindow:
    def __init__(self, root, db, on_success):
        self.root = root; self.db = db; self.on_success = on_success
        self.win = tk.Toplevel(root); self.win.title("Acceso v13.0"); self.win.geometry("350x300")
        self.texts = LANG["ES"]
        ttk.Label(self.win, text=self.texts["login_title"], font=("Arial", 14, "bold")).pack(pady=20)
        ttk.Label(self.win, text=self.texts["user"]).pack(); self.e_u = ttk.Entry(self.win); self.e_u.pack(pady=5)
        ttk.Label(self.win, text=self.texts["pass"]).pack(); self.e_p = ttk.Entry(self.win, show="*"); self.e_p.pack(pady=5)
        tk.Button(self.win, text=self.texts["btn_enter"], command=self.log, bg="#007acc", fg="white", font=("bold",10)).pack(pady=15, fill=tk.X, padx=30)
        tk.Button(self.win, text=self.texts["btn_reg"], command=self.reg, bg="#555", fg="white").pack(pady=5, fill=tk.X, padx=30)
        self.win.protocol("WM_DELETE_WINDOW", root.destroy)

    def log(self):
        u_val = self.e_u.get(); p_val = self.e_p.get()
        uid = self.db.verificar_usuario(u_val, p_val)
        if uid: self.win.destroy(); self.on_success(uid, u_val)
        else: messagebox.showerror("Error", self.texts["err_login"])

    def reg(self):
        if self.db.registrar_usuario(self.e_u.get(), self.e_p.get()): messagebox.showinfo("OK", self.texts["ok_reg"])
        else: messagebox.showerror("Error", self.texts["err_reg"])

# ==========================================
# 4. APP PRINCIPAL
# ==========================================
class AppBolsa:
    def __init__(self, root, uid, uname, db):
        self.root = root; self.uid = uid; self.db = db; self.eng = AnalistaBolsa()
        self.current_lang = "ES"; self.texts = LANG[self.current_lang]
        self.root.geometry("1600x950")
        ttk.Style().theme_use('clam')
        
        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL); main.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(main, width=450, relief=tk.RAISED); main.add(side, weight=1)
        
        self.lf1 = ttk.LabelFrame(side, padding=5); self.lf1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dash_frame = ttk.LabelFrame(self.lf1, text="💰 RESUMEN PATRIMONIO", padding=5)
        self.dash_frame.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_invested = ttk.Label(self.dash_frame, text="---", font=("bold", 10)); self.lbl_invested.pack(anchor="w")
        self.lbl_current = ttk.Label(self.dash_frame, text="---", font=("bold", 10)); self.lbl_current.pack(anchor="w")
        self.lbl_pl = ttk.Label(self.dash_frame, text="---", font=("bold", 10)); self.lbl_pl.pack(anchor="w")

        cols = ("tk", "pr", "sg")
        self.tr1 = ttk.Treeview(self.lf1, columns=cols, show="headings", height=10)
        for c in cols: self.tr1.column(c, anchor="center", stretch=True)
        self.tr1.pack(fill=tk.BOTH, expand=True)
        self.tr1.bind("<Double-1>", lambda e: self.sel_load(self.tr1, True))
        
        f1 = ttk.Frame(self.lf1); f1.pack(fill=tk.X)
        self.btn_act = tk.Button(f1, bg="orange", command=self.scan_own); self.btn_act.pack(fill=tk.X, pady=2)
        self.btn_save = ttk.Button(f1, command=self.save); self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_del = ttk.Button(f1, command=self.dele); self.btn_del.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_exp = tk.Button(f1, bg="#90ee90", command=self.exportar_cartera); self.btn_exp.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lf2 = ttk.LabelFrame(side, padding=5); self.lf2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        cols2 = ("tk", "sc", "ms")
        self.tr2 = ttk.Treeview(self.lf2, columns=cols2, show="headings", height=12)
        for c in cols2: self.tr2.column(c, anchor="center", stretch=True)
        self.tr2.pack(fill=tk.BOTH, expand=True)
        self.tr2.bind("<Double-1>", lambda e: self.sel_load(self.tr2, False))
        
        for tr in [self.tr1, self.tr2]:
            tr.tag_configure('buy', background='#00ff00', foreground='black')
            tr.tag_configure('near', background='#ccffcc', foreground='black')
            tr.tag_configure('trend', background='#fffacd', foreground='black')
            tr.tag_configure('sell', background='#ffcccc', foreground='black')
            tr.tag_configure('hold', foreground='black')

        self.btn_gem = tk.Button(self.lf2, bg="#007acc", fg="white", font=("bold",10), command=self.scan_mkt)
        self.btn_gem.pack(fill=tk.X, pady=5)

        cont = ttk.Frame(main); main.add(cont, weight=4)
        
        ctrl = ttk.LabelFrame(cont, text=" Analizar ", padding=5); ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="TICKER:").pack(side=tk.LEFT)
        self.e_tk = ttk.Entry(ctrl, width=10); self.e_tk.pack(side=tk.LEFT, padx=5)
        self.e_tk.bind('<Return>', lambda e: self.run())
        self.btn_run = tk.Button(ctrl, command=self.run, bg="#ddd"); self.btn_run.pack(side=tk.LEFT)
        self.b_rst = tk.Button(ctrl, command=self.zoom_rst, state="disabled"); self.b_rst.pack(side=tk.LEFT, padx=5)
        
        self.lbl_buy = ttk.Label(ctrl); self.lbl_buy.pack(side=tk.LEFT)
        self.e_pr = ttk.Entry(ctrl, width=8); self.e_pr.pack(side=tk.LEFT)
        self.lbl_qty = ttk.Label(ctrl); self.lbl_qty.pack(side=tk.LEFT)
        self.e_qt = ttk.Entry(ctrl, width=8); self.e_qt.pack(side=tk.LEFT)
        self.btn_clean = tk.Button(ctrl, text="🗑", command=self.limpiar_campos, bg="#ffcccc"); self.btn_clean.pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl, text="🧮", command=self.abrir_calculadora, bg="#e6e6fa").pack(side=tk.LEFT, padx=2)

        self.btn_conf = ttk.Button(ctrl, text="⚙️ CONFIG", command=self.abrir_config)
        self.btn_conf.pack(side=tk.RIGHT, padx=5)
        self.btn_refresh = tk.Button(ctrl, bg="purple", fg="white", font=("bold", 9), command=self.refresh_all)
        self.btn_refresh.pack(side=tk.RIGHT, padx=5)

        pan = ttk.PanedWindow(cont, orient=tk.HORIZONTAL); pan.pack(fill=tk.BOTH, expand=True, padx=10)
        self.txt = tk.Text(pan, width=40, bg="#111", fg="#0f0", font=("Consolas", 10)); pan.add(self.txt, weight=1)
        self.txt.tag_config("t", foreground="cyan", font=("bold",12))
        self.txt.tag_config("p", foreground="#0f0"); self.txt.tag_config("n", foreground="#f44")
        self.txt.tag_config("gold", foreground="gold"); self.txt.tag_config("w", foreground="white")
        self.txt.tag_config("news_bull", foreground="#00ff00"); self.txt.tag_config("news_bear", foreground="#ff4444")
        
        frg = ttk.Frame(pan)
        self.fig = Figure(figsize=(5,5), dpi=100)
        self.cv = FigureCanvasTkAgg(self.fig, master=frg)
        self.tb = NavigationToolbar2Tk(self.cv, frg); self.tb.update()
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        pan.add(frg, weight=3)
        
        self.update_ui_language()
        self.load_init()

    # --- CALCULADORA RIESGO ---
    def abrir_calculadora(self):
        cw = tk.Toplevel(self.root); cw.title(self.texts["calc_title"]); cw.geometry("250x350")
        ttk.Label(cw, text=self.texts["calc_cap"]).pack(pady=2); e_cap = ttk.Entry(cw); e_cap.pack(); e_cap.insert(0, "10000") 
        ttk.Label(cw, text=self.texts["calc_risk"]).pack(pady=2); e_risk = ttk.Entry(cw); e_risk.pack(); e_risk.insert(0, "1") 
        ttk.Label(cw, text="Entry ($):").pack(pady=2); e_ent = ttk.Entry(cw); e_ent.pack()
        if self.e_pr.get(): e_ent.insert(0, self.e_pr.get())
        ttk.Label(cw, text=self.texts["calc_stop"]).pack(pady=2); e_stop = ttk.Entry(cw); e_stop.pack()
        lbl_res = ttk.Label(cw, text="---", font=("bold", 12), foreground="blue"); lbl_res.pack(pady=10)
        def calcular():
            try:
                cap = float(e_cap.get()); r_pct = float(e_risk.get()); ent = float(e_ent.get()); stop = float(e_stop.get())
                if ent <= stop: lbl_res.config(text="Entry <= Stop!", foreground="red"); return
                risk_amt = cap * (r_pct/100); diff = ent - stop; qty = int(risk_amt / diff)
                lbl_res.config(text=f"{qty} Shares", foreground="green"); return qty
            except: lbl_res.config(text="Error", foreground="red")
        def aplicar():
            qty = calcular()
            if qty and qty > 0:
                self.e_qt.delete(0, tk.END); self.e_qt.insert(0, str(qty)); self.e_pr.delete(0, tk.END); self.e_pr.insert(0, e_ent.get()); cw.destroy()
        tk.Button(cw, text=self.texts["calc_btn"], command=calcular).pack(fill=tk.X, padx=20, pady=2)
        tk.Button(cw, text=self.texts["calc_apply"], command=aplicar, bg="#90ee90").pack(fill=tk.X, padx=20, pady=5)

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
        ttk.Label(cw, text=self.texts["conf_lang"], font=("bold", 10)).pack(pady=10)
        cb = ttk.Combobox(cw, values=["ES", "EN", "FR", "PT"], state="readonly")
        cb.set(self.current_lang); cb.pack(pady=5)
        def change_lang(e): self.current_lang = cb.get(); self.texts = LANG[self.current_lang]; self.update_ui_language(); cw.title(self.texts["conf_title"])
        cb.bind("<<ComboboxSelected>>", change_lang)
        ttk.Separator(cw, orient='horizontal').pack(fill='x', pady=20)
        def realizar_logout(): cw.destroy(); self.logout()
        tk.Button(cw, text=self.texts["conf_logout"], bg="#555", fg="white", font=("bold", 10), command=realizar_logout).pack(pady=5, fill=tk.X, padx=20)
        tk.Button(cw, text=self.texts["conf_del"], bg="red", fg="white", font=("bold", 10), command=self.borrar_cuenta).pack(pady=5, fill=tk.X, padx=20)

    def update_ui_language(self):
        t = self.texts; self.root.title(f"{t['app_title']} - {self.uid}")
        self.lf1.config(text=t["port_title"]); self.tr1.heading("tk", text=t["col_ticker"]); self.tr1.heading("pr", text=t["col_entry"]); self.tr1.heading("sg", text=t["col_state"])
        self.btn_act.config(text=t["scan_own"]); self.btn_save.config(text=t["save"]); self.btn_del.config(text=t["del"]); self.btn_exp.config(text=t["exp"])
        self.lf2.config(text=t["opp_title"]); self.tr2.heading("tk", text=t["col_ticker"]); self.tr2.heading("sc", text=t["col_score"]); self.tr2.heading("ms", text=t["col_diag"])
        self.btn_gem.config(text=t["scan_mkt"]); self.btn_run.config(text=t["analyze"]); self.b_rst.config(text=t["reset_zoom"])
        self.lbl_buy.config(text=t["buy_price"]); self.lbl_qty.config(text=t["qty"]); self.btn_refresh.config(text=t["refresh_all"])
        self.lbl_invested.config(text=f"{t['dash_inv']} ---"); self.lbl_current.config(text=f"{t['dash_val']} ---"); self.lbl_pl.config(text=f"{t['dash_pl']} ---")
        self.load_init()

    def logout(self):
        self.root.withdraw(); 
        for widget in self.root.winfo_children(): widget.destroy()
        LoginWindow(self.root, self.db, lambda u, n: (self.root.deiconify(), AppBolsa(self.root, u, n, self.db)))

    def borrar_cuenta(self):
        if messagebox.askyesno("!!!", self.texts["conf_del_confirm"]): self.db.borrar_usuario_completo(self.uid); self.root.destroy()

    def load_init(self):
        for i in self.tr1.get_children(): self.tr1.delete(i)
        for d in self.db.obtener_cartera(self.uid):
            pr = f"${d[1]}" if d[1]>0 else self.texts["vigil"]
            self.tr1.insert("", "end", iid=d[4], values=(d[0], pr, "..."))

    def limpiar_campos(self): self.e_pr.delete(0, tk.END); self.e_qt.delete(0, tk.END)

    def scan_own(self):
        self.btn_act.config(state="disabled", text=self.texts["msg_wait"])
        threading.Thread(target=self._th_own, daemon=True).start()

    def _th_own(self):
        total_inv = 0.0; total_val = 0.0
        children = self.tr1.get_children()
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
        self.root.after(0, lambda: self.btn_act.config(state="normal", text=self.texts["scan_own"]))

    def update_dashboard_ui(self, inv, val):
        pl = val - inv
        pl_pct = (pl / inv * 100) if inv > 0 else 0.0
        color = "green" if pl >= 0 else "red"
        self.lbl_invested.config(text=f"{self.texts['dash_inv']} ${inv:,.2f}")
        self.lbl_current.config(text=f"{self.texts['dash_val']} ${val:,.2f}")
        self.lbl_pl.config(text=f"{self.texts['dash_pl']} ${pl:,.2f} ({pl_pct:+.2f}%)", foreground=color)

    def update_row_ui(self, iid, tkr, pr_text, msg, tag):
        if self.tr1.exists(iid): self.tr1.item(iid, values=(tkr, pr_text, msg), tags=(tag,))

    def scan_mkt(self):
        self.btn_gem.config(state="disabled", text=self.texts["msg_scan"])
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
            self.root.after(0, self.tr2.insert, "", "end", values=(c["ticker"], c["score"], c["msg"]), tags=(c["tag"],))
        self.root.after(0, lambda: self.btn_gem.config(state="normal", text=self.texts["scan_mkt"]))

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
        tkr = self.e_tk.get().strip(); 
        if not tkr: return
        pp=0; qq=0; pos=False
        if self.e_pr.get(): 
            try: pp=float(self.e_pr.get()); qq=float(self.e_qt.get()); pos=(pp>0)
            except: pass
        
        try:
            self.eng.descargar_datos(tkr); df = self.eng.calcular_indicadores()
            spy = self.eng.obtener_benchmark() # NUEVO
            fund = self.eng.obtener_fundamentales(tkr)
            noticias = self.eng.obtener_noticias_analizadas(tkr)
            sim = self.eng.simular(); ev = self.eng.generar_diagnostico_interno(pp)
            
            # --- CALCULOS COMPARATIVA ---
            bench_stats = self.eng.calcular_beta_relativa(df, spy) if spy is not None else {"beta": 0, "rel_perf": 0}
            
            self.fig.clear()
            gs = self.fig.add_gridspec(3, 1, height_ratios=[3,1,1])
            ax1=self.fig.add_subplot(gs[0]); ax2=self.fig.add_subplot(gs[1], sharex=ax1); ax3=self.fig.add_subplot(gs[2], sharex=ax1)
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.15)
            
            d = df.tail(150)
            
            # --- PLOT COMPARATIVO VISUAL (NUEVO) ---
            ax1.plot(d.index, d['Close'], color='#333', linewidth=1.2, label='Precio')
            ax1.plot(d.index, d['SMA_50'], color='orange', linestyle='--', linewidth=1, label='SMA 50')
            ax1.plot(d.index, d['SMA_200'], color='purple', linewidth=1.5, label='SMA 200')
            
            # Overlay SPY Trend (Truco visual: Eje gemelo invisible para comparar tendencia)
            if spy is not None:
                d_spy = spy.tail(150)
                ax1b = ax1.twinx() # Eje secundario
                ax1b.plot(d_spy.index, d_spy['Close'], color='gray', alpha=0.3, linewidth=3, label='SPY (Ref)')
                ax1b.set_yticks([]) # Ocultar numeros eje derecho para no confundir
                # ax1b.legend(loc='upper left', fontsize=6) # Opcional
            
            if pos: ax1.axhline(pp, color='blue', linewidth=1.5, label='Entry')
            ax1.set_title(f"{tkr} (D) - {fund['sec']} / {fund['ind']}", fontsize=10); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)
            
            ax2.plot(d.index, d['CRSI'], color='#007acc', linewidth=1.2)
            ax2.axhline(80, color='red', linestyle=':', alpha=0.5); ax2.axhline(20, color='green', linestyle=':', alpha=0.5)
            ax2.set_ylabel("CRSI", fontsize=8); ax2.set_ylim(-5, 105); ax2.grid(True, alpha=0.2)
            
            cl = np.where(d['Vol_Osc']>0, 'green', 'red')
            ax3.bar(d.index, d['Vol_Osc'], color=cl, width=0.8, alpha=0.7)
            ax3.axhline(0, color='black', linewidth=0.5); ax3.set_ylabel("Vol%", fontsize=8); ax3.grid(True, alpha=0.2)
            ax1.tick_params(axis='x', labelbottom=False); ax2.tick_params(axis='x', labelbottom=False)
            
            self.cv.draw(); self.b_rst.config(state="normal")
            
            self.txt.delete(1.0, tk.END); self.txt.insert(tk.END, f"{tkr} - ${d['Close'].iloc[-1]:.2f}\n", "t")
            if pos:
                pl = (d['Close'].iloc[-1]*qq)-(pp*qq); pc = (pl/(pp*qq))*100
                self.txt.insert(tk.END, f"P&L: {pl:+.2f} ({pc:+.2f}%)\n", "p" if pl>=0 else "n")
            
            # --- DATOS COMPARATIVOS ---
            self.txt.insert(tk.END, f"\n{self.texts['bench_title']}\n", "gold")
            b_col = "news_bull" if bench_stats['rel_perf'] > 0 else "news_bear"
            self.txt.insert(tk.END, f"{self.texts['bench_beta']} {bench_stats['beta']:.2f} | {self.texts['bench_rel']} ", "w")
            self.txt.insert(tk.END, f"{bench_stats['rel_perf']:.2f}%\n", b_col)

            self.txt.insert(tk.END, f"\n{self.texts['fund_title']}\n", "gold")
            self.txt.insert(tk.END, f"{self.texts['fund_pe']} {fund['per']} | {self.texts['fund_div']} {fund['div']}\n")
            self.txt.insert(tk.END, f"{self.texts['fund_cap']} {fund['cap']}\n")

            self.txt.insert(tk.END, f"\nADX: {ev.get('adx',0):.1f} | CRSI: {ev.get('crsi',0):.1f}\n")
            self.txt.insert(tk.END, f"DX: {ev['msg']} (Score: {ev['score']})\n", "t")
            
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
        except Exception as e: messagebox.showerror("Err", str(e))

    def save(self):
        t=self.e_tk.get(); p=self.e_pr.get(); q=self.e_qt.get()
        if not p: p="0"; 
        if not q: q="0"
        if t: self.db.guardar_posicion(self.uid, t, float(p), float(q)); self.load_init()

    def dele(self):
        s = self.tr1.selection()
        if s: self.db.borrar_posicion(s[0]); self.load_init()

    def zoom_rst(self): self.tb.home()

if __name__ == "__main__":
    db = DatabaseManager()
    root = tk.Tk(); root.withdraw()
    LoginWindow(root, db, lambda u, n: (root.deiconify(), AppBolsa(root, u, n, db)))
    root.mainloop()