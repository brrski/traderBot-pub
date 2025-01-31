import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                            QTableWidgetItem, QTabWidget)
from PyQt6.QtCore import Qt, QTimer

class StockAnalyzer:
    def __init__(self):
        self.symbols = ['TSLA', 'AMD', 'NVDA', 'META', 'AAPL', 'QQQ', 'SPY']
        self.price_data = {}
        self.analysis_results = {}
        self.update_data()

    def update_data(self):
        """Fetch and update price data for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    self.price_data[symbol] = df
                    print(f"Fetched data for {symbol}: {len(df)} rows")
                else:
                    print(f"No data received for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd[-1], signal[-1]

    def analyze_stocks(self):
        """Perform analysis on all stocks"""
        for symbol in self.symbols:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                df = self.price_data[symbol]
                
                # Calculate returns for different periods
                current_price = df['Close'][-1]
                price_7d = df['Close'][-7] if len(df) >= 7 else df['Close'][0]
                price_30d = df['Close'][-30] if len(df) >= 30 else df['Close'][0]
                price_90d = df['Close'][-90] if len(df) >= 90 else df['Close'][0]
                price_365d = df['Close'][-365] if len(df) >= 365 else df['Close'][0]
                
                change_7d = ((current_price - price_7d) / price_7d * 100)
                change_30d = ((current_price - price_30d) / price_30d * 100)
                change_90d = ((current_price - price_90d) / price_90d * 100)
                change_365d = ((current_price - price_365d) / price_365d * 100)
                
                # Calculate technical indicators
                rsi = self.calculate_rsi(df['Close'])
                sma_10 = df['Close'].rolling(window=10).mean().iloc[-1]
                macd_line, signal_line = self.calculate_macd(df['Close'])
                
                # Determine trend
                trend = self.determine_trend(df)
                
                # Calculate support level (recent low)
                support = df['Low'][-20:].min()
                
                # Generate recommendation
                recommendation = self.generate_recommendation(
                    rsi, trend, change_30d, current_price, sma_10, macd_line, signal_line
                )
                
                self.analysis_results[symbol] = {
                    'current_price': round(current_price, 2),
                    'change_7d': round(change_7d, 2),
                    'change_30d': round(change_30d, 2),
                    'change_90d': round(change_90d, 2),
                    'change_365d': round(change_365d, 2),
                    'rsi': round(rsi, 2),
                    'sma_10': round(sma_10, 2),
                    'macd': round(macd_line, 2),
                    'signal': round(signal_line, 2),
                    'trend': trend,
                    'support': round(support, 2),
                    'recommendation': recommendation
                }
                
                print(f"Analysis completed for {symbol}")
            else:
                print(f"No data available for analysis of {symbol}")

    def calculate_rsi(self, prices, periods=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:periods] = 100. - 100./(1.+rs)

        for i in range(periods, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(periods-1) + upval)/periods
            down = (down*(periods-1) + downval)/periods
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi[-1]

    def determine_trend(self, df):
        """Determine price trend"""
        if len(df) >= 20:
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            
            if df['Close'][-1] > ma20[-1] > ma50[-1]:
                return 'UPTREND'
            elif df['Close'][-1] < ma20[-1] < ma50[-1]:
                return 'DOWNTREND'
        return 'SIDEWAYS'

    def generate_recommendation(self, rsi, trend, change_30d, price, sma_10, macd, signal):
        """Generate trading recommendation based on multiple indicators"""
        signals = []
        
        # RSI signals
        if rsi > 70:
            signals.append('OVERBOUGHT')
        elif rsi < 30:
            signals.append('OVERSOLD')
            
        # Trend and SMA signals
        if price > sma_10 and trend == 'UPTREND':
            signals.append('BUY')
        elif price < sma_10 and trend == 'DOWNTREND':
            signals.append('SELL')
            
        # MACD signals
        if macd > signal:
            signals.append('BUY')
        elif macd < signal:
            signals.append('SELL')
            
        # Combine signals
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if 'OVERBOUGHT' in signals:
            return 'OVERBOUGHT'
        elif 'OVERSOLD' in signals:
            return 'OVERSOLD'
        elif buy_signals > sell_signals:
            return 'BUY'
        elif sell_signals > buy_signals:
            return 'SELL'
        return 'HOLD'

class AnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tech Stock Analyzer')
        self.setGeometry(100, 100, 1400, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header controls
        header = QHBoxLayout()
        refresh_btn = QPushButton('Refresh Analysis')
        refresh_btn.clicked.connect(self.refresh_analysis)
        header.addWidget(refresh_btn)
        
        self.last_update_label = QLabel('Last Update: Never')
        header.addWidget(self.last_update_label)
        layout.addLayout(header)
        
        # Analysis table
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(11)
        self.analysis_table.setHorizontalHeaderLabels([
            'Symbol', '7d Change %', '30d Change %', '90d Change %', 'YTD Change %',
            'RSI', '10d SMA', 'MACD', 'Trend', 'Support', 'Recommendation'
        ])
        layout.addWidget(self.analysis_table)
        
        # Initial analysis
        self.refresh_analysis()
        
        # Auto-refresh timer (5 minutes)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_analysis)
        self.timer.start(300000)

    def refresh_analysis(self):
        """Refresh all analysis data"""
        try:
            print("Starting analysis refresh...")
            self.analyzer.update_data()
            self.analyzer.analyze_stocks()
            self.update_table()
            
            self.last_update_label.setText(
                f'Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            print("Analysis refresh completed")
        except Exception as e:
            print(f"Error refreshing analysis: {e}")

    def update_table(self):
        """Update the analysis table with latest results"""
        results = self.analyzer.analysis_results
        self.analysis_table.setRowCount(len(results))
        
        for i, (symbol, data) in enumerate(results.items()):
            try:
                # Create and set table items
                items = [
                    QTableWidgetItem(symbol),
                    QTableWidgetItem(f"{data['change_7d']}%"),
                    QTableWidgetItem(f"{data['change_30d']}%"),
                    QTableWidgetItem(f"{data['change_90d']}%"),
                    QTableWidgetItem(f"{data['change_365d']}%"),
                    QTableWidgetItem(f"{data['rsi']}"),
                    QTableWidgetItem(f"${data['sma_10']}"),
                    QTableWidgetItem(f"{data['macd']:.2f}"),
                    QTableWidgetItem(data['trend']),
                    QTableWidgetItem(f"${data['support']}"),
                    QTableWidgetItem(data['recommendation'])
                ]
                
                # Set items in table
                for col, item in enumerate(items):
                    self.analysis_table.setItem(i, col, item)
                
                # Color code the recommendation
                rec_item = self.analysis_table.item(i, 10)
                if data['recommendation'] == 'BUY':
                    rec_item.setBackground(Qt.GlobalColor.green)
                elif data['recommendation'] == 'SELL':
                    rec_item.setBackground(Qt.GlobalColor.red)
                elif data['recommendation'] in ['OVERBOUGHT', 'OVERSOLD']:
                    rec_item.setBackground(Qt.GlobalColor.yellow)
                
            except Exception as e:
                print(f"Error updating table row for {symbol}: {e}")
        
        # Adjust column widths
        self.analysis_table.resizeColumnsToContents()
        print("Table updated successfully")

def main():
    app = QApplication(sys.argv)
    ex = AnalyzerGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
