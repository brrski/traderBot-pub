import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import openai
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                            QTableWidgetItem, QTabWidget, QTextEdit, QLineEdit,
                            QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor

class StockAnalyzer:
    def __init__(self):
        self.symbols = ['TSLA', 'AMD', 'NVDA', 'META', 'AAPL', 'QQQ', 'SPY', 'TSM', 'PLTR']
        self.price_data = {}
        self.analysis_results = {}
        self.options_data = {}
        self.update_data()

    def get_atm_options(self, ticker, current_price):
        """Get at-the-money call and put options"""
        try:
            # Get options expirations
            expirations = ticker.options
            
            if not expirations:
                print(f"No options expirations found")
                return None, None
            
            print(f"Found expirations: {expirations[0]}")
            
            # Get the nearest expiration date
            nearest_expiry = expirations[0]
            
            # Get the options chain
            options = ticker.option_chain(nearest_expiry)
            if options is None:
                print("No options chain returned")
                return None, None
                
            calls = options.calls
            puts = options.puts
            
            print(f"Calls shape: {calls.shape}, Puts shape: {puts.shape}")
            
            # Find the strike prices closest to current price
            call_strikes = calls['strike'].values
            put_strikes = puts['strike'].values
            
            # Find nearest strike prices at or above and below current price
            strikes_above = call_strikes[call_strikes >= current_price]
            strikes_below = call_strikes[call_strikes <= current_price]
            
            if len(strikes_above) > 0:
                nearest_above = strikes_above[0]
            else:
                nearest_above = None
                
            if len(strikes_below) > 0:
                nearest_below = strikes_below[-1]
            else:
                nearest_below = None
                
            print(f"Current price: {current_price}")
            print(f"Nearest strike above: {nearest_above}")
            print(f"Nearest strike below: {nearest_below}")
            
            # Determine which strike price is closer to current price
            if nearest_above is not None and nearest_below is not None:
                diff_above = abs(nearest_above - current_price)
                diff_below = abs(nearest_below - current_price)
                atm_strike = nearest_above if diff_above <= diff_below else nearest_below
            elif nearest_above is not None:
                atm_strike = nearest_above
            elif nearest_below is not None:
                atm_strike = nearest_below
            else:
                return None, None
                
            print(f"Selected ATM strike: {atm_strike}")
            
            # Get the call and put prices for the selected strike
            atm_call = float(calls[calls['strike'] == atm_strike]['lastPrice'].iloc[0])
            atm_put = float(puts[puts['strike'] == atm_strike]['lastPrice'].iloc[0])
            
            # Get additional information for debugging
            call_volume = int(calls[calls['strike'] == atm_strike]['volume'].iloc[0])
            put_volume = int(puts[puts['strike'] == atm_strike]['volume'].iloc[0])
            call_open_interest = int(calls[calls['strike'] == atm_strike]['openInterest'].iloc[0])
            put_open_interest = int(puts[puts['strike'] == atm_strike]['openInterest'].iloc[0])
            
            print(f"ATM Call: ${atm_call:.2f} (Volume: {call_volume}, OI: {call_open_interest})")
            print(f"ATM Put: ${atm_put:.2f} (Volume: {put_volume}, OI: {put_open_interest})")
            
            return atm_call, atm_put
            
        except Exception as e:
            print(f"Error in get_atm_options: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None

    def update_data(self):
        """Fetch and update price data for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data

        # Format dates as strings in YYYY-MM-DD format
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
    
        for symbol in self.symbols:
            try:
                print(f"\nFetching data for {symbol}")
                ticker = yf.Ticker(symbol.strip())  # Added strip() to handle whitespace
                df = ticker.history(start=start_date_str, end=end_date_str)
            
                if not df.empty:
                    self.price_data[symbol] = df
                    current_price = float(df['lastPrice'])
                    print(f"Current price for {symbol}: {current_price}")
                
                    # Fetch options data
                    atm_call, atm_put = self.get_atm_options(ticker, current_price)
                
                    self.options_data[symbol] = {
                        'current_price': current_price,
                        'atm_call': atm_call,
                     'atm_put': atm_put
                 }
                
                    print(f"Data fetched for {symbol}:")
                    print(f"Price: ${current_price:.2f}")
                    print(f"ATM Call: ${atm_call if atm_call is not None else 'N/A'}")
                    print(f"ATM Put: ${atm_put if atm_put is not None else 'N/A'}")
                else:
                    print(f"No data received for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd[-1], signal[-1]

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

    def analyze_stocks(self):
        """Perform analysis on all stocks"""
        for symbol in self.symbols:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                df = self.price_data[symbol]
                options = self.options_data.get(symbol, {})
                
                # Get current price from options data or calculate from df
                current_price = options.get('current_price', float(df['Close'][-1]))
                
                # Calculate returns for different periods
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
                    'atm_call': options.get('atm_call'),
                    'atm_put': options.get('atm_put'),
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
                
                print(f"\nAnalysis results for {symbol}:")
                print(f"Current Price: ${self.analysis_results[symbol]['current_price']}")
                print(f"ATM Call: ${self.analysis_results[symbol]['atm_call']}")
                print(f"ATM Put: ${self.analysis_results[symbol]['atm_put']}")
                
            else:
                print(f"No data available for analysis of {symbol}")

class ChatWidget(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.chat_history = []
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        send_button = QPushButton('Send')
        send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
        
    def format_stock_context(self):
        """Format current stock analysis for context"""
        context = "Current market analysis:\n"
        for symbol, data in self.analyzer.analysis_results.items():
            context += f"\n{symbol}: ${data['current_price']} | {data['trend']} | RSI: {data['rsi']} | Recommendation: {data['recommendation']}"
        return context
        
    def send_message(self):
        message = self.chat_input.text().strip()
        if not message:
            return
            
        # Clear input
        self.chat_input.clear()
        
        # Display user message
        self.chat_display.append(f"\nYou: {message}")
        
        try:
            # Prepare context with current market data
            stock_context = self.format_stock_context()
            
            # Prepare the messages for ChatGPT
            messages = [
                {"role": "system", "content": """You are a helpful trading assistant. Analyze the provided market data and user questions to provide trading advice. 
                Always consider risk management and diversification in your recommendations. Remind users that this is educational and they should do their own research."""},
                {"role": "user", "content": f"Current market context:\n{stock_context}\n\nUser question: {message}"}
            ]
            
            # Get response from ChatGPT
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Display ChatGPT response
            assistant_response = response.choices[0].message.content
            self.chat_display.append(f"\nAssistant: {assistant_response}")
            
        except Exception as e:
            self.chat_display.append(f"\nError: Could not get response - {str(e)}")
        
        # Scroll to bottom
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

class AnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tech Stock Analyzer')
        self.setGeometry(100, 100, 1600, 800)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()
        
        # Header controls
        header = QHBoxLayout()
        refresh_btn = QPushButton('Refresh Analysis')
        refresh_btn.clicked.connect(self.refresh_analysis)
        header.addWidget(refresh_btn)
        
        self.last_update_label = QLabel('Last Update: Never')
        header.addWidget(self.last_update_label)
        analysis_layout.addLayout(header)
        
        # Analysis table
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(14)
        self.analysis_table.setHorizontalHeaderLabels([
            'Symbol', 'Current Price', 'ATM Call', 'ATM Put', '7d Change %', 
            '30d Change %', '90d Change %', 'YTD Change %', 'RSI', '10d SMA', 
            'MACD', 'Trend', 'Support', 'Recommendation'
        ])
        analysis_layout.addWidget(self.analysis_table)
        
        analysis_tab.setLayout(analysis_layout)
        self.tabs.addTab(analysis_tab, "Analysis")
        
        # Chat tab
        self.chat_widget = ChatWidget(self.analyzer)
        self.tabs.addTab(self.chat_widget, "Trading Assistant")
        
        # Initial analysis
        self.refresh_analysis()
        
        # Auto-refresh timer (5 minutes)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_analysis)
        self.timer.start(300000)

    def update_table(self):
        """Update the analysis table with latest results"""
        results = self.analyzer.analysis_results
        self.analysis_table.setRowCount(len(results))
        
        for i, (symbol, data) in enumerate(results.items()):
            try:
                # Format options data
                atm_call = f"${data['atm_call']:.2f}" if data['atm_call'] is not None else "N/A"
                atm_put = f"${data['atm_put']:.2f}" if data['atm_put'] is not None else "N/A"
                
                # Create and set table items
                items = [
                    QTableWidgetItem(symbol),
                    QTableWidgetItem(f"${data['current_price']:.2f}"),
                    QTableWidgetItem(atm_call),
                    QTableWidgetItem(atm_put),
                    QTableWidgetItem(f"{data['change_7d']:.2f}%"),
                    QTableWidgetItem(f"{data['change_30d']:.2f}%"),
                    QTableWidgetItem(f"{data['change_90d']:.2f}%"),
                    QTableWidgetItem(f"{data['change_365d']:.2f}%"),
                    QTableWidgetItem(f"{data['rsi']:.2f}"),
                    QTableWidgetItem(f"${data['sma_10']:.2f}"),
                    QTableWidgetItem(f"{data['macd']:.2f}"),
                    QTableWidgetItem(data['trend']),
                    QTableWidgetItem(f"${data['support']:.2f}"),
                    QTableWidgetItem(data['recommendation'])
                ]
                
                # Set items in table
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                    self.analysis_table.setItem(i, col, item)
                
                # Align symbol to left
                self.analysis_table.item(i, 0).setTextAlignment(Qt.AlignmentFlag.AlignLeft)
                
                # Color code the recommendation
                rec_item = self.analysis_table.item(i, 13)
                if data['recommendation'] == 'BUY':
                    rec_item.setBackground(Qt.GlobalColor.green)
                elif data['recommendation'] == 'SELL':
                    rec_item.setBackground(Qt.GlobalColor.red)
                elif data['recommendation'] in ['OVERBOUGHT', 'OVERSOLD']:
                    rec_item.setBackground(Qt.GlobalColor.yellow)
                
            except Exception as e:
                print(f"Error updating table row for {symbol}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Adjust column widths
        self.analysis_table.resizeColumnsToContents()
        print("Table updated successfully")

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
            import traceback
            print(traceback.format_exc())

def main():
    # Set your OpenAI API key
    openai.api_key = 'YOUR-API-KEY'
    
    app = QApplication(sys.argv)
    ex = AnalyzerGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
