import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
# import openai
import google.generativeai as genai
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

    # Other methods remain unchanged

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

    # Other methods remain unchanged

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
            
            # Prepare the messages for Google Gemini
            messages = [
                {"role": "system", "content": """You are a helpful trading assistant. Analyze the provided market data and user questions to provide trading advice. 
                Always consider risk management and diversification in your recommendations. Remind users that this is educational and they should do their own research."""},
                {"role": "user", "content": f"Current market context:\n{stock_context}\n\nUser question: {message}"}
            ]
            
            # Get response from Google Gemini
            response = gemini_sdk.ChatCompletion.create(
                model="gemini-llm",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Display Google Gemini response
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
    # Set your Google Gemini API key
    gemini_sdk.api_key = 'YOUR-GOOGLE-GEMINI-API-KEY'
    
    app = QApplication(sys.argv)
    ex = AnalyzerGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
