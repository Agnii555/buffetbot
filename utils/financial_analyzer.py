# Copy this into utils/financial_analyzer.py

"""Warren Buffett Financial Analysis Tools."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class BuffettFinancialAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.stock = yf.Ticker(self.symbol)
    
    def get_buffett_ratios(self):
        """Calculate key Buffett ratios."""
        try:
            financials = self.stock.financials
            latest_year = financials.columns[0]
            
            gross_profit = financials.loc['Gross Profit', latest_year]
            total_revenue = financials.loc['Total Revenue', latest_year]
            net_income = financials.loc['Net Income', latest_year]
            
            ratios = {}
            
            # Gross Margin
            ratios['Gross Margin'] = {
                'value': gross_profit / total_revenue,
                'rule': '≥ 40%',
                'passes': (gross_profit / total_revenue) >= 0.40
            }
            
            # Net Margin  
            ratios['Net Profit Margin'] = {
                'value': net_income / total_revenue,
                'rule': '≥ 20%',
                'passes': (net_income / total_revenue) >= 0.20
            }
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return {}
    
    def get_buffett_score(self):
        """Get overall score."""
        ratios = self.get_buffett_ratios()
        total = len(ratios)
        passed = sum(1 for r in ratios.values() if r.get('passes', False))
        return passed, total