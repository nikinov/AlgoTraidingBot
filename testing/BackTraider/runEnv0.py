import os, sys, argparse
import pandas as pd
import backtrader as bt
from Strategy0 import GoldenCross
from Strategy1 import BuyHold

strategies = {
    'golden_cross': GoldenCross,
    'buy_hold': BuyHold
}

#parser = argparse.ArgumentParser()
#parser.add_argument('strategy', help='which strategy to run', type=str)
#args = parser.parse_args()

#if not args.strategy in strategies:
#    print('invalid strategy, must be one of {}'.format(strategies.keys()))
#    sys.exit()

cerebro = bt.Cerebro()
cerebro.broker.setcash(1000000)

apple_prices = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

feed = bt.feeds.PandasData(dataname=apple_prices)
cerebro.adddata(feed)

cerebro.addstrategy(BuyHold)
print('cash: '+str(cerebro.broker.cash))
cerebro.run()
print('cash: '+str(cerebro.broker.cash))
cerebro.plot()