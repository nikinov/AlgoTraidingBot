import math
import backtrader as bt

class GoldenCross(bt.Strategy):
    params = (('fast', 90), ('slow', 160), ('order_percentage', 0.95), ('ticker', 'AAPL'))

    def __init__(self):
        self.fast_Moving_average = bt.indicators.SMA(
            self.data.close, period=self.params.fast, plotname=str(self.params.fast)+' day moving average'
        )
        self.slow_Moving_average = bt.indicators.SMA(
            self.data.close, period=self.params.slow, plotname=str(self.params.slow)+' day moving average'
        )

        self.crossover = bt.indicators.CrossOver(self.fast_Moving_average, self.slow_Moving_average)

    def next(self):
        if self.position.size == 0 and self.crossover > 0:
            amount_to_invest = self.params.order_percentage * self.broker.cash
            self.size = math.floor(amount_to_invest/self.data.close)

            print('Buy '+ str(self.size) + ' shares of ' + str(self.params.ticker) + ' at ' + str(self.data.close[0]))

            self.buy(size=self.size)

        elif self.position.size > 0 and self.crossover < 0:
            print('Sell '+ str(self.size) + ' shares of ' + str(self.params.ticker) + ' at ' + str(self.data.close[0]))
            self.close()

