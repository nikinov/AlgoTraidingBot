import backtrader as bt

ibstore = bt.stores.IBStore(host='127.0.0.1', port=7496, clientId=35)


data = ibstore.getdata(dataname='AAPL', timeframe=bt.TimeFrame.Seconds, compression=5)

# Create a cerebro entity
cerebro = bt.Cerebro()

cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=2)
print(cerebro.datas)
cerebro.run()