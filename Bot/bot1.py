import requests
from bs4 import BeautifulSoup
import string

def Get_Symbols(Aonly = False):

    symbols = []

    # Loop through the letters in the alphabet to get the stocks on each page
    # from the table and store them in a list
    if (Aonly == True):
        alpha = ["A"]
    else:
        alpha = list(string.ascii_uppercase)

    for each in alpha:
        print('getting info from '+'http://eoddata.com/stocklist/NYSE/'+each+'.htm')
        url = 'http://eoddata.com/stocklist/NYSE/'+each+'.htm'
        resp = requests.get(url)
        site = resp.content
        soup = BeautifulSoup(site, 'html.parser')
        table = soup.find('table', {'class': 'quotes'})
        for row in table.findAll('tr')[1:]:
            symbols.append(row.findAll('td')[0].text.rstrip())

            # Remove the extra letters on the end
    #symbols_clean = []

    #for each in symbols:
    #    each = each.replace('.', '-')
     #   symbols_clean.append((each.split('-')[0]))
    return symbols


def chunks(l, n):
    """
    Takes in a list and how long you want
    each chunk to be
    """
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


symbols_chunked = list(chunks(list(set(Get_Symbols(Aonly=True))), 200))
print(symbols_chunked)