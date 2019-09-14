import requests
proxies = {'http': "socks5://127.0.0.1:1080"}
res = requests.get('https://www.google.com', proxies=proxies,timeout=2)
print(res.status_code)