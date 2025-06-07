import requests

r = requests.get('https://apidorlombar-5bfc08d7688c.herokuapp.com/models')
print(r.text)