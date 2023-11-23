from flask import Flask

app = Flask(__name__)

# ... Cấu hình ứng dụng Flask của bạn ở đây ...

if __name__ == '__main__':
    context = ('/etc/letsencrypt/live/api.hutech.click/fullchain.pem', '/etc/letsencrypt/live/api.hutech.click/privkey.pem')
    app.run(ssl_context=context)
