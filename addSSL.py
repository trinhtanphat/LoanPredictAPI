from flask import Flask

app = Flask(__name__)

# ... Cấu hình ứng dụng Flask của bạn ở đây ...

if __name__ == '__main__':
    context = ('/home/ubuntu/LoanPredictAPI/fullchain.pem', '/home/ubuntu/LoanPredictAPI/privkey.pem')
    app.run(ssl_context=context)
