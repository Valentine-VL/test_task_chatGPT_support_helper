import json
import random

from bottle import route, run, template, request

from ask_gpt import gpt_answer


with open("FAQ.json", "r") as text:
    FAQ_SOURCE = json.load(text)

@route('/', method='GET')
def index():
    random_question = random.choice(FAQ_SOURCE)['Question_short']
    return template('chat_window.html', question=random_question)

@route('/send_msg', method='POST')
def chat_with_bot():
    message = request.json.get("message")
    try:
        answer = gpt_answer(message)
    except:
        answer = "Something went wrong"
    return answer

run(host='localhost', port=8080)

