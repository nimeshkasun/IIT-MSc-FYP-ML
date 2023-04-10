from flask import Flask, render_template
import arc_controller.CharacterController as characterController

app = Flask(__name__)

app.register_blueprint(characterController.characterBP, url_prefix='/character')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
