from flask import Flask, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# known states of each client drone. True means still live.
client_states = {0: 'alive', 1: 'alive', 2: 'alive'}


# Event handler for when a client connects
@socketio.on('connect')
def handle_connect():
    print('Client connected')


# Event handler for relaying information to all clients
def relay_info_to_clients(data):
    emit('info', data, broadcast=True)


@app.route('/notify', methods=['POST'])
def inform_server():
    # Get data from the request
    notification = request.json['notification']
    client_id = request.json['id']
    client_states.update({client_id, notification})
    print("here is everyone...")
    print(client_states)
    return 'notification received: {}'.format(notification)


# Example function to simulate relaying information
def simulate_information_relay():
    # Simulate some data
    # Relay the data to all clients
    relay_info_to_clients(client_states)


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)
