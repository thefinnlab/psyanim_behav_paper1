import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimClientNetworkManager extends PsyanimComponent {

    host;
    port;

    constructor(entity) {

        super(entity);

        this.host = "localhost";
        this.port = 3000;

        this.events = new Phaser.Events.EventEmitter();
    }

    connect(host = null, port = null) {

        if (host)
        {
            this.host = host;
        }

        if (port)
        {
            this.port = port;
        }

        this._ws = new WebSocket("ws://" + this.host + ":" + this.port);
        this._ws.binaryType = "blob";

        this._ws.addEventListener('open', (event) => {
            this.events.emit('open');
        });
    }

    disconnect() {

        if (this._ws)
        {
            this._ws.close();
            this._ws = null;
        }
    }

    doGet(path, callback) {

        let xhttp = new XMLHttpRequest();

        xhttp.onreadystatechange = () => {

            if (xhttp.readyState == 4 && xhttp.status == 200)
            {
                callback(xhttp.responseText);
            }
        };

        xhttp.open("GET", path, true);
        xhttp.send();
    }

    doPost(path, data) {

        let xhttp = new XMLHttpRequest();
        xhttp.open('POST', path);
        xhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhttp.send(data);
    }

    send(data) {

        this._ws.send(data);
    }

    sendBlob(data) {

        this._ws.send(data);
    }

    update(t, dt) {

        super.update(t, dt);

    }
}