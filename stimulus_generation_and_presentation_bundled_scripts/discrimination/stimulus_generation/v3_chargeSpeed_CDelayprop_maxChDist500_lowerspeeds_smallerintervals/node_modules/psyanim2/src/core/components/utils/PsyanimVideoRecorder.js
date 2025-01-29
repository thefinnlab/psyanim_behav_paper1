import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimVideoRecorder extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this.events = new Phaser.Events.EventEmitter();

        this._chunks = [];

        this._state = PsyanimVideoRecorder.STATE.READY_TO_RECORD;

        // setup media recorder
        this._mediaRecorder = new MediaRecorder(this.entity.scene.
            game.canvas.captureStream(60));

        this._mediaRecorder.ondataavailable = (e) => {
            this._chunks.push(e.data);
        };

        this._mediaRecorder.onstop = (e) => {

            this._state = PsyanimVideoRecorder.STATE.VIDEO_AVAILABLE;

            this.events.emit('videoFileReady');
        };
    }

    start() {

        this._state = PsyanimVideoRecorder.STATE.RECORDING;

        this._chunks = [];

        this._mediaRecorder.start();
    }

    stop() {

        this._mediaRecorder.stop();
    }

    getVideoBlob() {

        return new Blob(this._chunks, { type: "video/webm" });
    }

    update(t, dt) {

        super.update(t, dt);
    }
}

PsyanimVideoRecorder.STATE = {
    READY_TO_RECORD: 0x0001,
    RECORDING: 0x0002,
    VIDEO_READY: 0x0004
};