import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimExperimentTimer extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this._callback = null;
        this._callbackArgs = null;
        this._callbackScope = null;

        this._currentTimer = null;
    }

    setOnTimerElapsed(callback, args = [], callbackScope = null) {

        this._callback = callback;
        this._callbackArgs = args;
        this._callbackScope = callbackScope;
    }

    start(duration) {

        if (this.isRunning)
        {
            this.stop();
        }

        this._currentTimer = this.entity.scene.time.delayedCall(
            duration, this._callback, this._args, this._callbackScope
        );
    }

    get isRunning() {

        return this._currentTimer != null;
    }

    stop() {

        if (this._currentTimer == null)
        {
            return;
        }

        this._currentTimer.paused = true;
        this._currentTimer.remove();
        this._currentTimer = null;
    }
}