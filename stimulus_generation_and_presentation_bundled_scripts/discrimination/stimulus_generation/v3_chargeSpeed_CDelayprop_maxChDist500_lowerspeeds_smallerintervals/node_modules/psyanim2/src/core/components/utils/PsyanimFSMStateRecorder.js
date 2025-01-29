import PsyanimComponent from '../../PsyanimComponent.js';

import PsyanimBasicHFSM from '../ai/PsyanimBasicHFSM.js';
import PsyanimFSM from '../ai/PsyanimFSM.js';

import { PsyanimDebug } from 'psyanim-utils';

export default class PsyanimFSMStateRecorder extends PsyanimComponent {

    stateMachine;

    saveResumeEventSnapshot;
    savePauseEventSnapshot;
    saveStopEventSnapshot;
    saveEnterEventSnapshot;
    saveExitEventSnapshot;

    recordOnStart;

    debug;

    constructor(entity) {

        super(entity);

        this._stateBuffer = [];

        this.recordOnStart = true;

        this.saveResumeEventSnapshot = false;
        this.savePauseEventSnapshot = false;
        this.saveStopEventSnapshot = false;
        this.saveEnterEventSnapshot = false;
        this.saveExitEventSnapshot = false;

        this.debug = false;

        this._fsmResumedHandler = this._handleFSMResumed.bind(this);
        this._fsmPausedHandler = this._handleFSMPaused.bind(this);
        this._fsmStoppedHandler = this._handleFSMStopped.bind(this);
        this._fsmEnterHandler = this._handleFSMStateEntered.bind(this);
        this._fsmExitHandler = this._handleFSMStateExited.bind(this);

        this._recording = false;

        this._keys = {
            Q: entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.Q),
            R: entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.R),
            SPACE: entity.scene.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.SPACE),
        };
    }

    get data() {

        return this._stateBuffer;
    }

    record() {

        if (this._recording)
        {
            return;
        }

        if (this.stateMachine instanceof PsyanimBasicHFSM)
        {
            this.stateMachine.events.on('resume', this._fsmResumedHandler);
            this.stateMachine.events.on('pause', this._fsmPausedHandler);
            this.stateMachine.events.on('stop', this._fsmStoppedHandler);
            this.stateMachine.events.on('enter', this._fsmEnterHandler);
            this.stateMachine.events.on('exit', this._fsmExitHandler);
        }
        else if (this.stateMachine instanceof PsyanimFSM)
        {
            this.stateMachine.events.on('enter', this._fsmEnterHandler);
            this.stateMachine.events.on('exit', this._fsmExitHandler);
        }
        else
        {
            PsyanimDebug.error("'stateMachine' must be an instance of PsyanimFSM or PsyanimBasicFSM!");
        }

        this._recording = true;
    }

    stop() {
        
        if (this.stateMachine instanceof PsyanimBasicHFSM)
        {
            this.stateMachine.events.off('resume', this._fsmResumedHandler);
            this.stateMachine.events.off('pause', this._fsmPausedHandler);
            this.stateMachine.events.off('stop', this._fsmStoppedHandler);
            this.stateMachine.events.off('enter', this._fsmEnterHandler);
            this.stateMachine.events.off('exit', this._fsmExitHandler);
        }
        else if (this.stateMachine instanceof PsyanimFSM)
        {
            this.stateMachine.events.off('enter', this._fsmEnterHandler);
            this.stateMachine.events.off('exit', this._fsmExitHandler);
        }
        else
        {
            PsyanimDebug.error("'stateMachine' must be an instance of PsyanimFSM or PsyanimBasicFSM!");
        }

        this._recording = false;
    }

    afterCreate() {

        super.afterCreate();

        if (this.recordOnStart)
        {
            this.record();
        }
    }

    _handleFSMStateEntered(stateName) {

        let data = {
            eventType: 'enter', 
            stateName: stateName,
            t: this.entity.scene.time.now - this.entity.scene.timeSinceLastInit
        };

        if (this.saveEnterEventSnapshot)
        {
            data.stateVariableSnapshot = this.stateMachine.getStateVariableSnapshot();
        }

        this._stateBuffer.push(data);
    }

    _handleFSMStateExited(stateName) {

        let data = {
            eventType: 'exit',
            stateName, stateName,
            t: this.entity.scene.time.now - this.entity.scene.timeSinceLastInit
        };

        if (this.saveExitEventSnapshot)
        {
            data.stateVariableSnapshot = this.stateMachine.getStateVariableSnapshot();
        }

        this._stateBuffer.push(data);
    }

    _handleFSMResumed(fsm) {

        let data = {
            eventType: 'resume',
            stateMachine: fsm.constructor.name,
            currentState: fsm.currentStateName,
            t: this.entity.scene.time.now - this.entity.scene.timeSinceLastInit
        };

        if (this.saveResumeEventSnapshot)
        {
            data.stateVariableSnapshot = this.stateMachine.getStateVariableSnapshot();
        }

        this._stateBuffer.push(data);
    }

    _handleFSMStopped(fsm) {

        let data = {
            eventType: 'stop',
            stateMachine: fsm.constructor.name,
            currentState: fsm.currentStateName,
            t: this.entity.scene.time.now - this.entity.scene.timeSinceLastInit
        };

        if (this.saveStopEventSnapshot)
        {
            data.stateVariableSnapshot = this.stateMachine.getStateVariableSnapshot();
        }

        this._stateBuffer.push(data);
    }

    _handleFSMPaused(fsm) {

        let data = {
            eventType: 'pause',
            stateMachine: fsm.constructor.name,
            currentState: fsm.currentStateName,
            t: this.entity.scene.time.now - this.entity.scene.timeSinceLastInit
        };

        if (this.savePauseEventSnapshot)
        {
            data.stateVariableSnapshot = this.stateMachine.getStateVariableSnapshot();
        }

        this._stateBuffer.push(data);
    }

    update(t, dt) {

        super.update(t, dt);

        if (this.debug)
        {
            if (Phaser.Input.Keyboard.JustDown(this._keys.SPACE))
            {
                let length = JSON.stringify(this._stateBuffer).length;

                console.log('PsyanimFSMStateRecorder buffer length =', length, ', buffer =', this._stateBuffer);
            }

            if (Phaser.Input.Keyboard.JustDown(this._keys.R))
            {
                this.record();
            }

            if (Phaser.Input.Keyboard.JustDown(this._keys.Q))
            {
                this.stop();
            }
        }
    }
}