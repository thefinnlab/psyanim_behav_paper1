import PsyanimComponent from '../../PsyanimComponent.js';
import PsyanimFSMState from './PsyanimFSMState.js';

import { PsyanimDebug } from 'psyanim-utils';

export default class PsyanimFSM extends PsyanimComponent {

    /**
     *  The initial state of the FSM is the first state the FSM will enter() on start.
     *  @type {PsyanimFSMState}
     */
    initialState;

    /**
     *  If true, this FSM will run in debug mode.
     *  @type {boolean}
     */
    debugLogging;

    constructor(entity) {

        super(entity);

        this.debugLogging = false;
        this.debugGraphics = false;

        this._stateVariables = {};
        this._states = [];

        this._initialized = false;

        this._running = true;

        /**
         *  Event emitter for this FSM.
         *  @type {Phaser.Events.EventEmitter}
         */
        this.events = new Phaser.Events.EventEmitter();
    }

    get currentStateName() {

        return this._currentState.constructor.name;
    }

    getStateVariableSnapshot() {

        let snapshot = {};

        let stateKeys = Object.keys(this._stateVariables);

        // copy the whole object
        for (let i = 0; i < stateKeys.length; ++i)
        {
            let key = stateKeys[i];

            snapshot[key] = this._stateVariables[key];
        }

        return snapshot;
    }

    stringifyStateVariables() {

        return JSON.stringify(this._stateVariables, null, 4);
    }

    getState(stateType) {

        return this._states.find(state => state instanceof stateType);
    }

    hasState(stateType) {

        if (this.getState(stateType))
        {
            return true;
        }

        return false;
    }

    addState(stateType) {

        if (this.hasState(stateType))
        {
            console.error('state type already exists on this FSM: ', stateType.constructor.name);
            return;
        }

        let newState = new stateType(this);

        this._states.push(newState);

        return newState;
    }

    /**
     *  The state that is currently executing.
     *  @type {PsyanimFSMState}
     */
    get currentState() {

        return this._currentState;
    }

    afterCreate() {

        this._currentState = this.initialState;

        super.afterCreate();

        // call after create on each state
        for (let i = 0; i < this._states.length; ++i)
        {
            this._states[i].afterCreate();
        }
    }

    stateVariableExists(key) {

        return Object.hasOwn(this._stateVariables, key);
    }

    setStateVariable(key, value) {

        this._stateVariables[key] = value;
    }

    getStateVariable(key) {

        return this._stateVariables[key];
    }

    pause() {

        if (this._running)
        {
            this.onPause();

            this._running = false;
        }
    }

    stop() {

        if (this._running)
        {
            this.onStop();

            if (this._currentState.stage != PsyanimFSMState.STAGE.EXITED)
            {
                this._currentState.exit();
            }
    
            this._currentState = this.initialState;
    
            this._running = false;
    
            this._initialized = false;
        }
    }

    restart() {

        this.stop();
        this.resume();
    }

    resume() {

        if (this._running === false)
        {
            this.onResume();

            this._running = true;
        }
    }

    onPause() {

        this._currentState.onPause();
    }

    onStop() {

        this._currentState.onStop();
    }

    onResume() {

        this._currentState.onResume();
    }

    update(t, dt) {

        super.update(t, dt);

        if (!this._running)
        {
            return;
        }

        // we don't enter the 'initialState' until the first update() runs
        if (!this._initialized)
        {
            this._currentState.enter();
            this._initialized = true;
        }

        // see if we have any transition conditions met
        let currentTransition = null;

        for (let i = 0; i < this._currentState.transitions.length; ++i)
        {
            let t = this._currentState.transitions[i];

            if (t.isTriggered)
            {
                currentTransition = t;
                break;
            }
        }

        // if a transition is triggered, switch states
        if (currentTransition)
        {
            this._currentState.exit();

            this.events.emit('exit', this._currentState.name);

            let targetState = this.getState(currentTransition.targetStateType);

            targetState.enter();

            this.events.emit('enter', targetState.name);

            this._currentState = targetState;
        }

        // let the current state run
        this._currentState.run(t, dt);
    }
}