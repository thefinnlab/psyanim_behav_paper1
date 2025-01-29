import { PsyanimDebug } from "psyanim-utils";

import PsyanimFSMStateTransition from './PsyanimFSMStateTransition.js';

export default class PsyanimFSMState {

    constructor(fsm) {

        this._entity = fsm.entity;

        this._fsm = fsm;
        this._stage = PsyanimFSMState.STAGE.ENTERING;

        this._transitions = [];
    }

    get name() {

        return this.constructor.name;
    }

    get fsm() {

        return this._fsm;
    }

    get entity() {

        return this._entity;
    }

    get stage() {
        
        return this._stage;
    }

    get transitions() {

        return this._transitions;
    }

    get isActive() {

        return this._stage === PsyanimFSMState.STAGE.RUNNING;
    }

    getTransition(targetStateType, variableKey) {

        return this._transitions.find(t => {
            return t.targetStateType instanceof targetStateType
                && t.variableKey === variableKey
        });
    }

    hasTransition(targetStateType, variableKey) {

        if (this.getTransition(targetStateType, variableKey))
        {
            return true;
        }

        return false;
    }

    addTransition(targetStateType, variableKey, condition) {

        if (this.hasTransition(targetStateType, variableKey))
        {
            PsyanimDebug.error('transition already exists: targetState = ', typeof(targetStateType), ', key = ', variableKey);
            return;
        }

        let transition = new PsyanimFSMStateTransition(this._fsm, 
            targetStateType, variableKey, condition);

        this._transitions.push(transition);

        return transition;
    }

    afterCreate() {

    }

    enter() {

        this._stage = PsyanimFSMState.STAGE.RUNNING;

        if (this.fsm.debugLogging)
        {
            PsyanimDebug.log(this._entity.name, 'ENTERING state: ', this.constructor.name);
        }
    }

    exit() {

        this._stage = PsyanimFSMState.STAGE.EXITED;

        if (this.fsm.debugLogging)
        {
            PsyanimDebug.log(this._entity.name, 'EXITING state: ', this.constructor.name);
        }
    }

    onPause() {

    }

    onStop() {

    }

    onResume() {
        
    }

    run(t, dt) {

        if (this.fsm.debugLogging)
        {
            PsyanimDebug.log(this._entity.name, 'state RUNNING: ', this.constructor.name);            
        }
    }
}

PsyanimFSMState.STAGE = {
    ENTERING: 0x0001,
    RUNNING: 0x0002,
    EXITED:  0x0003
};