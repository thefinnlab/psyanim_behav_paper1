export default class PsyanimFSMStateTransition {

    constructor(fsm, targetStateType, variableKey, condition) {

        this._targetStateType = targetStateType;
        this._fsm = fsm;
        this._variableKey = variableKey;
        this._condition = condition;
    }

    get targetStateType() {

        return this._targetStateType;
    }

    get variableKey() {

        return this._variableKey;
    }

    get isTriggered() {

        let variableValue = this._fsm.getStateVariable(this._variableKey);

        return this._condition(variableValue);
    }
}