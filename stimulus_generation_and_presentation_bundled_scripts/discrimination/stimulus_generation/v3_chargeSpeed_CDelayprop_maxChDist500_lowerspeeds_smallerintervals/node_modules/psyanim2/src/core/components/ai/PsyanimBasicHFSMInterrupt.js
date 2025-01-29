export default class PsyanimBasicHFSMInterrupt {

    constructor(sourceFSM, variableKey, condition, destinationFSM = null) {

        this._sourceFSM = sourceFSM;
        this._destinationFSM = destinationFSM;
        this._variableKey = variableKey;
        this._condition = condition;
    }

    get sourceFSM() {

        return this._sourceFSM;
    }

    get destinationFSM() {

        return this._destinationFSM;
    }
    
    get isTriggered() {

        let variableValue = this._sourceFSM.getStateVariable(this._variableKey);

        return this._condition(variableValue);
    }
}