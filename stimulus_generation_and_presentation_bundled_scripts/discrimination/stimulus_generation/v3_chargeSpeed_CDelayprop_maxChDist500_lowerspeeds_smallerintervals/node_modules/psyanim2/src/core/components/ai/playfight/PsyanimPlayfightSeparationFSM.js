import PsyanimFSM from '../PsyanimFSM.js';

import PsyanimPlayfightSeparationState from './PsyanimPlayfightSeparationState.js';

export default class PsyanimPlayfightSeparationFSM extends PsyanimFSM {

    target;

    maxSeparationSpeed;
    maxSeparationAcceleration;

    constructor(entity) {

        super(entity);

        this.maxSeparationSpeed = 12;
        this.maxSeparationAcceleration = 0.5;

        this._separationState = this.addState(PsyanimPlayfightSeparationState);

        this.initialState = this._separationState;
    }

    afterCreate() {

        super.afterCreate();

        this._separationState.target = this.target;

        this._separationState.maxSpeed = this.maxSeparationSpeed;
        this._separationState.maxAcceleration = this.maxSeparationAcceleration;
    }

    onPause() {

        super.onPause();
    }

    onStop() {

        super.onStop();
    }

    onResume() {

        super.onResume();
    }

    update(t, dt) {

        super.update(t, dt);
    }
}