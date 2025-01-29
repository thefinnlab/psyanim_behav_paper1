import PsyanimFSMState from '../PsyanimFSMState.js';

import { PsyanimDebug } from 'psyanim-utils';

import PsyanimWanderBehavior from '../../steering/PsyanimWanderBehavior.js';

import PsyanimPlayfightChargeState from './PsyanimPlayfightChargeState.js';
import PsyanimVehicle from '../../steering/PsyanimVehicle.js';

export default class PsyanimPlayfightChargeDelayState extends PsyanimFSMState {

    averageDelay;
    delayVariance;

    constructor(fsm) {

        super(fsm);

        this.averageDelay = 600;
        this.delayVariance = 400;

        this._delay = this.averageDelay;

        this.fsm.setStateVariable('timer', 0);

        this.addTransition(PsyanimPlayfightChargeState, 'timer', (value) => value > this._delay);
    }

    _isTimerElapsed(timerValue) {

        return timerValue > this._delay;
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._wanderBehavior = this.entity.getComponent(PsyanimWanderBehavior);

        this._delay = this.averageDelay;
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

    enter() {

        super.enter();

        this.fsm.setStateVariable('timer', 0);

        this._delay = (2 * Math.random() - 1) * this.delayVariance + this.averageDelay;

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0xfc9803;
        }
    }

    exit() {

        super.exit();
    }

    run(t, dt) {

        super.run(t, dt);

        // update state variables
        let updatedTimer = this.fsm.getStateVariable('timer') + dt;

        this.fsm.setStateVariable('timer', updatedTimer);

        // apply steering
        let steering = this._wanderBehavior.getSteering();

        this._vehicle.steer(steering);
    }
}