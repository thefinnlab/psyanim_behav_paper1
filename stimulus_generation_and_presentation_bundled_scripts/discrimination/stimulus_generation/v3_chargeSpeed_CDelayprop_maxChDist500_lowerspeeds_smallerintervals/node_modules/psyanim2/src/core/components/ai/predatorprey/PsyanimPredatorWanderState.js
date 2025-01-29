import PsyanimFSMState from "../PsyanimFSMState.js";

import PsyanimPredatorChargeState from './PsyanimPredatorChargeState.js';

import PsyanimVehicle from "../../steering/PsyanimVehicle.js";
import PsyanimWanderBehavior from "../../steering/PsyanimWanderBehavior.js";

import { PsyanimUtils } from 'psyanim-utils';

export default class PsyanimPredatorWanderState extends PsyanimFSMState {

    target;

    averageWanderDuration;
    wanderDurationVariance;

    constructor(fsm) {

        super(fsm);

        this.averageWanderDuration = 2000;
        this.wanderDurationVariance = 500;

        this.fsm.setStateVariable('wanderTimer', 0);

        this.addTransition(PsyanimPredatorChargeState, 'wanderTimer', this._canTransitionToChargeState.bind(this));
    }

    _canTransitionToChargeState(wanderTimer) {

        return wanderTimer > this._wanderDuration;
    }

    _recomputeWanderDuration() {

        this._wanderDuration = PsyanimUtils.getRandomInt(
            this.averageWanderDuration - this.wanderDurationVariance,
            this.averageWanderDuration + this.wanderDurationVariance
        );
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._wanderBehavior = this.entity.getComponent(PsyanimWanderBehavior);
    }

    enter() {

        super.enter();

        this._recomputeWanderDuration();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x00ff00;
        }

        this.fsm.setStateVariable('wanderTimer', 0);
    }

    exit() {

        super.exit();
    }

    run(t, dt) {

        super.run(t, dt);

        let updatedWanderTimer = this.fsm.getStateVariable('wanderTimer') + dt;

        this.fsm.setStateVariable('wanderTimer', updatedWanderTimer);

        let steering = this._wanderBehavior.getSteering(this.target);

        this._vehicle.steer(steering);
    }
}