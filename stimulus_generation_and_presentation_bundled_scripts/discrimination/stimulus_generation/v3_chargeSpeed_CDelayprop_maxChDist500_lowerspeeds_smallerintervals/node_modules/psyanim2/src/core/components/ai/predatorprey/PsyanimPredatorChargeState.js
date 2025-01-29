import PsyanimFSMState from "../PsyanimFSMState.js";

import PsyanimVehicle from "../../steering/PsyanimVehicle.js";
import PsyanimArriveBehavior from "../../steering/PsyanimArriveBehavior.js";

import PsyanimPredatorWanderState from "./PsyanimPredatorWanderState.js";

import { PsyanimUtils } from 'psyanim-utils';

export default class PsyanimPredatorChargeState extends PsyanimFSMState {

    target;

    subtlety; // degrees
    subtletyLag; // ms

    averageChargeDuration;
    chargeDurationVariance;

    constructor(fsm) {

        super(fsm);

        console.warn("TODO: make sure the predator never chooses a direction that's into a wall!");

        this.subtlety = 30;
        this.subtletyLag = 500;

        this.averageChargeDuration = 1800;
        this.chargeDurationVariance = 500;

        this._subtletyAngle = 0;
        this._subtletyUpdateTimer = 0;
        this._arriveTarget = this.entity.scene.addEntity(this.entity.name + "_arriveTarget");

        this.fsm.setStateVariable('chargeDuration', 0);

        this.addTransition(PsyanimPredatorWanderState, 'chargeDuration', this._canTransitionToWanderState.bind(this));
    }

    _canTransitionToWanderState(chargeDuration) {

        return chargeDuration > this._chargeDuration;
    }

    _recomputeSubtletyAngle() {

        this._subtletyAngle = this.subtlety * (2 * Math.random() - 1);
    }

    _recomputeChargeDuration() {

        this._chargeDuration = PsyanimUtils.getRandomInt(
            this.averageChargeDuration - this.chargeDurationVariance,
            this.averageChargeDuration + this.chargeDurationVariance
        );
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._arriveBehavior = this.entity.getComponent(PsyanimArriveBehavior);

        // use an initial angle and subtlety that's randomized
        this.entity.setAngle(360 * (2 * Math.random() - 1));
    }

    enter() {

        super.enter();

        this.fsm.setStateVariable('chargeDuration', 0);

        this._recomputeChargeDuration();

        this._recomputeSubtletyAngle();

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0xff0000;
        }
    }

    exit() {

        super.exit();
    }

    _calculateSeekTargetPosition() {

        let targetRelativePosition = this.target.position
            .subtract(this.entity.position);

        targetRelativePosition.rotate(this._subtletyAngle * Math.PI / 180);

        return this.entity.position
            .add(targetRelativePosition);
    }

    _updateSeekTargetLocation() {

        let newTargetPosition = this._calculateSeekTargetPosition();

        while (!this.entity.scene.screenBoundary.isPointInBounds(newTargetPosition))
        {
            this._recomputeSubtletyAngle();
            newTargetPosition = this._calculateSeekTargetPosition();
        }

        this._arriveTarget.position = newTargetPosition;
    }

    _updateSubtlety(dt) {

        this._subtletyUpdateTimer += dt;

        if (this._subtletyUpdateTimer > this.subtletyLag)
        {
            this._subtletyUpdateTimer = 0;

            this._recomputeSubtletyAngle();
        }
    }

    run(t, dt) {

        super.run(t, dt);

        this._updateSubtlety(dt);
        this._updateSeekTargetLocation();

        let updatedChargeDuration = this.fsm.getStateVariable('chargeDuration') + dt;

        this.fsm.setStateVariable('chargeDuration', updatedChargeDuration);

        let steering = this._arriveBehavior.getSteering(this._arriveTarget);

        this._vehicle.steer(steering);
    }
}