import PsyanimFSMState from "../PsyanimFSMState.js";

import PsyanimVehicle from "../../steering/PsyanimVehicle.js";
import PsyanimFleeBehavior from "../../steering/PsyanimFleeBehavior.js";

import PsyanimPreyWanderState from "./PsyanimPreyWanderState.js";
import PsyanimPreyWallAvoidanceState from "./PsyanimPreyWallAvoidanceState.js";

export default class PsyanimPreyFleeState extends PsyanimFSMState {

    target;

    subtlety;
    subtletyLag;

    minimumWallSeparation;

    constructor(fsm) {

        super(fsm);

        console.warn("TODO: make sure the predator never chooses a direction that's into a wall!");

        this.subtlety = 30; // degrees
        this.subtletyLag = 500; // ms

        this.minimumWallSeparation = 50;

        this._subtletyAngle = 0;
        this._subtletyUpdateTimer = 0;

        this._fleeTarget = this.entity.scene.addEntity(this.entity.name + '_fleeTarget');

        this.fsm.setStateVariable('distanceToTarget', Infinity);
        this.fsm.setStateVariable('avoidWalls', false);

        this.addTransition(PsyanimPreyWanderState, 'distanceToTarget', this._canTransitionToWander.bind(this));
        this.addTransition(PsyanimPreyWallAvoidanceState, 'avoidWalls', (value) => value === true);
    }

    _recomputeSubtletyAngle() {

        let newAngle = this.subtlety * (2 * Math.random() - 1);

        if (Math.abs(newAngle) < 0.001)
        {
            newAngle = 0;
        }

        this._subtletyAngle = newAngle;
    }

    _canTransitionToWander(distanceToTarget) {

        return distanceToTarget > this._fleeBehavior.panicDistance;
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._fleeBehavior = this.entity.getComponent(PsyanimFleeBehavior);
    }

    enter() {

        super.enter();

        this._recomputeSubtletyAngle();
        this._subtletyUpdateTimer = 0;

        this._computeDistanceToTarget();

        this.fsm.setStateVariable('avoidWalls', false);

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0x0000ff;
        }
    }

    exit() {

        super.exit();
    }

    _computeDistanceToTarget() {

        let distanceToTarget = this.target.position
            .subtract(this.entity.position)
            .length();

        this.fsm.setStateVariable('distanceToTarget', distanceToTarget);
    }

    _calculateFleeTargetPosition() {

        let targetRelativePosition = this.target.position
            .subtract(this.entity.position);

        targetRelativePosition.rotate(this._subtletyAngle * Math.PI / 180.0);

        return this.entity.position
            .add(targetRelativePosition);
    }

    _updateFleeTargetLocation() {

        let newTargetPosition = this._calculateFleeTargetPosition();

        while (!this.entity.scene.screenBoundary.isPointInBounds(newTargetPosition))
        {
            this._recomputeSubtletyAngle();
            newTargetPosition = this._calculateFleeTargetPosition();
        }

        this._fleeTarget.position = newTargetPosition;
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

        this._computeDistanceToTarget();

        let closesetWallDistance = Math.min(...(this.entity.computeDistancesToScreenBoundaries()));

        if (closesetWallDistance < this.minimumWallSeparation)
        {
            this.fsm.setStateVariable('avoidWalls', true);
        }

        this._updateSubtlety(dt);

        this._updateFleeTargetLocation();

        let steering = this._fleeBehavior.getSteering(this._fleeTarget);

        this._vehicle.steer(steering);
    }
}