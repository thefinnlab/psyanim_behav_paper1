import PsyanimFSMState from "../PsyanimFSMState.js";

import PsyanimVehicle from "../../steering/PsyanimVehicle.js";
import PsyanimSeekBehavior from "../../steering/PsyanimSeekBehavior.js";
import PsyanimFleeBehavior from "../../steering/PsyanimFleeBehavior.js";

import PsyanimPreyWanderState from './PsyanimPreyWanderState.js';

import { PsyanimUtils } from 'psyanim-utils';

export default class PsyanimPreyWallAvoidanceState extends PsyanimFSMState {

    target;

    subtlety;
    subtletyLag;

    seekTargetStoppingDistance;

    seekTargetLocations;

    constructor(fsm) {

        super(fsm);

        this.subtlety = 30; // degrees
        this.subtletyLag = 500; // ms

        this.seekTargetStoppingDistance = 50;

        this.seekTargetLocations = [
            { x: 400, y: 450 }, // bottom middle
            { x: 150, y: 450 }, // bottom left
            { x: 650, y: 450 }, // bottom right
            { x: 400, y: 150 }, // top middle
            { x: 150, y: 150 }, // top left
            { x: 650, y: 150 }, // top right
            { x: 150, y: 300 }, // left middle
            { x: 650, y: 300 }, // right middle
        ];

        this._subtletyAngle = 0;
        this._subtletyUpdateTimer = 0;

        this.fsm.setStateVariable('wander', false);

        this._canvasHeight = this.entity.scene.game.config.height;
        this._canvasWidth = this.entity.scene.game.config.width;

        this._seekTarget = this.entity.scene.addEntity(this.entity.name + '_wallAvoidanceState');

        this.addTransition(PsyanimPreyWanderState, 'wander', (value) => value === true);
    }

    _recomputeSubtletyAngle() {

        let newAngle = this.subtlety * (2 * Math.random() - 1);

        if (Math.abs(newAngle) < 0.001)
        {
            newAngle = 0;
        }

        this._subtletyAngle = newAngle;
    }

    afterCreate() {

        super.afterCreate();

        this._vehicle = this.entity.getComponent(PsyanimVehicle);
        this._fleeBehavior = this.entity.getComponent(PsyanimFleeBehavior);
        this._seekBehavior = this.entity.getComponent(PsyanimSeekBehavior);
    }

    enter() {

        super.enter();

        this.fsm.setStateVariable('wander', false);

        this._recomputeSubtletyAngle();
        this._subtletyUpdateTimer = 0;

        this._seekBehavior.maxSpeed = this._fleeBehavior.maxSpeed;
        this._seekBehavior.maxAcceleration = this._fleeBehavior.maxAcceleration;

        let index = PsyanimUtils.getRandomInt(0, this.seekTargetLocations.length - 1);

        this.currentSeekTargetLocation = new Phaser.Math.Vector2(
            this.seekTargetLocations[index].x,
            this.seekTargetLocations[index].y);

        this._seekTarget.position = this.currentSeekTargetLocation;

        if (this.fsm.debugGraphics)
        {
            this.entity.color = 0xffff00;
        }
    }

    exit() {

        super.exit();
    }

    _canTransitionToWander() {

        let distanceToSeekTarget = this.entity.position
            .subtract(this.currentSeekTargetLocation)
            .length();

        let distanceToTarget = this.entity.position
            .subtract(this.target.position)
            .length();

        let hasReachedSeekTarget = distanceToSeekTarget < this.seekTargetStoppingDistance;

        let surroundingAreaIsSafe = distanceToTarget > this._fleeBehavior.panicDistance;

        return (hasReachedSeekTarget || surroundingAreaIsSafe);
    }

    _calculateSeekTargetLocation() {

        let targetRelativePosition = this.currentSeekTargetLocation.clone()
            .subtract(this.entity.position);

        targetRelativePosition.rotate(this._subtletyAngle * Math.PI / 180.0);

        return this.entity.position
            .add(targetRelativePosition);

    }

    _updateSeekTargetLocation() {

        let newTargetPosition = this._calculateSeekTargetLocation();

        while (!this.entity.scene.screenBoundary.isPointInBounds(newTargetPosition))
        {
            this._recomputeSubtletyAngle();
            newTargetPosition = this._calculateSeekTargetLocation();
        }

        this._seekTarget.position = newTargetPosition;
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

        let steering = this._seekBehavior.getSteering(this._seekTarget);

        if (this._canTransitionToWander())
        {
            this.fsm.setStateVariable('wander', true);
        }
        else
        {
            this._vehicle.steer(steering);
        }
    }
}