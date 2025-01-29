import Phaser from 'phaser';

import PsyanimComponent from "../../PsyanimComponent.js";

export default class PsyanimPreciselyTimedArriveBehavior extends PsyanimComponent {

    chargeDuration;

    maxAcceleration;

    innerDecelerationRadius;
    outerDecelerationRadius;

    constructor(entity) {

        super(entity);

        this.chargeDuration = 2.0;

        this.maxAcceleration = 10;
    
        this.innerDecelerationRadius = 10;
        this.outerDecelerationRadius = 150;

        this._maxSpeed = 0;
    }

    get maxSpeed() {

        return this._maxSpeed;
    }

    /**
     * Recomputes max speed as average speed required to reach target within 'chargeDuration'
     * @param {*} target 
     */
    computeMaxSpeed(target) {

        let distanceToTarget = this.entity.position
            .subtract(target.position)
            .length();

        if (distanceToTarget <= 0.0)
        {
            console.error("ERROR: distance to target = " + distanceToTarget);
        }

        // convert charge duration to simulation 'steps' instead of 'seconds'
        let chargeDurationSteps = (this.chargeDuration * 1000) / 16.666;

        this._maxSpeed = distanceToTarget / chargeDurationSteps;
    }

    getSteering(target) {

        let currentPosition = new Phaser.Math.Vector2(this.entity.x, this.entity.y);

        let targetRelativePosition = new Phaser.Math.Vector2(target.x, target.y);
        targetRelativePosition.subtract(currentPosition);

        let r = targetRelativePosition.length();

        let desiredSpeed = 0;

        let scaledMaxAcceleration = this.maxAcceleration;

        if (r <= this.innerDecelerationRadius)
        {
            this.entity.setVelocity(0, 0);

            return new Phaser.Math.Vector2(0, 0);
        }
        else if (r > this.outerDecelerationRadius)
        {
            desiredSpeed = this._maxSpeed;
        }
        else // ((r > this.innerDecelerationRadius) && r < this.r2)
        {
            //  v(r) = ((v_max) / (r2 - r1)) * (r - r1)
            desiredSpeed = ((this._maxSpeed) / (this.outerDecelerationRadius - this.innerDecelerationRadius)) * (r - this.innerDecelerationRadius);
            scaledMaxAcceleration = (r - this.innerDecelerationRadius) / (this.outerDecelerationRadius - this.innerDecelerationRadius) * this.maxAcceleration;
        }

        let desiredVelocity = targetRelativePosition.clone();

        desiredVelocity.setLength(desiredSpeed);

        let currentVelocityXY = this.entity.getVelocity();

        let currentVelocity = new Phaser.Math.Vector2(currentVelocityXY.x, currentVelocityXY.y);

        let acceleration = desiredVelocity.clone();
        acceleration.subtract(currentVelocity);

        if (acceleration.length() > scaledMaxAcceleration)
        {
            acceleration.setLength(scaledMaxAcceleration);
        }

        return acceleration;
    }
}