import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimChargeBehavior extends PsyanimComponent {

    chargeDuration;
    innerDecelerationRadius;
    outerDecelerationRadius;

    constructor(entity) {

        super(entity);

        this.chargeDuration = 2.0;
        this.innerDecelerationRadius = 10;
        this.outerDecelerationRadius = 300;    

        this._chargeAcceleration = new Phaser.Math.Vector2(0, 0);
        this._maxChargeSpeed = new Phaser.Math.Vector2(0, 0);
    }

    computeChargeParams(target) {

        let t_ms = 1000 * this.chargeDuration;

        this._chargeAcceleration = target.position
            .subtract(this.entity.position)
            .scale(2 / (t_ms * t_ms));

        this._maxChargeSpeed = this.entity.velocity
            .scale(1/16.666) // convert to px/ms
            .add(this._chargeAcceleration.clone().scale(t_ms))
            .scale(16.666); // convert back to px/step

        // give it some initial velocity proportional to the distance so it's not so sluggish starting out... 
        // for starters, maybe we do a fraction of the average velocity as the initial velocity... 
        let v0 = target.position
            .subtract(this.entity.position)
            .scale(1 / (this.chargeDuration * 1000)) // avg velocity in px/ms
            .scale(16.666); // convert to px/step
        
        v0.scale(0.05); // take a fraction of avg. velocity as v0

        this.entity.setVelocity(v0.x, v0.y);
    }
    
    /**
     * 
     * NOTE: charge only works if the following settings are applied
     * 
     * this.entity.body.friction = 0;
     * this.entity.body.frictionAir = 0;
     * this.entity.body.frictionStatic = 0;
     * 
     * vehicle.useAcceleration = true;
     * 
     * @param {*} target 
     * @returns 
     */
    getSteering(target) {

        let currentPosition = new Phaser.Math.Vector2(this.entity.x, this.entity.y);

        let targetRelativePosition = new Phaser.Math.Vector2(target.x, target.y);
        targetRelativePosition.subtract(currentPosition);

        let r = targetRelativePosition.length();

        let desiredSpeed = 0;

        let scaledMaxAcceleration = this._chargeAcceleration;

        if (r <= this.innerDecelerationRadius)
        {
            this.entity.setVelocity(0, 0);

            return new Phaser.Math.Vector2(0, 0);
        }
        else if (r > this.outerDecelerationRadius)
        {
            return this._chargeAcceleration;
        }
        else // ((r > this.innerDecelerationRadius) && r < this.r2)
        {
            //  v(r) = ((v_max) / (r2 - r1)) * (r - r1)
            desiredSpeed = ((this._maxChargeSpeed) / (this.outerDecelerationRadius - this.innerDecelerationRadius)) * (r - this.innerDecelerationRadius);
            scaledMaxAcceleration = (r - this.innerDecelerationRadius) / (this.outerDecelerationRadius - this.innerDecelerationRadius) * this._chargeAcceleration.length();
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