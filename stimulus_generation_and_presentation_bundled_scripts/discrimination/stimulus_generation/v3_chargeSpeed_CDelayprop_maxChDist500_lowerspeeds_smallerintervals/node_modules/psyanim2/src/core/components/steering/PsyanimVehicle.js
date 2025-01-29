import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimVehicle extends PsyanimComponent {

    /**
     *  NOTES: 
     * 
     *  Default params for friction are:
     * 
     *      this.entity.body.friction = 0.1;
     *      this.entity.body.frictionAir = 0.01;
     *      this.entity.body.frictionStatic = 0.5;
     * 
     */

    maxSpeed;

    turnSpeed;

    smoothLookDirection;

    nSamplesForLookSmoothing;

    constructor(entity) {

        super(entity);

        this.maxSpeed = 5;

        this.turnSpeed = 0.2;
    
        this.smoothLookDirection = true;
    
        this.nSamplesForLookSmoothing = 6;    

        this._velocitySamples = [];
    }

    onDisable() {

        this.entity.setVelocity(0, 0);
        this.entity.body.force = { x: 0, y: 0 };
    }

    lookWhereYoureGoing() {

        let velocity = this.entity.velocity;

        let direction = new Phaser.Math.Vector2(0, 0);

        if (this.smoothLookDirection)
        {
            if (this._velocitySamples.length == this.nSamplesForLookSmoothing)
            {
                this._velocitySamples.shift();
            }

            this._velocitySamples.push(velocity);

            for (let i = 0; i < this._velocitySamples.length; ++i)
            {
                direction.add(this._velocitySamples[i]);
            }
    
            direction.scale(1 / this._velocitySamples.length);    
        }
        else
        {
            direction = velocity;
        }

        if (direction.length() > 1e-3)
        {
            direction.normalize();

            let targetAngle = Math.atan2(direction.y, direction.x) * 180 / Math.PI;

            if (this.turnSpeed == Infinity)
            {
                this.entity.setAngle(targetAngle);
            }
            else
            {
                let lerpedAngle = Phaser.Math.Angle.RotateTo(
                    this.entity.angle * Math.PI / 180,
                    targetAngle,
                    this.turnSpeed);
    
                this.entity.setAngle(lerpedAngle);                    
            }
        }
    }

    steer(steering) {

        // clamp velocity to max speed
        let velocity = new Phaser.Math.Vector2(this.entity.getVelocity());

        if (velocity.length() > this.maxSpeed)
        {
            velocity.setLength(this.maxSpeed);

            this.entity.setVelocity(velocity.x, velocity.y);    
        }

        // apply steering
        this.entity.applyForce(steering);

        this.lookWhereYoureGoing();
    }
}