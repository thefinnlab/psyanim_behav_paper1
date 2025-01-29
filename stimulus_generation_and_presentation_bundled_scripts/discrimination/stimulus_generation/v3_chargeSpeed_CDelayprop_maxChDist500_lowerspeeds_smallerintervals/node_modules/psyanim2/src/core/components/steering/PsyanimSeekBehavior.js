import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimSeekBehavior extends PsyanimComponent {

    maxSpeed;
    maxAcceleration;

    constructor(entity) {

        super(entity);

        this.maxSpeed = 5;
        this.maxAcceleration = 0.2;    
    }

    getSteering(target) {

        let currentPosition = new Phaser.Math.Vector2(this.entity.x, this.entity.y);

        let desiredVelocity = new Phaser.Math.Vector2(target.x, target.y);
        desiredVelocity.subtract(currentPosition);
        desiredVelocity.setLength(this.maxSpeed);

        let currentVelocityXY = this.entity.getVelocity();

        let currentVelocity = new Phaser.Math.Vector2(currentVelocityXY.x, currentVelocityXY.y);

        let acceleration = desiredVelocity.clone();
        acceleration.subtract(currentVelocity);

        if (acceleration.length() > this.maxAcceleration)
        {
            acceleration.setLength(this.maxAcceleration);
        }

        return acceleration;
    }
}