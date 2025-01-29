import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimObstacleAvoidanceBehavior extends PsyanimComponent {

    multiRaySensor;

    seekBehavior;
    maxSeekSpeed;
    maxSeekAcceleration;

    avoidDistance;

    constructor(entity) {

        super(entity);

        /**
         *  PsyanimObstacleAvoidanceBehavior should only be used on its own for avoiding 
         *  screen boundaries.  Use a pathfinder along with this for more complex geometry.
         */

        this.maxSeekSpeed = 5;
        this.maxSeekAcceleration = 0.2;
        this.avoidDistance = 25;

        this._seekTarget = this.scene.addEntity('_' + this.entity.name + '_seekTarget');
    }

    afterCreate() {

        super.afterCreate();
    }

    getSteering() {

        this.seekBehavior.maxSpeed = this.maxSeekSpeed;
        this.seekBehavior.maxAcceleration = this.maxSeekAcceleration;

        for (let ray of this.multiRaySensor.rayMap.values())
        {
            if (ray.collisions.length > 0)
            {
                let collision = ray.collisions[0];

                let targetPosition = new Phaser.Math.Vector2(
                    collision.point.x, collision.point.y)
                    .add(new Phaser.Math.Vector2(
                        collision.normal.x, collision.normal.y
                    ).scale(this.avoidDistance));

                this._seekTarget.position = targetPosition;

                return this.seekBehavior.getSteering(this._seekTarget);
            }
        }

        return Phaser.Math.Vector2.ZERO.clone();
    }

    _convertEntityAngleToVectorAngleD(entityAngleDegrees) {

        let newAngle = entityAngleDegrees;

        if (entityAngleDegrees < 0)
        {
            newAngle *= -1;
        }
        else
        {
            newAngle = 360 - newAngle;
        }

        return newAngle;
    }
}