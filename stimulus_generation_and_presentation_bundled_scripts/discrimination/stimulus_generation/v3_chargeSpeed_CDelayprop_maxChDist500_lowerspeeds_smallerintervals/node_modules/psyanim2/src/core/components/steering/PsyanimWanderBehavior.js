import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

import { PsyanimUtils } from 'psyanim-utils';

export default class PsyanimWanderBehavior extends PsyanimComponent {

    radius;
    offset;

    minScreenBoundaryDistance;

    maxAngleChangePerFrame;

    seekBehavior;
    maxSeekSpeed;
    maxSeekAcceleration;

    constructor(entity) {

        super(entity);

        this.maxSeekSpeed = 5;
        this.maxSeekAcceleration = 0.2;

        this.radius = 50;
        this.offset = 150;
    
        this.minScreenBoundaryDistance = 50;
    
        this.maxAngleChangePerFrame = 20;    

        this._debug = false;

        this._angle = PsyanimUtils.getRandomInt(-180, 180);
        this.entity.angle = this._angle;

        this._targetVector = null;

        this.target = this.entity.scene.addEntity("_" + this.entity.name + '_wanderTarget', 0, 0, { isEmpty: true });

        this._circleCenterVector = Phaser.Math.Vector2.ZERO.clone();
        this._targetVector = Phaser.Math.Vector2.ZERO.clone();

        this.screenBoundaries = [
            this.entity.scene.screenBoundary.topBoundary.body,
            this.entity.scene.screenBoundary.bottomBoundary.body,
            this.entity.scene.screenBoundary.leftBoundary.body,
            this.entity.scene.screenBoundary.rightBoundary.body,
        ];
    }

    _isTargetOffScreen() {

        let startVector = this.entity.position;
        let endVector = this._targetVector.clone()
            .subtract(this.entity.position)
            .setLength(this.minScreenBoundaryDistance)
            .add(this.entity.position);

        let start = { x: startVector.x, y: startVector.y };
        let end = { x: endVector.x, y: endVector.y };

        let collisions = this.entity.scene.matter
            .query.ray(this.screenBoundaries, start, end);

        if (collisions && collisions.length != 0)
        {
            return true;
        }

        return false;
    }

    _updateTargetVector()
    {
        // update target vector
        let offsetVector = new Phaser.Math.Vector2(this.offset, 0);
        offsetVector.setAngle(this.entity.rotation);

        this._circleCenterVector = this.entity.position.add(offsetVector);

        this._targetVector = this.entity.forward.setLength(this.radius);
        this._targetVector.rotate(this._angle * Math.PI / 180);
        this._targetVector.add(this._circleCenterVector);
    }

    getSteering() {

        // update params
        this.seekBehavior.maxSpeed = this.maxSeekSpeed;
        this.seekBehavior.maxAcceleration = this.maxSeekAcceleration;

        // compute angle change
        this._angle += (Math.random() * 2 - 1) * this.maxAngleChangePerFrame;

        if (this._angle > 360)
        {
            this._angle - 360;
        }
        else if (this._angle < 0)
        {
            this._angle + 360;
        }

        this._updateTargetVector();

        if (this._isTargetOffScreen())
        {
            let entityAngle = this.entity.angle;
            entityAngle += 180;

            if (entityAngle > 360)
            {
                entityAngle - 360;
            }
            else if (entityAngle < 0)
            {
                entityAngle + 360;
            }

            this.entity.setAngle(entityAngle);

            this._updateTargetVector();
        }

        this.target.position = this._targetVector;

        return this.seekBehavior.getSteering(this.target);
    }
}