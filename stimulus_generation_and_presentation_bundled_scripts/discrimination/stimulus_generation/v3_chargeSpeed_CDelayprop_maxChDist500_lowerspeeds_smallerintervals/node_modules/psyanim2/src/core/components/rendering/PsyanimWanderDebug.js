import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimWanderDebug extends PsyanimComponent {

    wanderBehavior;

    constructor(entity) {

        super(entity);
    }

    onEnable() {

        this._setupDebugGraphics();
    }

    onDisable() {

        this.graphics.destroy();
        this.graphics = null;
    }

    _setupDebugGraphics() {

        this.graphics = this.entity.scene.add.graphics({ 
            lineStyle: {
                width: 2, color: 0xa020f0, alpha: 0.6
            }
        });

        this._debugCircle = new Phaser.Geom.Circle(
            this.entity.x,
            this.entity.y,
            this.wanderBehavior.radius
        );

        this._debugCircleAngleLine = new Phaser.Geom.Line(
            this._debugCircle.x,
            this._debugCircle.y,
            this._debugCircle.x + this.wanderBehavior.radius,
            this._debugCircle.y
        );

        this._debugTargetLine = new Phaser.Geom.Line(
            0,
            0,
            this.entity.x,
            this.entity.y,
        )
    }

    _drawDebugGraphics() {

        // update render state
        this._debugCircle.radius = this.wanderBehavior.radius;

        this._debugCircle.setPosition(
            this.wanderBehavior._circleCenterVector.x,
            this.wanderBehavior._circleCenterVector.y
        );

        this._debugCircleAngleLine.setTo(
            this.wanderBehavior._circleCenterVector.x, 
            this.wanderBehavior._circleCenterVector.y,
            this.wanderBehavior._targetVector.x, 
            this.wanderBehavior._targetVector.y
        );

        this._debugTargetLine.setTo(
            this.entity.x, this.entity.y,
            this.wanderBehavior._targetVector.x, this.wanderBehavior._targetVector.y
        );

        // draw
        this.graphics.clear();

        this.graphics.strokeLineShape(this._debugCircleAngleLine);
        this.graphics.strokeLineShape(this._debugTargetLine);
        this.graphics.strokeCircleShape(this._debugCircle);
    }

    update(t, dt) {

        super.update(t, dt);

        if (this.enabled && this.graphics == null)
        {
            this._setupDebugGraphics();
        }

        this._drawDebugGraphics();
    }
}