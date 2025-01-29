import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimCollisionAvoidanceDebug extends PsyanimComponent {

    collisionAvoidanceBehavior;

    constructor(entity) {

        super(entity);
        
        this.graphics = entity.scene.add.graphics({ lineStyle: { width: 2, color: 0x00ff00, alpha: 1.0 } });

        this._line = new Phaser.Geom.Line(0, 0, 0, 0);
        this._circle = new Phaser.Geom.Circle(0, 0, 4);
    }

    update(t, dt) {

        super.update(t, dt);

        this.graphics.clear();

        let _collisionAvoidanceTarget = this.collisionAvoidanceBehavior._collisionAvoidanceTarget;

        if (_collisionAvoidanceTarget == null)
        {
            return;
        }

        let targetPosition = _collisionAvoidanceTarget.target.position;
        let predictedPosition = this.entity.position.add(_collisionAvoidanceTarget.finalRelativePosition);

        this._circle.setPosition(
            predictedPosition.x, predictedPosition.y
        );

        this._line.setTo(
            targetPosition.x, targetPosition.y,
            predictedPosition.x, predictedPosition.y
        );

        this.graphics.strokeCircleShape(this._circle);
        this.graphics.strokeLineShape(this._line);
    }
}