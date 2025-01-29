import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimCircleRenderer extends PsyanimComponent {
    
    offset;
    angle;
    radius;

    constructor(entity) {

        super(entity);

        this.offset = 150;
        this.angle = 0;
        this.radius = 20;

        this.graphics = entity.scene.add.graphics({ lineStyle: { width: 2, color: 0xbf40bf, alpha: 0.6 } });

        this._circle = new Phaser.Geom.Circle(
            entity.x, 
            entity.y, 
            this.radius);
    }

    update(t, dt) {

        super.update(t, dt);

        let offsetVector = new Phaser.Math.Vector2(this.offset, 0);
        offsetVector.setAngle((this.entity.angle + this.angle) * Math.PI / 180);

        let newPosition = this.entity.position.add(offsetVector);

        this._circle.setPosition(
            newPosition.x, 
            newPosition.y);

        this.graphics.clear();

        this.graphics.strokeCircleShape(this._circle);
    }
}