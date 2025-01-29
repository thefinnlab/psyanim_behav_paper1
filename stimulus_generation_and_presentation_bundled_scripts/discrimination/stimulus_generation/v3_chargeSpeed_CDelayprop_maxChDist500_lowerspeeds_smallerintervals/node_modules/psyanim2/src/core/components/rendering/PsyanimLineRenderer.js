import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimLineRenderer extends PsyanimComponent {

    originPoint;
    endPoint;

    lineColor;
    lineWidth;

    alpha;

    constructor(entity) {

        super(entity);

        this.originPoint = new Phaser.Math.Vector2(0, 0);
        this.endPoint = new Phaser.Math.Vector2(100, 0);
    
        this.lineColor = 0xa020f0;
        this.lineWidth = 2;
    
        this.alpha = 0.6;

        this.graphics = entity.scene.add.graphics();

        this._line = new Phaser.Geom.Line(
            this.originPoint.x, this.originPoint.y,
            this.endPoint.x, this.endPoint.y);
    }

    onDisable() {

        super.onDisable();

        this.graphics.clear();
    }

    update(t, dt) {

        super.update(t, dt);

        this.graphics.clear();

        this.graphics.lineStyle(this.lineWidth, this.lineColor, this.alpha);

        this._line.setTo(
            this.originPoint.x, this.originPoint.y,
            this.endPoint.x, this.endPoint.y);

        this.graphics.strokeLineShape(this._line);
    }
}