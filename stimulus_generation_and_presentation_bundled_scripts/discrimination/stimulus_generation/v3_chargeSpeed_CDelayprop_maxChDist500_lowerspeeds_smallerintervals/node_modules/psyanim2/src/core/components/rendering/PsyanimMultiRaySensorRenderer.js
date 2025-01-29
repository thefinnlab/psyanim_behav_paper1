import Phaser from 'phaser';

import PsyanimComponent from "../../PsyanimComponent.js";

export default class PsyanimMultiRaySensorRenderer extends PsyanimComponent {

    raySensor;

    lineWidth;
    lineAlpha;

    normalLength;

    constructor(entity) {

        super(entity);

        this.lineWidth = 2;
        this.lineAlpha = 0.6;

        this.normalLength = 75;

        this._graphics = this.entity.scene.add.graphics();
    }

    afterCreate() {

        this._line = new Phaser.Geom.Line(0, 0, 0, 0);
        this._point = new Phaser.Geom.Point(0, 0);
    }

    _drawLine(startPoint, endPoint, color) {

        this._line.setTo(startPoint.x, startPoint.y, endPoint.x, endPoint.y);

        this._graphics.lineStyle(this.lineWidth, color, this.lineAlpha);

        this._graphics.strokeLineShape(this._line);
    }

    _drawPoint(point, color, size = 5) {

        this._point.setTo(point.x, point.y);

        this._graphics.fillStyle(color);

        this._graphics.fillPointShape(this._point, size);
    }

    update(t, dt) {

        super.update(t, dt);

        this._graphics.clear();

        for (let ray of this.raySensor.rayMap.values())
        {
            if (ray.collisions.length > 0)
            {
                this._drawLine(ray.startPoint, ray.endPoint, 0x00ff00);

                ray.collisions.forEach(collision => {

                    let intersectionPoint = collision.point;

                    this._drawPoint(intersectionPoint, 0x0000ff);

                    let normalEndPoint = new Phaser.Math.Vector2(intersectionPoint.x, intersectionPoint.y)
                        .add(new Phaser.Math.Vector2(collision.normal.x, collision.normal.y)
                            .scale(this.normalLength));

                    this._drawLine(intersectionPoint, normalEndPoint, 0xffa500);
                });
            }
            else
            {
                this._drawLine(ray.startPoint, ray.endPoint, 0xff0000);
            }
        }
    }
}