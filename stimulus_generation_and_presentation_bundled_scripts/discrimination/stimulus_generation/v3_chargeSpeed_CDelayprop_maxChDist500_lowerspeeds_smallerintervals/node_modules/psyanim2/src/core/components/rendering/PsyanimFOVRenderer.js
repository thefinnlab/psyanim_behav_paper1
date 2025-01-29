import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimFOVRenderer extends PsyanimComponent {

    fovSensor;

    constructor(entity) {
        
        super(entity);

        this.graphics = entity.scene.add.graphics({ lineStyle: { width: 2, color: 0xa020f0, alpha: 0.6 } });

        this._leftRayLine = new Phaser.Geom.Line(0, 0, 0, 0);
        this._rightRayLine = new Phaser.Geom.Line(0, 0, 0, 0);
        this._centerRayLine = new Phaser.Geom.Line(0, 0, 0, 0);
    }

    update(t,dt) {

        super.update(t, dt);

        // compute ray endpoints
        let nRays = this.fovSensor.rayEndpoints.length;

        let centerRayEndpoint = this.fovSensor.computeRayEndpoint(this.entity.angle);
        let leftRayEndpoint = this.fovSensor.rayEndpoints[0];
        let rightRayEndpoint = this.fovSensor.rayEndpoints[nRays - 1];

        // update line geometry
        this._centerRayLine.setTo(
            this.entity.x, this.entity.y,
            centerRayEndpoint.x, centerRayEndpoint.y
        );

        this._leftRayLine.setTo(
            this.entity.x, this.entity.y,
            leftRayEndpoint.x, leftRayEndpoint.y
        );

        this._rightRayLine.setTo(
            this.entity.x, this.entity.y,
            rightRayEndpoint.x, rightRayEndpoint.y
        );

        // draw graphics
        this.graphics.clear();

        this.graphics.strokeLineShape(this._centerRayLine);
        this.graphics.strokeLineShape(this._leftRayLine);
        this.graphics.strokeLineShape(this._rightRayLine);
    }
}