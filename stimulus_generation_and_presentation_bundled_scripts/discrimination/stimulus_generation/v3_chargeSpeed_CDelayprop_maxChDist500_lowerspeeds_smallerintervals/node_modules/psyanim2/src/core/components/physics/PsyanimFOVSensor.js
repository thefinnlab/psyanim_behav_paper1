import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimFOVSensor extends PsyanimComponent {

    fovAngle; // degrees

    fovRange; // pixels

    resolution; // degrees

    constructor(entity) {

        super(entity);

        this.fovAngle = 120;
        this.fovRange = 200;
        this.resolution = 5;

        this._computeRayEndpoints();
    }

    get rayEndpoints() {

        return this._rayEndpoints;
    }

    computeRayEndpoint(angle) {

        let rayEndpoint = new Phaser.Math.Vector2(this.fovRange, 0);

        rayEndpoint.setAngle(angle * Math.PI / 180.0);

        rayEndpoint.add(this.entity.position);

        return rayEndpoint;
    }

    _computeRayAngles() {

        let startAngle = this.entity.angle - this.fovAngle / 2;

        let nRays = this.fovAngle / this.resolution + 1;

        let rayAngles = [];

        for (let i = 0; i < nRays; ++i)
        {
            rayAngles.push(startAngle + i * this.resolution);
        }

        return rayAngles;
    }

    _computeRayEndpoints() {

        this._rayEndpoints = [];

        let rayAngles = this._computeRayAngles();

        for (let i = 0; i < rayAngles.length; ++i)
        {
            let rayEndpoint = this.computeRayEndpoint(rayAngles[i]);

            this._rayEndpoints.push(rayEndpoint);
        }
    }

    getEntitiesInSight(entityList) {

        let bodyList = entityList.map(e => e.body);

        let entitiesInSight = [];

        let rayStartPoint = { x: this.entity.x, y: this.entity.y };

        for (let i = 0; i < this._rayEndpoints.length; ++i)
        {
            let rayEndpoint = this._rayEndpoints[i];

            let collisions = this.scene.matter.query.ray(
                bodyList, rayStartPoint, rayEndpoint);

            if (collisions && collisions.length != 0)
            {
                collisions.forEach(c => {

                    if (!entitiesInSight.includes(c.body.gameObject))
                    {
                        entitiesInSight.push(c.body.gameObject);
                    }
                });
            }
        }

        return entitiesInSight;
    }

    update(t, dt) {

        super.update(t, dt);

        this._computeRayEndpoints();
    }
}