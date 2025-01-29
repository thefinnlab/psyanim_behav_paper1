import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';
import PsyanimConstants from '../../PsyanimConstants.js';
import PsyanimSensor from '../physics/PsyanimSensor.js';

export default class PsyanimCollisionAvoidanceBehavior extends PsyanimComponent {

    collisionRadius;
    maxAcceleration;

    constructor(entity) {

        super(entity);

        this.collisionRadius = 8;
        this.maxAcceleration = 0.2;    

        this._collisionAvoidanceTarget = null;

        this._nearbyAgents = [];

        this.sensor = this.entity.addComponent(PsyanimSensor);

        this.sensor.events.on('triggerEnter', (entity) => {

            this._nearbyAgents.push(entity);
        });

        this.sensor.events.on('triggerExit', (entity) => {

            this._nearbyAgents = this._nearbyAgents.filter(e => {
                e.name != entity.name;
            });
        });
    }

    setSensorRadius(radius) {

        this.sensor.setBody({
            shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
            radius: radius
        });
    }

    getSteering() {

        /**
         *  Algorithm here is lifted mostly straight from 'Artificial Inteligence for Games'
         *  by Ian Millington, 3rd ed.
         */

        if (this._nearbyAgents.length == 0)
        {
            this._collisionAvoidanceTarget = null;
            return new Phaser.Math.Vector2(0, 0);
        }

        let shortestTime = Infinity;
        let firstTarget = null;
        let firstMinSeparation = 0;
        let firstDistance = 0;
        let firstRelativePosition = null;
        let firstRelativeVelocity = null;

        // find target for collision avoidance
        for (let i = 0; i < this._nearbyAgents.length; ++i)
        {
            let agent = this._nearbyAgents[i];
            let relativePosition = agent.position.subtract(this.entity.position);
            let relativeVelocity = agent.velocity.subtract(this.entity.velocity);
            let relativeSpeed = relativeVelocity.length();

            let timeToCollision = -1 * relativePosition.dot(relativeVelocity) / (relativeSpeed * relativeSpeed);

            let distance = relativePosition.length();

            let minSeparation = relativeVelocity.clone()
                .scale(timeToCollision)
                .add(relativePosition)
                .length();

            if (minSeparation > 2 * this.collisionRadius)
            {
                continue;
            }

            if (timeToCollision > 0 && timeToCollision < shortestTime)
            {
                shortestTime = timeToCollision;
                firstTarget = agent;
                firstMinSeparation = minSeparation;
                firstDistance = distance;
                firstRelativePosition = relativePosition;
                firstRelativeVelocity = relativeVelocity;
            }
        }

        // calculate steering
        if (firstTarget == null)
        {
            this._collisionAvoidanceTarget = null;

            return new Phaser.Math.Vector2(0, 0);
        }

        let finalRelativePosition = null;

        if (firstMinSeparation <= 0 || firstDistance < 2 * this.collisionRadius)
        {
            finalRelativePosition = firstTarget.position.subtract(this.entity.position);
        }
        else
        {
            finalRelativePosition = firstRelativePosition.add(firstRelativeVelocity.scale(shortestTime));
        }

        this._collisionAvoidanceTarget = {
            target: firstTarget,
            finalRelativePosition: finalRelativePosition.clone()
        };

        finalRelativePosition.normalize();

        let steering = finalRelativePosition.scale(-1 * this.maxAcceleration);

        return steering;
    }
}