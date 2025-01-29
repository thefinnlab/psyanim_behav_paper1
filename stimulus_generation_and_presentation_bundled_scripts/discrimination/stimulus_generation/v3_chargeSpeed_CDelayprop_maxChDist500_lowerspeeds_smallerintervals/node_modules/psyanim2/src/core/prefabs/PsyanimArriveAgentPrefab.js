import PsyanimEntityPrefab from '../PsyanimEntityPrefab.js';

import PsyanimVehicle from '../components/steering/PsyanimVehicle.js';

import PsyanimArriveBehavior from '../components/steering/PsyanimArriveBehavior.js';
import PsyanimArriveAgent from '../components/steering/agents/PsyanimArriveAgent.js';

/**
 *  Prefab for creating `Arrive Agents`.
 * 
 *  An `Arrive Agent` seeks the target, slowing down before reaching it, eventually coming to rest.
 * 
 *  An `Arrive Agent` has the following components: 
 * 
 *  `PsyanimVehicle`, `PsyanimArriveBehavior`, `PsyanimArriveAgent`
 */
export default class PsyanimArriveAgentPrefab extends PsyanimEntityPrefab {

    /**
     *  Maximum speed this entity's vehicle can travel.
     *  @type {Number}
     */
    maxSpeed;

    /**
     *  Distance, in px, from target which agent will come to rest.
     *  @type {Number}
     */
    innerDecelerationRadius;

    /**
     *  Distance, in px, from target which the agent will begin slowing down.
     *  @type {Number}
     */
    outerDecelerationRadius;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.maxSpeed = 8;
        this.innerDecelerationRadius = 25;
        this.outerDecelerationRadius = 140;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);

        let arriveBehavior = entity.addComponent(PsyanimArriveBehavior);

        arriveBehavior.maxSpeed = this.maxSpeed;
        arriveBehavior.innerDecelerationRadius = this.innerDecelerationRadius;
        arriveBehavior.outerDecelerationRadius = this.outerDecelerationRadius;

        let arriveAgent = entity.addComponent(PsyanimArriveAgent);
        arriveAgent.arriveBehavior = arriveBehavior;
        arriveAgent.vehicle = vehicle;

        return entity;
    }
}