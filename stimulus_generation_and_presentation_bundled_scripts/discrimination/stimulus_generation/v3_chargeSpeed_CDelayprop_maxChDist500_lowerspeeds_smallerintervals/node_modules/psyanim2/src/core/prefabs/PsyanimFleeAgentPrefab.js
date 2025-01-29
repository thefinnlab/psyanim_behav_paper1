import PsyanimEntityPrefab from '../PsyanimEntityPrefab.js';

import PsyanimVehicle from '../components/steering/PsyanimVehicle.js';
import PsyanimFleeBehavior from '../components/steering/PsyanimFleeBehavior.js';
import PsyanimFleeAgent from '../components/steering/agents/PsyanimFleeAgent.js';

export default class PsyanimFleeAgentPrefab extends PsyanimEntityPrefab {

    maxSpeed;
    maxAcceleration;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.maxSpeed = 6;
        this.maxAcceleration = 0.2;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);

        let flee = entity.addComponent(PsyanimFleeBehavior);

        flee.maxSpeed = this.maxSpeed;
        flee.maxAcceleration = this.maxAcceleration;

        let fleeAgent = entity.addComponent(PsyanimFleeAgent);

        fleeAgent.fleeBehavior = flee;
        fleeAgent.vehicle = vehicle;

        return entity;
    }
}